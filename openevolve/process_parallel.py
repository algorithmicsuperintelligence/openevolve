"""
Process-based parallel controller for true parallelism
"""

import asyncio
import hashlib
import json
import logging
import multiprocessing as mp
import pickle
import signal
import time
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.evaluation_result import EVALUATION_FAILED_METRIC
from openevolve.utils.metrics_utils import safe_numeric_average

logger = logging.getLogger(__name__)


@dataclass
class SerializableResult:
    """Result that can be pickled and sent between processes"""

    child_program_dict: Optional[Dict[str, Any]] = None
    parent_id: Optional[str] = None
    iteration_time: float = 0.0
    prompt: Optional[Dict[str, str]] = None
    llm_response: Optional[str] = None
    llm_metadata: Optional[Dict[str, Any]] = None
    artifacts: Optional[Dict[str, Any]] = None
    iteration: int = 0
    error: Optional[str] = None
    target_island: Optional[int] = None  # Island where child should be placed


def _worker_init(config_dict: dict, evaluation_file: str, parent_env: dict = None) -> None:
    """Initialize worker process with necessary components"""
    import os

    # Set environment from parent process
    if parent_env:
        os.environ.update(parent_env)

    global _worker_config
    global _worker_evaluation_file
    global _worker_evaluator
    global _worker_llm_ensemble
    global _worker_prompt_sampler

    # Store config for later use
    # Reconstruct Config object from nested dictionaries
    from openevolve.config import (
        Config,
        ControllerSchedulerConfig,
        DatabaseConfig,
        EvaluatorConfig,
        LLMConfig,
        LLMModelConfig,
        PromptConfig,
    )

    # Reconstruct model objects
    models = [LLMModelConfig(**m) for m in config_dict["llm"]["models"]]
    evaluator_models = [LLMModelConfig(**m) for m in config_dict["llm"]["evaluator_models"]]

    # Create LLM config with models
    llm_dict = config_dict["llm"].copy()
    llm_dict["models"] = models
    llm_dict["evaluator_models"] = evaluator_models
    llm_config = LLMConfig(**llm_dict)

    # Create other configs
    prompt_config = PromptConfig(**config_dict["prompt"])
    database_dict = config_dict["database"].copy()
    scheduler = database_dict.get("controller_scheduler")
    if isinstance(scheduler, dict):
        database_dict["controller_scheduler"] = ControllerSchedulerConfig(**scheduler)
    database_config = DatabaseConfig(**database_dict)
    evaluator_config = EvaluatorConfig(**config_dict["evaluator"])

    _worker_config = Config(
        llm=llm_config,
        prompt=prompt_config,
        database=database_config,
        evaluator=evaluator_config,
        **{
            k: v
            for k, v in config_dict.items()
            if k not in ["llm", "prompt", "database", "evaluator"]
        },
    )
    _worker_evaluation_file = evaluation_file

    # These will be lazily initialized on first use
    _worker_evaluator = None
    _worker_llm_ensemble = None
    _worker_prompt_sampler = None


def _lazy_init_worker_components():
    """Lazily initialize expensive components on first use"""
    global _worker_evaluator
    global _worker_llm_ensemble
    global _worker_prompt_sampler

    if _worker_llm_ensemble is None:
        from openevolve.llm.ensemble import LLMEnsemble

        _worker_llm_ensemble = LLMEnsemble(_worker_config.llm.models)

    if _worker_prompt_sampler is None:
        from openevolve.prompt.sampler import PromptSampler

        _worker_prompt_sampler = PromptSampler(_worker_config.prompt)

    if _worker_evaluator is None:
        from openevolve.evaluator import Evaluator
        from openevolve.llm.ensemble import LLMEnsemble
        from openevolve.prompt.sampler import PromptSampler

        # Create evaluator-specific components
        evaluator_llm = LLMEnsemble(_worker_config.llm.evaluator_models)
        evaluator_prompt = PromptSampler(_worker_config.prompt)
        evaluator_prompt.set_templates("evaluator_system_message")

        _worker_evaluator = Evaluator(
            _worker_config.evaluator,
            _worker_evaluation_file,
            evaluator_llm,
            evaluator_prompt,
            database=None,  # No shared database in worker
            suffix=getattr(_worker_config, "file_suffix", ".py"),
        )


def _with_archive_context(
    parent_artifacts: Optional[Dict[str, Any]],
    db_snapshot: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Add evaluator-defined complete search history to one prompt."""

    archive_artifact = _worker_config.prompt.archive_context_artifact
    neighborhood_artifact = (
        _worker_config.prompt.proposal_neighborhood_artifact
    )
    if archive_artifact is None and neighborhood_artifact is None:
        return parent_artifacts
    augmented = dict(parent_artifacts or {})
    snapshot_artifacts = db_snapshot.get("artifacts")
    historical_artifacts = db_snapshot.get("historical_artifact_values")
    if archive_artifact is not None:
        live_values = _snapshot_artifact_values(
            snapshot_artifacts,
            archive_artifact,
        )
        historical_values = _historical_artifact_values(
            historical_artifacts,
            archive_artifact,
        )
        values = live_values | historical_values
        ordered = sorted(values)
        limit = _worker_config.prompt.archive_context_max_items
        selected = ordered[:limit]
        context = {
            "artifact": archive_artifact,
            "archive_item_count": len(ordered),
            "live_archive_item_count": len(live_values),
            "historical_item_count": len(historical_values),
            "included_item_count": len(selected),
            "items": selected,
            "truncated": len(selected) != len(ordered),
        }
        augmented["archive-context.json"] = json.dumps(
            context,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    if neighborhood_artifact is not None:
        raw_neighborhood = augmented.get(neighborhood_artifact)
        if isinstance(raw_neighborhood, bytes):
            raw_neighborhood = raw_neighborhood.decode(
                "utf-8", errors="strict"
            )
        if not isinstance(raw_neighborhood, str):
            raise ValueError(
                "parent is missing configured proposal neighborhood artifact: "
                f"{neighborhood_artifact}"
            )
        try:
            neighborhood = json.loads(raw_neighborhood)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"proposal neighborhood artifact is not valid JSON: {error}"
            ) from error
        if (
            not isinstance(neighborhood, dict)
            or neighborhood.get("schema_version") != 1
        ):
            raise ValueError(
                "proposal neighborhood artifact must use schema_version 1"
            )
        identity_artifact = neighborhood.get("identity_artifact")
        options = neighborhood.get("options")
        if (
            not isinstance(identity_artifact, str)
            or not identity_artifact.strip()
            or not isinstance(options, list)
        ):
            raise ValueError(
                "proposal neighborhood must declare identity_artifact and options"
            )
        retained = _snapshot_artifact_values(
            snapshot_artifacts,
            identity_artifact,
        )
        historical = _historical_artifact_values(
            historical_artifacts,
            identity_artifact,
        )
        known = retained | historical
        available: list[dict[str, Any]] = []
        seen: set[str] = set()
        excluded_retained = 0
        excluded_historical = 0
        excluded_known = 0
        for option in options:
            if not isinstance(option, dict):
                raise ValueError(
                    "proposal neighborhood options must be JSON objects"
                )
            identity = option.get("identity")
            option_id = option.get("id")
            if (
                not isinstance(identity, str)
                or not identity
                or not isinstance(option_id, str)
                or not option_id
            ):
                raise ValueError(
                    "proposal neighborhood options require non-empty id and identity"
                )
            if identity in seen:
                continue
            seen.add(identity)
            if identity in known:
                excluded_known += 1
                if identity in retained:
                    excluded_retained += 1
                if identity in historical:
                    excluded_historical += 1
                continue
            available.append(option)
        limit = _worker_config.prompt.proposal_options_max_items
        selected_options = available[:limit]
        proposal_context = {
            "source_artifact": neighborhood_artifact,
            "identity_artifact": identity_artifact,
            "declared_option_count": len(options),
            "excluded_retained_count": excluded_retained,
            "excluded_historical_count": excluded_historical,
            "excluded_known_count": excluded_known,
            "available_option_count": len(available),
            "included_option_count": len(selected_options),
            "options": selected_options,
            "truncated": len(selected_options) != len(available),
        }
        augmented["proposal-options.json"] = json.dumps(
            proposal_context,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    return augmented


def _snapshot_artifact_values(
    snapshot_artifacts: Any,
    artifact_name: str,
) -> set[str]:
    """Return complete text values for one artifact across a worker snapshot."""

    values: set[str] = set()
    if not isinstance(snapshot_artifacts, dict):
        return values
    for program_id in sorted(snapshot_artifacts):
        artifacts = snapshot_artifacts.get(program_id)
        if not isinstance(artifacts, dict):
            continue
        value = artifacts.get(artifact_name)
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if isinstance(value, str):
            values.add(value)
    return values


def _historical_artifact_values(
    historical_artifacts: Any,
    artifact_name: str,
) -> set[str]:
    """Return persisted text values for one append-only artifact ledger."""

    if not isinstance(historical_artifacts, dict):
        return set()
    values = historical_artifacts.get(artifact_name)
    if not isinstance(values, list):
        return set()
    return {value for value in values if isinstance(value, str)}


def _run_iteration_worker(
    iteration: int, db_snapshot: Dict[str, Any], parent_id: str, inspiration_ids: List[str]
) -> SerializableResult:
    """Run a single iteration in a worker process"""
    prompt: Optional[Dict[str, str]] = None
    llm_response: Optional[str] = None
    llm_metadata: Optional[Dict[str, Any]] = None
    try:
        # Lazy initialization
        _lazy_init_worker_components()

        # Reconstruct programs from snapshot
        programs = {pid: Program(**prog_dict) for pid, prog_dict in db_snapshot["programs"].items()}

        parent = programs[parent_id]
        inspirations = [programs[pid] for pid in inspiration_ids if pid in programs]

        # Get parent artifacts if available
        parent_artifacts = _with_archive_context(
            db_snapshot["artifacts"].get(parent_id),
            db_snapshot,
        )

        # Get island-specific programs for context
        parent_island = parent.metadata.get("island", db_snapshot["current_island"])
        island_programs = [
            programs[pid] for pid in db_snapshot["islands"][parent_island] if pid in programs
        ]

        # Sort by metrics for top programs
        island_programs.sort(
            key=lambda p: p.metrics.get("combined_score", safe_numeric_average(p.metrics)),
            reverse=True,
        )

        # Use config values for limits instead of hardcoding
        # Programs for LLM display (includes both top and diverse for inspiration)
        programs_for_prompt = island_programs[
            : _worker_config.prompt.num_top_programs + _worker_config.prompt.num_diverse_programs
        ]
        # Best programs only (for previous attempts section, focused on top performers)
        best_programs_only = island_programs[: _worker_config.prompt.num_top_programs]

        # Build prompt
        if _worker_config.prompt.programs_as_changes_description:
            parent_changes_desc = (
                parent.changes_description or _worker_config.prompt.initial_changes_description
            )
            child_changes_desc = parent_changes_desc
        else:
            parent_changes_desc = None
            child_changes_desc = None

        prompt = _worker_prompt_sampler.build_prompt(
            current_program=parent.code,
            parent_program=parent.code,
            program_metrics=parent.metrics,
            previous_programs=[p.to_dict() for p in best_programs_only],
            top_programs=[p.to_dict() for p in programs_for_prompt],
            inspirations=[p.to_dict() for p in inspirations],
            language=_worker_config.language,
            evolution_round=iteration,
            diff_based_evolution=_worker_config.diff_based_evolution,
            program_artifacts=parent_artifacts,
            feature_dimensions=db_snapshot.get("feature_dimensions", []),
            current_changes_description=parent_changes_desc,
        )

        iteration_start = time.time()

        # Generate code modification (sync wrapper for async)
        try:
            llm_response = asyncio.run(
                _worker_llm_ensemble.generate_with_context(
                    system_message=prompt["system"],
                    messages=[{"role": "user", "content": prompt["user"]}],
                )
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            llm_metadata = dict(
                getattr(_worker_llm_ensemble, "last_call_metadata", {}) or {}
            ) or None
            return SerializableResult(
                error=f"LLM generation failed: {str(e)}",
                iteration=iteration,
                prompt=prompt,
                llm_metadata=llm_metadata,
            )

        llm_metadata = dict(
            getattr(_worker_llm_ensemble, "last_call_metadata", {}) or {}
        ) or None
        # Check for None response
        if llm_response is None:
            return SerializableResult(
                error="LLM returned None response",
                iteration=iteration,
                prompt=prompt,
                llm_metadata=llm_metadata,
            )

        def reject_proposal(message: str) -> SerializableResult:
            return SerializableResult(
                error=message,
                iteration=iteration,
                parent_id=parent.id,
                prompt=prompt,
                llm_response=llm_response,
                llm_metadata=llm_metadata,
                target_island=db_snapshot.get("sampling_island"),
            )

        # Parse response based on evolution mode
        if _worker_config.diff_based_evolution:
            from openevolve.utils.code_utils import (
                apply_diff,
                apply_diff_blocks_strict,
                apply_diff_strict,
                apply_diff_blocks,
                extract_diffs,
                format_diff_summary,
                split_diffs_by_target,
            )

            diff_blocks = extract_diffs(llm_response, _worker_config.diff_pattern)
            if not diff_blocks:
                return reject_proposal("No valid diffs found in response")

            if _worker_config.prompt.programs_as_changes_description:
                try:
                    code_blocks, desc_blocks, unmatched = split_diffs_by_target(
                        diff_blocks,
                        code_text=parent.code,
                        changes_description_text=parent_changes_desc,
                    )
                except Exception as e:
                    return reject_proposal(str(e))

                if unmatched:
                    return reject_proposal(
                        f"{len(unmatched)} SEARCH/REPLACE blocks match no declared target"
                    )
                if _worker_config.strict_diff_application:
                    try:
                        child_code = apply_diff_blocks_strict(
                            parent.code,
                            code_blocks,
                            enforce_evolve_blocks=_worker_config.enforce_evolve_blocks,
                            max_diff_blocks=_worker_config.max_diff_blocks,
                        )
                        child_changes_desc = apply_diff_blocks_strict(
                            parent_changes_desc,
                            desc_blocks,
                            max_diff_blocks=_worker_config.max_diff_blocks,
                        )
                        desc_applied = len(desc_blocks)
                    except Exception as e:
                        return reject_proposal(str(e))
                else:
                    child_code, _ = apply_diff_blocks(parent.code, code_blocks)
                    child_changes_desc, desc_applied = apply_diff_blocks(
                        parent_changes_desc, desc_blocks
                    )

                # Must update the previous changes description
                if (
                    desc_applied == 0
                    or not child_changes_desc.strip()
                    or child_changes_desc.strip() == parent_changes_desc.strip()
                ):
                    return reject_proposal(
                        "changes_description was not updated or empty, program is discarded"
                    )

                changes_summary = format_diff_summary(
                    code_blocks,
                    max_line_len=_worker_config.prompt.diff_summary_max_line_len,
                    max_lines=_worker_config.prompt.diff_summary_max_lines,
                )
            else:
                # All diffs applied only to code
                if _worker_config.strict_diff_application:
                    try:
                        child_code = apply_diff_strict(
                            parent.code,
                            llm_response,
                            _worker_config.diff_pattern,
                            enforce_evolve_blocks=_worker_config.enforce_evolve_blocks,
                            max_diff_blocks=_worker_config.max_diff_blocks,
                        )
                    except Exception as e:
                        return reject_proposal(str(e))
                else:
                    child_code = apply_diff(
                        parent.code, llm_response, _worker_config.diff_pattern
                    )
                changes_summary = format_diff_summary(
                    diff_blocks,
                    max_line_len=_worker_config.prompt.diff_summary_max_line_len,
                    max_lines=_worker_config.prompt.diff_summary_max_lines,
                )
        else:
            from openevolve.utils.code_utils import parse_full_rewrite

            new_code = parse_full_rewrite(llm_response, _worker_config.language)
            if not new_code:
                return reject_proposal("No valid code found in response")

            child_code = new_code
            changes_summary = "Full rewrite"

        # Check code length
        if len(child_code) > _worker_config.max_code_length:
            return reject_proposal(
                f"Generated code exceeds maximum length ({len(child_code)} > {_worker_config.max_code_length})"
            )

        # Evaluate the child program
        import uuid

        child_id = str(uuid.uuid4())
        child_metrics = asyncio.run(_worker_evaluator.evaluate_program(child_code, child_id))

        # Get artifacts
        artifacts = _worker_evaluator.get_pending_artifacts(child_id)
        if bool(child_metrics.pop(EVALUATION_FAILED_METRIC, 0.0)):
            detail = "candidate evaluation failed"
            if isinstance(artifacts, dict):
                failure = artifacts.get("stderr") or artifacts.get("error_type")
                if failure:
                    detail = f"{detail}: {failure}"
            return SerializableResult(
                error=detail,
                iteration=iteration,
                prompt=prompt,
                llm_response=llm_response,
                llm_metadata=llm_metadata,
                artifacts=artifacts,
                target_island=db_snapshot.get("sampling_island"),
            )

        identity_artifact = _worker_config.program_identity_artifact
        if identity_artifact is not None:
            if not isinstance(parent_artifacts, dict) or identity_artifact not in parent_artifacts:
                return reject_proposal(
                    f"Parent is missing configured program identity artifact: {identity_artifact}"
                )
            if not isinstance(artifacts, dict) or identity_artifact not in artifacts:
                return reject_proposal(
                    f"Candidate is missing configured program identity artifact: {identity_artifact}"
                )
            if artifacts[identity_artifact] == parent_artifacts[identity_artifact]:
                return reject_proposal(
                    "Candidate has the same program identity as its parent"
                )

        # Create child program
        child_program = Program(
            id=child_id,
            code=child_code,
            changes_description=child_changes_desc,
            language=_worker_config.language,
            parent_id=parent.id,
            generation=parent.generation + 1,
            metrics=child_metrics,
            iteration_found=iteration,
            metadata={
                "changes": changes_summary,
                "parent_metrics": parent.metrics,
                "island": parent_island,
                "llm": llm_metadata,
            },
        )

        iteration_time = time.time() - iteration_start

        # Get target island from snapshot (where child should be placed)
        target_island = db_snapshot.get("sampling_island")

        return SerializableResult(
            child_program_dict=child_program.to_dict(),
            parent_id=parent.id,
            iteration_time=iteration_time,
            prompt=prompt,
            llm_response=llm_response,
            llm_metadata=llm_metadata,
            artifacts=artifacts,
            iteration=iteration,
            target_island=target_island,
        )

    except Exception as e:
        logger.exception(f"Error in worker iteration {iteration}")
        return SerializableResult(
            error=str(e),
            iteration=iteration,
            prompt=prompt,
            llm_response=llm_response,
            llm_metadata=llm_metadata,
        )


def _wait_for_processes(processes: tuple[mp.Process, ...], timeout: float) -> list[mp.Process]:
    """Wait for process handles to observe worker exits without blocking indefinitely."""
    deadline = time.monotonic() + timeout
    alive = list(processes)
    while alive:
        next_alive = []
        for process in alive:
            try:
                process.join(timeout=0)
                if process.is_alive():
                    next_alive.append(process)
            except (AssertionError, ValueError):
                continue
        alive = next_alive
        remaining = deadline - time.monotonic()
        if not alive or remaining <= 0:
            break
        time.sleep(min(0.001, remaining))
    return alive


def _terminate_process_pool(executor: ProcessPoolExecutor) -> None:
    """Cancel queued work and ensure all process-pool workers have exited."""
    # Python < 3.14 has no public force-shutdown API. Capture only this
    # executor's workers before shutdown clears its private process mapping.
    process_map = getattr(executor, "_processes", None) or {}
    processes = tuple(process_map.copy().values())
    terminate_workers = getattr(executor, "terminate_workers", None)

    if callable(terminate_workers):
        terminate_workers()
    else:
        executor.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            try:
                if process.is_alive():
                    process.terminate()
            except (ProcessLookupError, ValueError):
                continue

    surviving_processes = _wait_for_processes(processes, timeout=1.0)
    for process in surviving_processes:
        try:
            process.kill()
        except (ProcessLookupError, ValueError):
            continue

    surviving_processes = _wait_for_processes(tuple(surviving_processes), timeout=1.0)
    if surviving_processes:
        logger.warning(
            "Process-pool workers did not exit: %s",
            [process.pid for process in surviving_processes],
        )


class ProcessParallelController:
    """Controller for process-based parallel evolution"""

    def __init__(
        self,
        config: Config,
        evaluation_file: str,
        database: ProgramDatabase,
        evolution_tracer=None,
        file_suffix: str = ".py",
        usage_output_path: Optional[str] = None,
    ):
        self.config = config
        self.evaluation_file = evaluation_file
        self.database = database
        self.evolution_tracer = evolution_tracer
        self.file_suffix = file_suffix
        self.usage_output_path = Path(usage_output_path) if usage_output_path else None

        self.executor: Optional[ProcessPoolExecutor] = None
        self.shutdown_event = mp.Event()
        self.early_stopping_triggered = False
        self.target_score_reached = False
        self.completion_reason = "not_started"
        self.last_completed_iteration: Optional[int] = None
        self.completed_iteration_count = 0
        self.submitted_proposal_count = 0
        self.llm_call_count = 0
        self.prompt_token_count = 0
        self.completion_token_count = 0
        self.total_provider_tokens = 0
        self.unreported_provider_token_calls = 0
        self.llm_call_budget_overshoot = 0
        self.provider_token_budget_overshoot = 0
        self.budget_limits_reached: List[str] = []
        self.budget_completion_reason: Optional[str] = None
        self.inflight_proposals_at_budget_stop = 0
        self.accepted_proposal_count = 0
        self.rejected_proposal_count = 0
        self._seen_provider_response_ids: set[str] = set()
        self._known_program_identities: set[tuple[str, Any]] = set()
        self._known_phenotype_identities: set[tuple[str, Any]] = set()
        self._historical_artifact_values = (
            self.database.historical_artifact_values
        )
        self._tracked_history_artifact_names: set[str] = {
            artifact_name
            for artifact_name in (
                self.config.prompt.archive_context_artifact,
                self.config.program_identity_artifact,
                self.config.phenotype_identity_artifact,
            )
            if isinstance(artifact_name, str) and artifact_name
        }
        self._tracked_history_artifact_names.update(
            self._historical_artifact_values
        )
        for program_id in self.database.programs:
            self._discover_neighborhood_identity_artifact(
                self.database.get_artifacts(program_id)
            )
        for program_id in self.database.programs:
            self._record_historical_artifacts(
                self.database.get_artifacts(program_id)
            )
        identity_artifact = self.config.program_identity_artifact
        if identity_artifact is not None:
            for value in self._historical_artifact_values.get(
                identity_artifact, set()
            ):
                identity = self._identity_key(value)
                if identity is not None:
                    self._known_program_identities.add(identity)
        phenotype_artifact = self.config.phenotype_identity_artifact
        if phenotype_artifact is not None:
            for value in self._historical_artifact_values.get(
                phenotype_artifact, set()
            ):
                identity = self._identity_key(value)
                if identity is not None:
                    self._known_phenotype_identities.add(identity)
        self._controller_island_state: Dict[int, Dict[str, Any]] = {
            island_id: {
                "accepted": 0,
                "best_score": 0.0,
                "calls": 0,
                "rejected": 0,
                "tokens": 0,
                "unique": set(),
            }
            for island_id in range(config.database.num_islands)
        }

        # Number of worker processes
        self.num_workers = config.evaluator.parallel_evaluations
        self.num_islands = config.database.num_islands

        logger.info(f"Initialized process parallel controller with {self.num_workers} workers")

    def _early_stopping_score(self, metrics: Dict[str, Any]) -> Optional[float]:
        """Return the configured convergence score for one evaluated program."""
        metric = self.config.early_stopping_metric
        if metric in metrics:
            value = metrics[metric]
        elif metric == "combined_score":
            value = safe_numeric_average(metrics)
        else:
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None

    def _initial_early_stopping_score(self) -> float:
        """Seed convergence tracking from programs already in the archive."""
        scores = (
            self._early_stopping_score(program.metrics)
            for program in self.database.programs.values()
        )
        return max((score for score in scores if score is not None), default=float("-inf"))

    @staticmethod
    def _nonnegative_int(value: Any) -> Optional[int]:
        """Return provider counters only when they are exact non-negative integers."""
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

    @staticmethod
    def _identity_key(value: Any) -> Optional[tuple[str, Any]]:
        """Normalize supported identity artifact values into stable keys."""
        if isinstance(value, str):
            return ("text", value)
        if isinstance(value, bytes):
            return ("bytes", bytes(value))
        return None

    @staticmethod
    def _artifact_text(value: Any) -> Optional[str]:
        """Normalize a persistable evaluator artifact to text."""

        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, str):
            return value
        return None

    def _discover_neighborhood_identity_artifact(
        self,
        artifacts: Any,
    ) -> None:
        """Track the identity named by a valid configured neighborhood."""

        neighborhood_name = (
            self.config.prompt.proposal_neighborhood_artifact
        )
        if not isinstance(neighborhood_name, str) or not isinstance(
            artifacts, dict
        ):
            return
        raw = self._artifact_text(artifacts.get(neighborhood_name))
        if raw is None:
            return
        try:
            neighborhood = json.loads(raw)
        except json.JSONDecodeError:
            return
        if (
            not isinstance(neighborhood, dict)
            or neighborhood.get("schema_version") != 1
        ):
            return
        identity_artifact = neighborhood.get("identity_artifact")
        if isinstance(identity_artifact, str) and identity_artifact:
            self._tracked_history_artifact_names.add(identity_artifact)

    def _record_historical_artifacts(self, artifacts: Any) -> None:
        """Remember relevant measured artifacts even if MAP-Elites drops them."""

        if not isinstance(artifacts, dict):
            return
        self._discover_neighborhood_identity_artifact(artifacts)
        for artifact_name in self._tracked_history_artifact_names:
            value = self._artifact_text(artifacts.get(artifact_name))
            if value is not None:
                self._historical_artifact_values.setdefault(
                    artifact_name, set()
                ).add(value)

    def _apply_program_identity_gate(self, result: SerializableResult) -> None:
        """Reject a measured child whose structural or phenotype identity exists."""
        artifact_name = self.config.program_identity_artifact
        if result.error is not None or not result.child_program_dict:
            return
        artifacts = result.artifacts if isinstance(result.artifacts, dict) else {}
        if artifact_name is not None:
            identity = self._identity_key(artifacts.get(artifact_name))
            if identity is None:
                result.error = (
                    "Candidate is missing configured program identity artifact: "
                    f"{artifact_name}"
                )
                return
            if identity in self._known_program_identities:
                result.error = "Candidate program identity already exists in the archive"
                return
            # Reserve at result processing time so simultaneously generated
            # duplicates cannot both enter before the database snapshot refreshes.
            self._known_program_identities.add(identity)

        phenotype_artifact = self.config.phenotype_identity_artifact
        if phenotype_artifact is None:
            return
        phenotype = self._identity_key(artifacts.get(phenotype_artifact))
        if phenotype is None:
            result.error = (
                "Candidate is missing configured phenotype identity artifact: "
                f"{phenotype_artifact}"
            )
            return
        if phenotype in self._known_phenotype_identities:
            result.error = "Candidate phenotype identity already exists in the archive"
            return
        self._known_phenotype_identities.add(phenotype)

    @staticmethod
    def _budget_reason(limits_reached: List[str]) -> Optional[str]:
        """Return a stable public completion reason for the first reached budget."""
        call_limit = "max_llm_calls" in limits_reached
        token_limit = "max_total_provider_tokens" in limits_reached
        if call_limit and token_limit:
            return "max_llm_calls_and_max_total_provider_tokens_reached"
        if call_limit:
            return "max_llm_calls_reached"
        if token_limit:
            return "max_total_provider_tokens_reached"
        return None

    def _refresh_budget_state(self, inflight_proposals: int) -> Optional[str]:
        """Refresh budget state after a reservation or verified usage receipt."""
        limits_reached = []
        if (
            self.config.max_llm_calls is not None
            and self.submitted_proposal_count >= self.config.max_llm_calls
        ):
            limits_reached.append("max_llm_calls")
            self.llm_call_budget_overshoot = (
                self.submitted_proposal_count - self.config.max_llm_calls
            )
        if (
            self.config.max_total_provider_tokens is not None
            and self.total_provider_tokens >= self.config.max_total_provider_tokens
        ):
            limits_reached.append("max_total_provider_tokens")
            self.provider_token_budget_overshoot = (
                self.total_provider_tokens - self.config.max_total_provider_tokens
            )
        self.budget_limits_reached = limits_reached

        reached_reason = self._budget_reason(limits_reached)
        if reached_reason and self.budget_completion_reason is None:
            self.budget_completion_reason = reached_reason
            self.inflight_proposals_at_budget_stop = max(0, inflight_proposals)
        return self.budget_completion_reason

    def _record_controller_result(
        self,
        island_id: Optional[int],
        result: SerializableResult,
    ) -> None:
        """Update only observed per-island counters used by the live allocator."""

        if result.error is not None or not result.child_program_dict:
            self.rejected_proposal_count += 1
        else:
            self.accepted_proposal_count += 1

        if (
            not self.config.database.controller_scheduler.enabled
            or not isinstance(island_id, int)
            or island_id not in self._controller_island_state
        ):
            return
        state = self._controller_island_state[island_id]
        state["calls"] += 1
        metadata = result.llm_metadata if isinstance(result.llm_metadata, dict) else {}
        usage = metadata.get("usage")
        if isinstance(usage, dict):
            total_tokens = self._nonnegative_int(usage.get("total_tokens"))
            if total_tokens is not None:
                state["tokens"] += total_tokens
        if result.error is not None or not result.child_program_dict:
            state["rejected"] += 1
            return
        state["accepted"] += 1
        child = result.child_program_dict
        diversity_artifact = (
            self.config.database.controller_scheduler.diversity_artifact
        )
        if diversity_artifact is not None:
            artifacts = result.artifacts if isinstance(result.artifacts, dict) else {}
            diversity_identity = self._identity_key(
                artifacts.get(diversity_artifact)
            )
            if diversity_identity is not None:
                state["unique"].add(diversity_identity)
        else:
            code = child.get("code")
            if isinstance(code, str):
                state["unique"].add(
                    ("code", hashlib.sha256(code.encode("utf-8")).hexdigest())
                )
        metrics = child.get("metrics")
        if isinstance(metrics, dict):
            score_metric = self.config.database.controller_scheduler.score_metric
            score = metrics.get(score_metric)
            if (
                score_metric == "combined_score"
                and (
                    not isinstance(score, (int, float))
                    or isinstance(score, bool)
                )
            ):
                score = safe_numeric_average(metrics)
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                state["best_score"] = max(float(state["best_score"]), float(score))

    def _select_controller_island(
        self,
        island_pending: Dict[int, List[int]],
        batch_size: int,
    ) -> int:
        """Choose an island from prior observed results with deterministic ties."""

        scheduler = self.config.database.controller_scheduler
        available = [
            island_id
            for island_id in range(self.num_islands)
            if len(island_pending[island_id]) < batch_size
        ]
        if not available:
            raise RuntimeError("controller scheduler has no available island")
        if self.submitted_proposal_count < scheduler.minimum_calls:
            return min(
                available,
                key=lambda island_id: (
                    int(self._controller_island_state[island_id]["calls"]),
                    len(island_pending[island_id]),
                    island_id,
                ),
            )
        if scheduler.leader_score_band is not None:
            global_best = max(
                self._controller_island_best_score(island_id)
                for island_id in range(self.num_islands)
            )
            quality_frontier = [
                island_id
                for island_id in available
                if (
                    global_best - self._controller_island_best_score(island_id)
                    <= float(scheduler.leader_score_band)
                )
            ]
            if quality_frontier:
                available = quality_frontier

        def priority(island_id: int) -> tuple[float, int]:
            state = self._controller_island_state[island_id]
            calls = int(state["calls"])
            accepted = int(state["accepted"])
            rejected = int(state["rejected"])
            tokens = int(state["tokens"])
            unique = len(state["unique"])
            best_score = self._controller_island_best_score(island_id)
            validity = accepted / calls if calls else 1.0
            diversity = unique / accepted if accepted else 1.0
            token_efficiency = best_score / max(tokens / 1000.0, 1.0)
            value = (
                scheduler.exploitation_weight * best_score
                + scheduler.underexplored_weight / (1.0 + calls)
                + scheduler.validity_weight * validity
                + scheduler.diversity_weight * diversity
                + scheduler.token_efficiency_weight * token_efficiency
                - scheduler.rejection_penalty
                * (rejected / calls if calls else 0.0)
            )
            return (-value, island_id)

        return min(available, key=priority)

    def _controller_island_best_score(self, island_id: int) -> float:
        """Return the best observed or currently retained score for an island."""

        state = self._controller_island_state[island_id]
        best_score = float(state["best_score"])
        score_metric = self.config.database.controller_scheduler.score_metric
        if not 0 <= island_id < len(self.database.islands):
            return best_score
        for program_id in self.database.islands[island_id]:
            program = self.database.programs.get(program_id)
            if program is None or not isinstance(program.metrics, dict):
                continue
            score = program.metrics.get(score_metric)
            if (
                score_metric == "combined_score"
                and (
                    not isinstance(score, (int, float))
                    or isinstance(score, bool)
                )
            ):
                score = safe_numeric_average(program.metrics)
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                best_score = max(best_score, float(score))
        return best_score

    def _controller_parallelism(self, max_iterations: int) -> int:
        """Return active scheduler slots while preserving a selectable island."""

        if max_iterations < 1:
            return 0
        scheduler = self.config.database.controller_scheduler
        if not scheduler.enabled:
            return min(self.num_islands, max_iterations)
        selectable_islands = max(1, self.num_islands - scheduler.reserve_islands)
        return min(
            self.num_workers,
            selectable_islands,
            max_iterations,
        )

    def _adaptive_controller_parallelism(self, max_iterations: int) -> int:
        """Return the lower quality-phase frontier when one is configured."""

        warmup = self._controller_parallelism(max_iterations)
        configured = (
            self.config.database.controller_scheduler.adaptive_parallelism
        )
        if configured is None:
            return warmup
        return min(warmup, configured)

    def _target_controller_parallelism(self, max_iterations: int) -> int:
        """Return warmup or adaptive frontier from submitted proposal count."""

        scheduler = self.config.database.controller_scheduler
        if (
            scheduler.enabled
            and self.submitted_proposal_count >= scheduler.minimum_calls
        ):
            return self._adaptive_controller_parallelism(max_iterations)
        return self._controller_parallelism(max_iterations)

    @property
    def controller_allocation(self) -> Dict[str, Any]:
        """Return a serializable receipt for observed-result island allocation."""

        scheduler = self.config.database.controller_scheduler
        islands = []
        for island_id in range(self.num_islands):
            state = self._controller_island_state[island_id]
            islands.append(
                {
                    "island": island_id,
                    "calls": int(state["calls"]),
                    "accepted": int(state["accepted"]),
                    "rejected": int(state["rejected"]),
                    "distinct_behaviors": len(state["unique"]),
                    "best_score": self._controller_island_best_score(island_id),
                    "tokens": int(state["tokens"]),
                }
            )
        return {
            "enabled": bool(scheduler.enabled),
            "score_metric": scheduler.score_metric,
            "diversity_artifact": scheduler.diversity_artifact,
            "leader_score_band": scheduler.leader_score_band,
            "parent_score_band": scheduler.parent_score_band,
            "reserve_islands": scheduler.reserve_islands,
            "active_parallelism": self._controller_parallelism(
                max(1, self.config.max_iterations)
            ),
            "adaptive_parallelism": self._adaptive_controller_parallelism(
                max(1, self.config.max_iterations)
            ),
            "islands": islands,
        }

    def _register_submitted_proposal(self, inflight_proposals: int) -> Optional[str]:
        """Reserve one logical proposal-model call against the exact call cap."""
        self.submitted_proposal_count += 1
        return self._refresh_budget_state(inflight_proposals)

    @property
    def llm_usage(self) -> Dict[str, Any]:
        """Public, JSON-serializable proposal-model usage summary for this run."""
        return {
            "call_budget_counting_basis": "submitted_proposals",
            "provider_usage_counting_basis": "unique_provider_receipts",
            "llm_calls_submitted": self.submitted_proposal_count,
            "llm_calls": self.llm_call_count,
            "accepted_proposals": self.accepted_proposal_count,
            "rejected_proposals": self.rejected_proposal_count,
            "prompt_tokens": self.prompt_token_count,
            "completion_tokens": self.completion_token_count,
            "total_provider_tokens": self.total_provider_tokens,
            "unreported_provider_token_calls": self.unreported_provider_token_calls,
            "max_llm_calls": self.config.max_llm_calls,
            "max_total_provider_tokens": self.config.max_total_provider_tokens,
            "llm_call_budget_overshoot": self.llm_call_budget_overshoot,
            "provider_token_budget_overshoot": self.provider_token_budget_overshoot,
            "limits_reached": list(self.budget_limits_reached),
            "inflight_proposals_at_budget_stop": self.inflight_proposals_at_budget_stop,
            "measured_artifact_history": {
                name: len(values)
                for name, values in sorted(
                    self._historical_artifact_values.items()
                )
            },
            "controller_allocation": self.controller_allocation,
        }

    def _record_llm_usage(
        self,
        iteration: int,
        result: SerializableResult,
        inflight_proposals: int = 0,
    ) -> Optional[str]:
        """Aggregate and persist one uniquely identified provider receipt.

        The call cap is reserved before submission. Provider response identity
        is required for token accounting; missing counters are never guessed.
        """
        if not isinstance(result.llm_metadata, dict):
            return self.budget_completion_reason

        metadata = result.llm_metadata
        response_id = metadata.get("provider_response_id")
        verified_receipt = isinstance(response_id, str) and bool(response_id.strip())
        duplicate_receipt = bool(
            verified_receipt and response_id in self._seen_provider_response_ids
        )
        usage = metadata.get("usage")
        usage = usage if isinstance(usage, dict) else {}

        if verified_receipt and not duplicate_receipt:
            self._seen_provider_response_ids.add(response_id)
            self.llm_call_count += 1

            prompt_tokens = self._nonnegative_int(usage.get("prompt_tokens"))
            completion_tokens = self._nonnegative_int(usage.get("completion_tokens"))
            total_tokens = self._nonnegative_int(usage.get("total_tokens"))

            if prompt_tokens is not None:
                self.prompt_token_count += prompt_tokens
            if completion_tokens is not None:
                self.completion_token_count += completion_tokens
            if (
                total_tokens is None
                and prompt_tokens is not None
                and completion_tokens is not None
            ):
                total_tokens = prompt_tokens + completion_tokens

            if total_tokens is None:
                self.unreported_provider_token_calls += 1
            else:
                self.total_provider_tokens += total_tokens

        self._refresh_budget_state(inflight_proposals)

        payload = {
            **metadata,
            "iteration": iteration,
            "parent_id": result.parent_id,
            "target_island": result.target_island,
            "candidate_measured": bool(result.child_program_dict)
            or bool(result.artifacts),
            "accepted_for_archive": (
                result.error is None and bool(result.child_program_dict)
            ),
            # Backward-compatible field retained for existing receipt readers.
            "accepted_for_evaluation": result.error is None,
            "rejection_reason": result.error,
            "llm_response_sha256": (
                hashlib.sha256(result.llm_response.encode("utf-8")).hexdigest()
                if isinstance(result.llm_response, str)
                else None
            ),
            "verified_provider_receipt": verified_receipt,
            "duplicate_provider_receipt": duplicate_receipt,
            "cumulative_usage": self.llm_usage,
        }
        if self.usage_output_path:
            self.usage_output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.usage_output_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
                )

        return self.budget_completion_reason

    def _serialize_config(self, config: Config) -> dict:
        """Serialize config object to a dictionary that can be pickled"""
        # Manual serialization to handle nested objects properly

        # The asdict() call itself triggers the deepcopy which tries to serialize novelty_llm. Remove it first.
        config.database.novelty_llm = None

        return {
            "llm": {
                "models": [asdict(m) for m in config.llm.models],
                "evaluator_models": [asdict(m) for m in config.llm.evaluator_models],
                "api_base": config.llm.api_base,
                "api_key": config.llm.api_key,
                "temperature": config.llm.temperature,
                "top_p": config.llm.top_p,
                "max_tokens": config.llm.max_tokens,
                "timeout": config.llm.timeout,
                "retries": config.llm.retries,
                "retry_delay": config.llm.retry_delay,
            },
            "prompt": asdict(config.prompt),
            "database": asdict(config.database),
            "evaluator": asdict(config.evaluator),
            "max_iterations": config.max_iterations,
            "max_llm_calls": config.max_llm_calls,
            "max_total_provider_tokens": config.max_total_provider_tokens,
            "checkpoint_interval": config.checkpoint_interval,
            "log_level": config.log_level,
            "log_dir": config.log_dir,
            "random_seed": config.random_seed,
            "diff_based_evolution": config.diff_based_evolution,
            "max_code_length": config.max_code_length,
            "diff_pattern": config.diff_pattern,
            "strict_diff_application": config.strict_diff_application,
            "enforce_evolve_blocks": config.enforce_evolve_blocks,
            "max_diff_blocks": config.max_diff_blocks,
            "program_identity_artifact": config.program_identity_artifact,
            "phenotype_identity_artifact": config.phenotype_identity_artifact,
            "language": config.language,
            "file_suffix": self.file_suffix,
        }

    def start(self) -> None:
        """Start the process pool"""
        # Convert config to dict for pickling
        # We need to be careful with nested dataclasses
        config_dict = self._serialize_config(self.config)

        # Pass current environment to worker processes
        import os
        import sys

        current_env = dict(os.environ)

        executor_kwargs = {
            "max_workers": self.num_workers,
            "initializer": _worker_init,
            "initargs": (config_dict, self.evaluation_file, current_env),
        }
        if sys.version_info >= (3, 11):
            logger.info(f"Set max {self.config.max_tasks_per_child} tasks per child")
            executor_kwargs["max_tasks_per_child"] = self.config.max_tasks_per_child
        elif self.config.max_tasks_per_child is not None:
            logger.warn(
                "max_tasks_per_child is only supported in Python 3.11+. "
                "Ignoring max_tasks_per_child and using spawn start method."
            )
            executor_kwargs["mp_context"] = mp.get_context("spawn")

        # Create process pool with initializer
        self.executor = ProcessPoolExecutor(**executor_kwargs)
        logger.info(f"Started process pool with {self.num_workers} processes")

    def stop(self) -> None:
        """Stop the process pool"""
        self.shutdown_event.set()

        executor = self.executor
        self.executor = None
        if executor:
            _terminate_process_pool(executor)

        logger.info("Stopped process pool")

    def request_shutdown(self) -> None:
        """Request graceful shutdown"""
        logger.info("Graceful shutdown requested...")
        self.shutdown_event.set()

    def _create_database_snapshot(self) -> Dict[str, Any]:
        """Create a serializable snapshot of the database state"""
        # Only include necessary data for workers
        snapshot = {
            "programs": {pid: prog.to_dict() for pid, prog in self.database.programs.items()},
            "islands": [list(island) for island in self.database.islands],
            "current_island": self.database.current_island,
            "feature_dimensions": self.database.config.feature_dimensions,
            "artifacts": {},  # Will be populated selectively
            "historical_artifact_values": {
                name: sorted(values)
                for name, values in sorted(
                    self._historical_artifact_values.items()
                )
            },
        }

        # Include artifacts for programs that might be selected
        # This limits artifacts (execution outputs/errors) to avoid large snapshot sizes.
        # This does NOT affect program code - all programs are fully serialized above.
        # With max_artifact_bytes=20KB and population_size=1000, artifacts could be 20MB total,
        # which would significantly slow worker process initialization. The default limit of 100
        # keeps artifact data under 2MB while still providing execution context for recent programs.
        # Workers can still evolve properly as they have access to ALL program code.
        # Configure via database.max_snapshot_artifacts (None for unlimited).
        max_artifacts = self.database.config.max_snapshot_artifacts
        program_ids = list(self.database.programs.keys())
        if max_artifacts is not None:
            program_ids = program_ids[:max_artifacts]
        for pid in program_ids:
            artifacts = self.database.get_artifacts(pid)
            if artifacts:
                snapshot["artifacts"][pid] = artifacts

        return snapshot

    async def run_evolution(
        self,
        start_iteration: int,
        max_iterations: int,
        target_score: Optional[float] = None,
        checkpoint_callback=None,
    ):
        """Run evolution with process-based parallelism"""
        if not self.executor:
            raise RuntimeError("Process pool not started")

        total_iterations = start_iteration + max_iterations

        logger.info(
            f"Starting process-based evolution from iteration {start_iteration} "
            f"for {max_iterations} iterations (total: {total_iterations})"
        )

        # Track pending futures by island to maintain distribution
        pending_futures: Dict[int, Future] = {}
        island_pending: Dict[int, List[int]] = {i: [] for i in range(self.num_islands)}
        # Keep at most one proposal in flight per island. Multiple simultaneous
        # calls from the same unchanged parent can carry identical prompts and
        # waste the proposal budget on duplicate children.
        batch_per_island = 1 if max_iterations > 0 else 0
        current_iteration = start_iteration
        stop_scheduling = False
        self.completion_reason = "running"

        # The live controller deliberately leaves configured reserve islands
        # outside the active frontier. With at least one free island, every
        # completion creates a real allocation choice instead of mechanically
        # refilling the island that just finished.
        initial_parallelism = self._controller_parallelism(max_iterations)
        initial_island_pending: Dict[int, List[int]] = {
            i: [] for i in range(self.num_islands)
        }
        initial_islands: List[int] = []
        for slot in range(initial_parallelism):
            if self.config.database.controller_scheduler.enabled:
                island_id = self._select_controller_island(
                    initial_island_pending,
                    batch_per_island,
                )
            else:
                island_id = slot
            initial_islands.append(island_id)
            initial_island_pending[island_id].append(start_iteration + slot)

        # Round-robin distribution across the active island frontier.
        for island_id in initial_islands:
            for _ in range(batch_per_island):
                if current_iteration < total_iterations and not stop_scheduling:
                    future = self._submit_iteration(current_iteration, island_id)
                    if future:
                        pending_futures[current_iteration] = future
                        island_pending[island_id].append(current_iteration)
                        budget_reason = self._register_submitted_proposal(
                            inflight_proposals=len(pending_futures)
                        )
                        if budget_reason:
                            stop_scheduling = True
                            self.completion_reason = budget_reason
                    current_iteration += 1

        next_iteration = current_iteration
        completed_iterations = 0

        # Early stopping tracking
        early_stopping_enabled = self.config.early_stopping_patience is not None
        if early_stopping_enabled:
            best_score = self._initial_early_stopping_score()
            iterations_without_improvement = 0
            if self.config.early_stopping_patience < 0:
                logger.info(
                    f"Early stopping patience is set to a negative value, running event-based early-stopping, "
                    f"Early stop when metric '{self.config.early_stopping_metric}' reaches {self.config.convergence_threshold}"
                )
            else:
                logger.info(
                    f"Early stopping enabled: patience={self.config.early_stopping_patience}, "
                    f"threshold={self.config.convergence_threshold}, "
                    f"metric={self.config.early_stopping_metric}, "
                    f"initial_best={best_score:.4f}"
                )
        else:
            logger.info("Early stopping disabled")

        # Process results as they complete
        while (
            pending_futures
            and completed_iterations < max_iterations
            and not self.shutdown_event.is_set()
        ):
            # Find completed futures
            completed_iteration = None
            for iteration, future in list(pending_futures.items()):
                if future.done():
                    completed_iteration = iteration
                    break

            if completed_iteration is None:
                await asyncio.sleep(0.01)
                continue

            # Process completed result
            future = pending_futures.pop(completed_iteration)

            try:
                # Use evaluator timeout + buffer to gracefully handle stuck processes
                timeout_seconds = self.config.evaluator.timeout + 30
                result = future.result(timeout=timeout_seconds)
                completed_island = (
                    result.target_island
                    if isinstance(result.target_island, int)
                    else next(
                        (
                            island_id
                            for island_id, iterations in island_pending.items()
                            if completed_iteration in iterations
                        ),
                        None,
                    )
                )
                self._record_historical_artifacts(result.artifacts)
                self._apply_program_identity_gate(result)
                self._record_controller_result(completed_island, result)
                budget_reason = self._record_llm_usage(
                    completed_iteration,
                    result,
                    inflight_proposals=len(pending_futures),
                )
                if budget_reason:
                    if not stop_scheduling:
                        logger.info(
                            "LLM budget reached at iteration %s; draining %s in-flight "
                            "proposal(s) without scheduling new work",
                            completed_iteration,
                            len(pending_futures),
                        )
                    stop_scheduling = True
                    self.completion_reason = budget_reason

                if result.error:
                    logger.warning(f"Iteration {completed_iteration} error: {result.error}")
                elif result.child_program_dict:
                    # Reconstruct program from dict
                    child_program = Program(**result.child_program_dict)
                    # Capture lineage before insertion. MAP-Elites replacement
                    # may remove the parent while adding the child.
                    trace_parent_program = (
                        self.database.get(result.parent_id)
                        if result.parent_id
                        else None
                    )

                    # Add to database with explicit target_island to ensure proper island placement
                    # This fixes issue #391: children should go to the target island, not inherit
                    # from the parent (which may be from a different island due to fallback sampling)
                    self.database.add(
                        child_program,
                        iteration=completed_iteration,
                        target_island=result.target_island,
                    )

                    # Store artifacts
                    if result.artifacts:
                        self.database.store_artifacts(child_program.id, result.artifacts)

                    # Log evolution trace
                    if self.evolution_tracer:
                        if trace_parent_program:
                            # Determine island ID
                            island_id = child_program.metadata.get(
                                "island", self.database.current_island
                            )

                            self.evolution_tracer.log_trace(
                                iteration=completed_iteration,
                                parent_program=trace_parent_program,
                                child_program=child_program,
                                prompt=result.prompt,
                                llm_response=result.llm_response,
                                artifacts=result.artifacts,
                                island_id=island_id,
                                metadata={
                                    "iteration_time": result.iteration_time,
                                    "changes": child_program.metadata.get("changes", ""),
                                    "llm": result.llm_metadata,
                                },
                            )

                    # Log prompts
                    if result.prompt:
                        self.database.log_prompt(
                            template_key=(
                                "full_rewrite_user"
                                if not self.config.diff_based_evolution
                                else "diff_user"
                            ),
                            program_id=child_program.id,
                            prompt=result.prompt,
                            responses=[result.llm_response] if result.llm_response else [],
                        )

                    # Island management
                    # get current program island id
                    island_id = child_program.metadata.get("island", self.database.current_island)
                    # use this to increment island generation
                    self.database.increment_island_generation(island_idx=island_id)

                    # Check migration
                    if self.database.should_migrate():
                        logger.info(f"Performing migration at iteration {completed_iteration}")
                        self.database.migrate_programs()
                        self.database.log_island_status()

                    # Log progress
                    logger.info(
                        f"Iteration {completed_iteration}: "
                        f"Program {child_program.id} "
                        f"(parent: {result.parent_id}) "
                        f"completed in {result.iteration_time:.2f}s"
                    )

                    if child_program.metrics:
                        metrics_str = ", ".join(
                            [
                                f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                                for k, v in child_program.metrics.items()
                            ]
                        )
                        logger.info(f"Metrics: {metrics_str}")

                        # Check if this is the first program without combined_score
                        if not hasattr(self, "_warned_about_combined_score"):
                            self._warned_about_combined_score = False

                        if (
                            "combined_score" not in child_program.metrics
                            and not self._warned_about_combined_score
                        ):
                            avg_score = safe_numeric_average(child_program.metrics)
                            logger.warning(
                                f"⚠️  No 'combined_score' metric found in evaluation results. "
                                f"Using average of all numeric metrics ({avg_score:.4f}) for evolution guidance. "
                                f"For better evolution results, please modify your evaluator to return a 'combined_score' "
                                f"metric that properly weights different aspects of program performance."
                            )
                            self._warned_about_combined_score = True

                    # Check for new best
                    if self.database.best_program_id == child_program.id:
                        logger.info(
                            f"🌟 New best solution found at iteration {completed_iteration}: "
                            f"{child_program.id}"
                        )

                    # Checkpoint callback
                    # Don't checkpoint at iteration 0 (that's just the initial program)
                    if (
                        completed_iteration > 0
                        and completed_iteration % self.config.checkpoint_interval == 0
                    ):
                        logger.info(
                            f"Checkpoint interval reached at iteration {completed_iteration}"
                        )
                        self.database.log_island_status()
                        if checkpoint_callback:
                            checkpoint_callback(completed_iteration)

                    # Check target score
                    if target_score is not None and child_program.metrics:
                        if (
                            "combined_score" in child_program.metrics
                            and child_program.metrics["combined_score"] >= target_score
                        ):
                            logger.info(
                                f"Target score {target_score} reached at iteration {completed_iteration}"
                            )
                            self.target_score_reached = True
                            self.completion_reason = "target_score_reached"
                            stop_scheduling = True

                    # Check early stopping
                    if early_stopping_enabled and child_program.metrics:
                        current_score = self._early_stopping_score(child_program.metrics)
                        if current_score is None:
                            logger.warning(
                                f"Early stopping metric '{self.config.early_stopping_metric}' "
                                "not found or non-numeric; ignoring this result"
                            )

                        if current_score is not None:
                            # Check for improvement
                            if self.config.early_stopping_patience > 0:
                                improvement = current_score - best_score
                                if improvement >= self.config.convergence_threshold:
                                    best_score = current_score
                                    iterations_without_improvement = 0
                                    logger.debug(
                                        f"New best score: {best_score:.4f} (improvement: {improvement:+.4f})"
                                    )
                                else:
                                    iterations_without_improvement += 1
                                    logger.debug(
                                        f"No improvement: {iterations_without_improvement}/{self.config.early_stopping_patience}"
                                    )

                                # Check if we should stop
                                if (
                                    iterations_without_improvement
                                    >= self.config.early_stopping_patience
                                    and (
                                        not self.config.database.controller_scheduler.enabled
                                        or completed_iterations + 1
                                        >= self.config.database.controller_scheduler.minimum_calls
                                    )
                                ):
                                    self.early_stopping_triggered = True
                                    self.completion_reason = "early_stopping"
                                    stop_scheduling = True
                                    logger.info(
                                        f"🛑 Early stopping triggered at iteration {completed_iteration}: "
                                        f"No improvement for {iterations_without_improvement} iterations "
                                        f"(best score: {best_score:.4f})"
                                    )

                            else:
                                # Event-based early stopping
                                if current_score == self.config.convergence_threshold:
                                    best_score = current_score
                                    logger.info(
                                        f"🛑 Early stopping (event-based) triggered at iteration {completed_iteration}: "
                                        f"Task successfully solved with score {best_score:.4f}."
                                    )
                                    self.early_stopping_triggered = True
                                    self.completion_reason = "early_stopping"
                                    stop_scheduling = True

            except FutureTimeoutError:
                logger.error(
                    f"⏰ Iteration {completed_iteration} timed out after {timeout_seconds}s "
                    f"(evaluator timeout: {self.config.evaluator.timeout}s + 30s buffer). "
                    f"Canceling future and continuing with next iteration."
                )
                # Cancel the future to clean up the process
                future.cancel()
            except Exception as e:
                logger.error(f"Error processing result from iteration {completed_iteration}: {e}")

            completed_iterations += 1
            self.completed_iteration_count = completed_iterations
            self.last_completed_iteration = (
                completed_iteration
                if self.last_completed_iteration is None
                else max(self.last_completed_iteration, completed_iteration)
            )
            self.database.last_iteration = max(
                self.database.last_iteration, completed_iteration
            )

            # Remove completed iteration from island tracking
            for island_id, iteration_list in island_pending.items():
                if completed_iteration in iteration_list:
                    iteration_list.remove(completed_iteration)
                    break

            # Submit the next iteration either with the original balanced
            # allocator or the explicitly enabled observed-result controller.
            if self.config.database.controller_scheduler.enabled:
                target_parallelism = self._target_controller_parallelism(
                    max_iterations
                )
                island_order = (
                    [
                        self._select_controller_island(
                            island_pending,
                            batch_per_island,
                        )
                    ]
                    if len(pending_futures) < target_parallelism
                    else []
                )
                per_island_limit = batch_per_island
            else:
                island_order = range(self.num_islands)
                per_island_limit = batch_per_island
            for island_id in island_order:
                if (
                    len(island_pending[island_id]) < per_island_limit
                    and next_iteration < total_iterations
                    and not self.shutdown_event.is_set()
                    and not stop_scheduling
                ):
                    future = self._submit_iteration(next_iteration, island_id)
                    if future:
                        pending_futures[next_iteration] = future
                        island_pending[island_id].append(next_iteration)
                        budget_reason = self._register_submitted_proposal(
                            inflight_proposals=len(pending_futures)
                        )
                        if budget_reason:
                            stop_scheduling = True
                            self.completion_reason = budget_reason
                        next_iteration += 1
                        break  # Only submit one iteration per completion to maintain balance

        # Handle shutdown
        if self.shutdown_event.is_set():
            logger.info("Shutdown requested, canceling remaining evaluations...")
            for future in pending_futures.values():
                future.cancel()

        # Log completion reason
        if self.early_stopping_triggered:
            logger.info("✅ Evolution completed - Early stopping triggered due to convergence")
        elif self.target_score_reached:
            logger.info("✅ Evolution completed - Target reached; in-flight work recorded")
        elif self.budget_completion_reason:
            self.completion_reason = self.budget_completion_reason
            logger.info(
                "✅ Evolution completed - %s; usage=%s",
                self.budget_completion_reason,
                self.llm_usage,
            )
        elif self.shutdown_event.is_set():
            self.completion_reason = "shutdown_requested"
            logger.info("✅ Evolution completed - Shutdown requested")
        else:
            self.completion_reason = "maximum_iterations"
            logger.info("✅ Evolution completed - Maximum iterations reached")

        return self.database.get_best_program()

    def _submit_iteration(
        self, iteration: int, island_id: Optional[int] = None
    ) -> Optional[Future]:
        """Submit an iteration to the process pool, optionally pinned to a specific island"""
        try:
            # Use specified island or current island
            target_island = island_id if island_id is not None else self.database.current_island

            # Use thread-safe sampling that doesn't modify shared state
            # This fixes the race condition from GitHub issue #246
            # Inspirations are the diverse/creative examples; size them by
            # num_diverse_programs (not num_top_programs) so the config parameter
            # actually controls the inspiration count (GitHub issue #452).
            scheduler = self.config.database.controller_scheduler
            if (
                scheduler.enabled
                and self.submitted_proposal_count >= scheduler.minimum_calls
                and scheduler.parent_score_band is not None
            ):
                parent, inspirations = (
                    self.database.sample_from_island_score_band(
                        target_island,
                        score_metric=scheduler.score_metric,
                        score_band=float(scheduler.parent_score_band),
                        num_inspirations=(
                            self.config.prompt.num_diverse_programs
                        ),
                    )
                )
            else:
                parent, inspirations = self.database.sample_from_island(
                    island_id=target_island,
                    num_inspirations=(
                        self.config.prompt.num_diverse_programs
                    ),
                )

            # Create database snapshot
            db_snapshot = self._create_database_snapshot()
            db_snapshot["sampling_island"] = target_island  # Mark which island this is for
            # The selected parent must always carry its evaluator context even
            # when snapshot artifact limits omit older archive entries.
            parent_artifacts = self.database.get_artifacts(parent.id)
            if parent_artifacts:
                db_snapshot["artifacts"][parent.id] = parent_artifacts

            # Submit to process pool
            future = self.executor.submit(
                _run_iteration_worker,
                iteration,
                db_snapshot,
                parent.id,
                [insp.id for insp in inspirations],
            )

            return future

        except Exception as e:
            logger.error(f"Error submitting iteration {iteration}: {e}")
            return None
