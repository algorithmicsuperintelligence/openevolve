"""
Process-based parallel controller for true parallelism
"""

import asyncio
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
    database_config = DatabaseConfig(**config_dict["database"])
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


def _run_iteration_worker(
    iteration: int, db_snapshot: Dict[str, Any], parent_id: str, inspiration_ids: List[str]
) -> SerializableResult:
    """Run a single iteration in a worker process"""
    try:
        # Lazy initialization
        _lazy_init_worker_components()

        # Reconstruct programs from snapshot
        programs = {pid: Program(**prog_dict) for pid, prog_dict in db_snapshot["programs"].items()}

        parent = programs[parent_id]
        inspirations = [programs[pid] for pid in inspiration_ids if pid in programs]

        # Get parent artifacts if available
        parent_artifacts = db_snapshot["artifacts"].get(parent_id)

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
            return SerializableResult(error=f"LLM generation failed: {str(e)}", iteration=iteration)

        # Check for None response
        if llm_response is None:
            return SerializableResult(error="LLM returned None response", iteration=iteration)

        # Parse response based on evolution mode
        if _worker_config.diff_based_evolution:
            from openevolve.utils.code_utils import (
                apply_diff,
                apply_diff_blocks,
                extract_diffs,
                format_diff_summary,
                split_diffs_by_target,
            )

            diff_blocks = extract_diffs(llm_response, _worker_config.diff_pattern)
            if not diff_blocks:
                return SerializableResult(
                    error="No valid diffs found in response", iteration=iteration
                )

            if _worker_config.prompt.programs_as_changes_description:
                try:
                    code_blocks, desc_blocks, _unmatched = split_diffs_by_target(
                        diff_blocks,
                        code_text=parent.code,
                        changes_description_text=parent_changes_desc,
                    )
                except Exception as e:
                    return SerializableResult(error=str(e), iteration=iteration)

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
                    return SerializableResult(
                        error="changes_description was not updated or empty, program is discarded",
                        iteration=iteration,
                    )

                changes_summary = format_diff_summary(
                    code_blocks,
                    max_line_len=_worker_config.prompt.diff_summary_max_line_len,
                    max_lines=_worker_config.prompt.diff_summary_max_lines,
                )
            else:
                # All diffs applied only to code
                child_code = apply_diff(parent.code, llm_response, _worker_config.diff_pattern)
                changes_summary = format_diff_summary(
                    diff_blocks,
                    max_line_len=_worker_config.prompt.diff_summary_max_line_len,
                    max_lines=_worker_config.prompt.diff_summary_max_lines,
                )
        else:
            from openevolve.utils.code_utils import parse_full_rewrite

            new_code = parse_full_rewrite(llm_response, _worker_config.language)
            if not new_code:
                return SerializableResult(
                    error=f"No valid code found in response", iteration=iteration
                )

            child_code = new_code
            changes_summary = "Full rewrite"

        # Check code length
        if len(child_code) > _worker_config.max_code_length:
            return SerializableResult(
                error=f"Generated code exceeds maximum length ({len(child_code)} > {_worker_config.max_code_length})",
                iteration=iteration,
            )

        # Evaluate the child program
        import uuid

        child_id = str(uuid.uuid4())
        child_metrics = asyncio.run(_worker_evaluator.evaluate_program(child_code, child_id))

        # Get artifacts
        artifacts = _worker_evaluator.get_pending_artifacts(child_id)

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
            artifacts=artifacts,
            iteration=iteration,
            target_island=target_island,
        )

    except Exception as e:
        logger.exception(f"Error in worker iteration {iteration}")
        return SerializableResult(error=str(e), iteration=iteration)


class ProcessParallelController:
    """Controller for process-based parallel evolution"""

    def __init__(
        self,
        config: Config,
        evaluation_file: str,
        database: ProgramDatabase,
        evolution_tracer=None,
        file_suffix: str = ".py",
    ):
        self.config = config
        self.evaluation_file = evaluation_file
        self.database = database
        self.evolution_tracer = evolution_tracer
        self.file_suffix = file_suffix

        self.executor: Optional[ProcessPoolExecutor] = None
        self.shutdown_event = mp.Event()
        self.early_stopping_triggered = False

        # Number of worker processes
        self.num_workers = config.evaluator.parallel_evaluations
        self.num_islands = config.database.num_islands

        logger.info(f"Initialized process parallel controller with {self.num_workers} workers")

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
            "checkpoint_interval": config.checkpoint_interval,
            "log_level": config.log_level,
            "log_dir": config.log_dir,
            "random_seed": config.random_seed,
            "diff_based_evolution": config.diff_based_evolution,
            "max_code_length": config.max_code_length,
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

        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None

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
        batch_size = min(self.num_workers * 2, max_iterations)

        # Submit initial batch - distribute across islands
        batch_per_island = max(1, batch_size // self.num_islands) if batch_size > 0 else 0
        current_iteration = start_iteration

        # Round-robin distribution across islands
        for island_id in range(self.num_islands):
            for _ in range(batch_per_island):
                if current_iteration < total_iterations:
                    future = self._submit_iteration(current_iteration, island_id)
                    if future:
                        pending_futures[current_iteration] = future
                        island_pending[island_id].append(current_iteration)
                    current_iteration += 1

        next_iteration = current_iteration
        completed_iterations = 0

        # Early stopping tracking
        early_stopping_enabled = self.config.early_stopping_patience is not None
        if early_stopping_enabled:
            best_score = float("-inf")
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
                    f"metric={self.config.early_stopping_metric}"
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

                if result.error:
                    logger.warning(f"Iteration {completed_iteration} error: {result.error}")
                elif result.child_program_dict:
                    # Reconstruct program from dict
                    child_program = Program(**result.child_program_dict)

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
                        # Retrieve parent program for trace logging
                        parent_program = (
                            self.database.get(result.parent_id) if result.parent_id else None
                        )
                        if parent_program:
                            # Determine island ID
                            island_id = child_program.metadata.get(
                                "island", self.database.current_island
                            )

                            self.evolution_tracer.log_trace(
                                iteration=completed_iteration,
                                parent_program=parent_program,
                                child_program=child_program,
                                prompt=result.prompt,
                                llm_response=result.llm_response,
                                artifacts=result.artifacts,
                                island_id=island_id,
                                metadata={
                                    "iteration_time": result.iteration_time,
                                    "changes": child_program.metadata.get("changes", ""),
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
                            break

                    # Check early stopping
                    if early_stopping_enabled and child_program.metrics:
                        # Get the metric to track for early stopping
                        current_score = None
                        if self.config.early_stopping_metric in child_program.metrics:
                            current_score = child_program.metrics[self.config.early_stopping_metric]
                        elif self.config.early_stopping_metric == "combined_score":
                            # Default metric not found, use safe average (standard pattern)
                            current_score = safe_numeric_average(child_program.metrics)
                        else:
                            # User specified a custom metric that doesn't exist
                            logger.warning(
                                f"Early stopping metric '{self.config.early_stopping_metric}' not found, using safe numeric average"
                            )
                            current_score = safe_numeric_average(child_program.metrics)

                        if current_score is not None and isinstance(current_score, (int, float)):
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
                                ):
                                    self.early_stopping_triggered = True
                                    logger.info(
                                        f"🛑 Early stopping triggered at iteration {completed_iteration}: "
                                        f"No improvement for {iterations_without_improvement} iterations "
                                        f"(best score: {best_score:.4f})"
                                    )
                                    break

                            else:
                                # Event-based early stopping
                                if current_score == self.config.convergence_threshold:
                                    best_score = current_score
                                    logger.info(
                                        f"🛑 Early stopping (event-based) triggered at iteration {completed_iteration}: "
                                        f"Task successfully solved with score {best_score:.4f}."
                                    )
                                    self.early_stopping_triggered = True
                                    break

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

            # Remove completed iteration from island tracking
            for island_id, iteration_list in island_pending.items():
                if completed_iteration in iteration_list:
                    iteration_list.remove(completed_iteration)
                    break

            # Submit next iterations maintaining island balance
            for island_id in range(self.num_islands):
                if (
                    len(island_pending[island_id]) < batch_per_island
                    and next_iteration < total_iterations
                    and not self.shutdown_event.is_set()
                ):
                    future = self._submit_iteration(next_iteration, island_id)
                    if future:
                        pending_futures[next_iteration] = future
                        island_pending[island_id].append(next_iteration)
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
        elif self.shutdown_event.is_set():
            logger.info("✅ Evolution completed - Shutdown requested")
        else:
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
            parent, inspirations = self.database.sample_from_island(
                island_id=target_island, num_inspirations=self.config.prompt.num_top_programs
            )

            # Create database snapshot
            db_snapshot = self._create_database_snapshot()
            db_snapshot["sampling_island"] = target_island  # Mark which island this is for

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
