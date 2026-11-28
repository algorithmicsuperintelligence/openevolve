"""
Evaluation system for OpenEvolve
"""

import asyncio
import functools
import importlib.util
import inspect
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from openevolve.config import EvaluatorConfig
from openevolve.database import ProgramDatabase
from openevolve.evaluation_result import EvaluationResult
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.prompt.sampler import PromptSampler
from openevolve.utils.async_utils import TaskPool, run_in_executor
from openevolve.utils.format_utils import format_metrics_safe
from openevolve.utils.metrics_utils import get_fitness_score, is_failed_evaluation

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluates programs and assigns scores

    The evaluator is responsible for executing programs, measuring their performance,
    and assigning scores based on the evaluation criteria.
    """

    def __init__(
        self,
        config: EvaluatorConfig,
        evaluation_file: str,
        llm_ensemble: Optional[LLMEnsemble] = None,
        prompt_sampler: Optional[PromptSampler] = None,
        database: Optional[ProgramDatabase] = None,
        suffix: Optional[str] = ".py",
        feature_dimensions: Optional[List[str]] = None,
    ):
        self.config = config
        self.evaluation_file = evaluation_file
        self.program_suffix = suffix
        self.llm_ensemble = llm_ensemble
        self.prompt_sampler = prompt_sampler
        self.database = database
        self.feature_dimensions = (
            list(feature_dimensions) if feature_dimensions is not None else None
        )

        # Keyword arguments forwarded to the user's evaluate() and cascade stage
        # functions; populated by _load_evaluation_function() based on
        # config.evaluator.evaluator_args and the signatures of loaded functions.
        self._evaluator_kwargs: Dict[str, Any] = {}
        self._stage_evaluator_kwargs: Dict[str, Dict[str, Any]] = {}
        self.evaluation_module: Any = None
        self._evaluation_module_stat: Optional[Tuple[int, int]] = None

        # Create a task pool for parallel evaluation
        self.task_pool = TaskPool(max_concurrency=config.parallel_evaluations)

        # Dedicated executor (not the loop's default) so a timed-out evaluation's
        # orphaned thread can't block asyncio.run()'s shutdown_default_executor().
        self._executor = self._new_executor()

        # Set up evaluation function if file exists
        self._load_evaluation_function()

        # Pending artifacts storage for programs
        self._pending_artifacts: Dict[str, Dict[str, Union[str, bytes]]] = {}

        logger.info(f"Initialized evaluator with {evaluation_file}")

    def _new_executor(self) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(
            max_workers=max(1, self.config.parallel_evaluations),
            thread_name_prefix="openevolve-eval",
        )

    async def _run_with_timeout(self, func: Callable, *args: Any) -> Any:
        """
        Run a blocking evaluation function in the executor, bounded by config.timeout

        A timed-out call keeps running in its thread because Python cannot stop it.
        That thread would hold an executor slot indefinitely (forever for an infinite
        loop), so later evaluations would queue behind it and time out without ever
        starting. The executor is replaced on timeout so they get a free slot.

        Raises:
            asyncio.TimeoutError: If the function exceeds the timeout
        """
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(self._executor, func, *args), timeout=self.config.timeout
            )
        except asyncio.TimeoutError:
            stuck_executor = self._executor
            self._executor = self._new_executor()
            stuck_executor.shutdown(wait=False)
            raise

    def _load_evaluation_function(self) -> None:
        """Load the evaluation function from the evaluation file"""
        if not os.path.exists(self.evaluation_file):
            raise ValueError(f"Evaluation file {self.evaluation_file} not found")

        try:
            # Add the evaluation file's directory to Python path so it can import local modules
            eval_dir = os.path.dirname(os.path.abspath(self.evaluation_file))
            if eval_dir not in sys.path:
                sys.path.insert(0, eval_dir)
                logger.debug(f"Added {eval_dir} to Python path for local imports")

            spec = importlib.util.spec_from_file_location("evaluation_module", self.evaluation_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load spec from {self.evaluation_file}")

            module = importlib.util.module_from_spec(spec)
            sys.modules["evaluation_module"] = module
            spec.loader.exec_module(module)

            if not hasattr(module, "evaluate"):
                raise AttributeError(
                    f"Evaluation file {self.evaluation_file} does not contain an 'evaluate' function"
                )

            self.evaluation_module = module
            try:
                stat_result = os.stat(self.evaluation_file)
                self._evaluation_module_stat = (stat_result.st_mtime_ns, stat_result.st_size)
            except OSError:
                self._evaluation_module_stat = None

            self.evaluate_function = module.evaluate
            has_cascade_stages = bool(
                self.config.cascade_evaluation and hasattr(module, "evaluate_stage1")
            )
            if has_cascade_stages:
                self._evaluator_kwargs = self._resolve_stage_kwargs(module.evaluate)
                self._stage_evaluator_kwargs = self._resolve_all_cascade_stage_kwargs(module)
            else:
                self._evaluator_kwargs = self._resolve_evaluator_kwargs(module.evaluate)
                self._stage_evaluator_kwargs = {}
            logger.info(f"Successfully loaded evaluation function from {self.evaluation_file}")

            # Validate cascade configuration
            self._validate_cascade_configuration(module)
        except Exception as e:
            logger.error(f"Error loading evaluation function: {str(e)}")
            raise

    def _get_evaluation_module(self) -> Any:
        """Return the cached evaluation module, reloading only if the file changed on disk."""
        try:
            stat_result = os.stat(self.evaluation_file)
            current_stat: Optional[Tuple[int, int]] = (
                stat_result.st_mtime_ns,
                stat_result.st_size,
            )
        except OSError:
            current_stat = None

        if self.evaluation_module is not None and (
            current_stat is None or current_stat == self._evaluation_module_stat
        ):
            return self.evaluation_module

        self._load_evaluation_function()
        return self.evaluation_module

    def _get_feature_dimensions(self) -> List[str]:
        """Return configured MAP-Elites feature dimensions from explicit args or database."""
        explicit = getattr(self, "feature_dimensions", None)
        if explicit is not None:
            return list(explicit)
        database = getattr(self, "database", None)
        db_config = getattr(database, "config", None)
        dims = getattr(db_config, "feature_dimensions", None)
        return list(dims) if dims else []

    def _resolve_evaluator_kwargs(self, evaluate_function: Callable) -> Dict[str, Any]:
        """
        Resolve the keyword arguments to forward to the user's evaluate() function.

        Uses inspect.signature to verify that evaluate() actually accepts the
        configured evaluator_args (GitHub issue #474):

        - Evaluators accepting **kwargs get all configured kwargs.
        - Evaluators whose named parameters cover the configured keys get them
          forwarded on every call.
        - Evaluators that cannot accept a configured key fail here, at startup,
          with an actionable error instead of failing on every evaluation.

        Evaluators with no configured evaluator_args (the common single
        `program_path` argument case) are unaffected and keep their existing
        behavior.

        Args:
            evaluate_function: The user-provided evaluate() function

        Returns:
            Dict of keyword arguments to forward (empty when nothing configured)

        Raises:
            ValueError: If evaluate() cannot accept a configured evaluator_arg
        """
        evaluator_args = self.config.evaluator_args or {}
        if not evaluator_args:
            return {}

        function_name = getattr(evaluate_function, "__name__", "evaluate")
        try:
            signature = inspect.signature(evaluate_function)
        except (TypeError, ValueError):
            # Some callables (e.g. certain builtins) expose no signature; forward
            # as-is and let the call surface any mismatch
            logger.warning(
                f"Could not inspect the signature of '{function_name}'; forwarding "
                f"evaluator_args keys: {sorted(evaluator_args.keys())}"
            )
            return dict(evaluator_args)

        # Guard against kwargs that cannot be forwarded together with the program
        # path passed positionally (e.g. evaluator_args={"program_path": ...} for
        # evaluate(program_path)). Runs before the **kwargs shortcut below: binding
        # mirrors the actual call functools.partial(evaluate, **args)(program_path),
        # so it also rejects a configured key shadowing a positional parameter of a
        # **kwargs evaluator (TypeError: multiple values for argument).
        try:
            signature.bind(object(), **evaluator_args)
        except TypeError as e:
            raise ValueError(
                f"Configured evaluator_args {sorted(evaluator_args.keys())} cannot be "
                f"forwarded to evaluation function '{function_name}' in "
                f"{self.evaluation_file}: {e}. Its signature is '{function_name}{signature}'."
            ) from e

        if any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
        ):
            # **kwargs swallows everything
            logger.info(
                f"Forwarding evaluator_args keys to '{function_name}': "
                f"{sorted(evaluator_args.keys())}"
            )
            return dict(evaluator_args)

        named_params = {
            name
            for name, param in signature.parameters.items()
            if param.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        unsupported = sorted(set(evaluator_args.keys()) - named_params)
        if unsupported:
            raise ValueError(
                f"Evaluation function '{function_name}' in {self.evaluation_file} does not "
                f"accept the configured evaluator_args keys: {unsupported}. Its signature is "
                f"'{function_name}{signature}'. Add the parameters to evaluate(), accept "
                f"**kwargs, or remove the keys from evaluator.evaluator_args / --evaluator-args."
            )

        logger.info(
            f"Forwarding evaluator_args keys to '{function_name}': "
            f"{sorted(evaluator_args.keys())}"
        )
        return dict(evaluator_args)

    def _resolve_stage_kwargs(self, stage_function: Callable) -> Dict[str, Any]:
        """
        Resolve the subset of evaluator_args accepted by a cascade stage function.

        In cascade mode, stage 1 is often a lightweight syntax/smoke check taking only
        `program_path` while later stages accept parameterized `evaluator_args`. Each stage
        receives the configured keys its signature accepts (or all keys if it accepts
        `**kwargs`), while still rejecting positional-parameter collisions at startup.
        """
        evaluator_args = self.config.evaluator_args or {}
        if not evaluator_args:
            return {}

        function_name = getattr(stage_function, "__name__", "evaluate_stage")
        try:
            signature = inspect.signature(stage_function)
        except (TypeError, ValueError):
            logger.warning(
                f"Could not inspect the signature of '{function_name}'; forwarding "
                f"evaluator_args keys: {sorted(evaluator_args.keys())}"
            )
            return dict(evaluator_args)

        has_var_keyword = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
        )
        if has_var_keyword:
            forwarded = dict(evaluator_args)
        else:
            named_params = {
                name
                for name, param in signature.parameters.items()
                if param.kind
                in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            }
            forwarded = {k: v for k, v in evaluator_args.items() if k in named_params}

        try:
            signature.bind(object(), **forwarded)
        except TypeError as e:
            raise ValueError(
                f"Configured evaluator_args {sorted(evaluator_args.keys())} cannot be "
                f"forwarded to evaluation function '{function_name}' in "
                f"{self.evaluation_file}: {e}. Its signature is '{function_name}{signature}'."
            ) from e

        if forwarded:
            logger.info(
                f"Forwarding evaluator_args keys to '{function_name}': {sorted(forwarded.keys())}"
            )
        return forwarded

    def _resolve_all_cascade_stage_kwargs(self, module: Any) -> Dict[str, Dict[str, Any]]:
        """Validate and resolve evaluator_args across all defined cascade stages."""
        stage_names = [
            name
            for name in ("evaluate_stage1", "evaluate_stage2", "evaluate_stage3")
            if hasattr(module, name) and callable(getattr(module, name))
        ]
        resolved: Dict[str, Dict[str, Any]] = {}
        for stage_name in stage_names:
            resolved[stage_name] = self._resolve_stage_kwargs(getattr(module, stage_name))

        evaluator_args = self.config.evaluator_args or {}
        if evaluator_args and stage_names:
            accepted_keys = set()
            for kwargs_for_stage in resolved.values():
                accepted_keys.update(kwargs_for_stage.keys())
            unsupported = sorted(set(evaluator_args.keys()) - accepted_keys)
            if unsupported:
                signatures = ", ".join(
                    f"{name}{inspect.signature(getattr(module, name))}" for name in stage_names
                )
                raise ValueError(
                    f"Cascade evaluation functions in {self.evaluation_file} do not "
                    f"accept the configured evaluator_args keys: {unsupported}. Stage signatures: "
                    f"{signatures}. Add the parameters to a cascade stage, accept **kwargs, "
                    f"or remove the keys from evaluator.evaluator_args / --evaluator-args."
                )

        return resolved

    def _validate_cascade_configuration(self, module) -> None:
        """
        Validate cascade evaluation configuration and warn about potential issues

        Args:
            module: The loaded evaluation module
        """
        if self.config.cascade_evaluation:
            # Check if cascade functions exist
            has_stage1 = hasattr(module, "evaluate_stage1")
            has_stage2 = hasattr(module, "evaluate_stage2")
            has_stage3 = hasattr(module, "evaluate_stage3")

            if not has_stage1:
                logger.warning(
                    f"Configuration has 'cascade_evaluation: true' but evaluator "
                    f"'{self.evaluation_file}' does not define 'evaluate_stage1' function. "
                    f"This will fall back to direct evaluation, making the cascade setting useless. "
                    f"Consider setting 'cascade_evaluation: false' or implementing cascade functions."
                )
            elif not (has_stage2 or has_stage3):
                logger.warning(
                    f"Evaluator '{self.evaluation_file}' defines 'evaluate_stage1' but no additional "
                    f"cascade stages (evaluate_stage2, evaluate_stage3). Consider implementing "
                    f"multi-stage evaluation for better cascade benefits."
                )
            else:
                logger.debug(
                    f"Cascade evaluation properly configured with available stage functions"
                )

    async def evaluate_program(
        self,
        program_code: str,
        program_id: str = "",
    ) -> Dict[str, float]:
        """
        Evaluate a program and return scores

        Args:
            program_code: Code to evaluate
            program_id: Optional ID for logging

        Returns:
            Dictionary of metric name to score
        """
        start_time = time.time()
        program_id_str = f" {program_id}" if program_id else ""

        # Check if artifacts are enabled
        artifacts_enabled = os.environ.get("ENABLE_ARTIFACTS", "true").lower() == "true"

        # Retry logic for evaluation
        last_exception = None
        for attempt in range(self.config.max_retries + 1):
            # Create a temporary file for the program
            with tempfile.NamedTemporaryFile(suffix=self.program_suffix, delete=False) as temp_file:
                temp_file.write(program_code.encode("utf-8"))
                temp_file_path = temp_file.name

            try:
                # Run evaluation
                if self.config.cascade_evaluation:
                    # Run cascade evaluation
                    result = await self._cascade_evaluate(temp_file_path)
                else:
                    # Run direct evaluation
                    result = await self._direct_evaluate(temp_file_path)

                # Process the result based on type
                eval_result = self._process_evaluation_result(result)

                # Check if this was a timeout and capture artifacts if enabled
                if artifacts_enabled and program_id and eval_result.metrics.get("timeout") is True:
                    if program_id not in self._pending_artifacts:
                        self._pending_artifacts[program_id] = {}

                    self._pending_artifacts[program_id].update(
                        {
                            "timeout": True,
                            "timeout_duration": self.config.timeout,
                            "failure_stage": "evaluation",
                            "error_type": "timeout",
                        }
                    )

                # Add LLM feedback if configured (only when programmatic evaluation succeeded,
                # so LLM quality scores cannot inflate failed or timed-out candidates).
                llm_eval_result = None
                if (
                    self.config.use_llm_feedback
                    and self.llm_ensemble
                    and not self._is_failed_evaluation(eval_result.metrics)
                ):
                    llm_result = await self._llm_evaluate(program_code, program_id=program_id)
                    llm_eval_result = self._process_evaluation_result(llm_result)

                    # Combine metrics
                    llm_scores = []
                    for name, value in llm_eval_result.metrics.items():
                        weighted_value = value * self.config.llm_feedback_weight
                        eval_result.metrics[f"llm_{name}"] = weighted_value
                        llm_scores.append(value)  # Use unweighted value for average

                    # Add average of LLM metrics
                    if llm_scores:
                        llm_average = sum(llm_scores) / len(llm_scores)
                        eval_result.metrics["llm_average"] = (
                            llm_average * self.config.llm_feedback_weight
                        )

                        # Recalculate combined_score if it exists
                        if "combined_score" in eval_result.metrics:
                            # Original combined_score is just accuracy
                            accuracy = eval_result.metrics["combined_score"]
                            if (
                                isinstance(accuracy, (int, float))
                                and not isinstance(accuracy, bool)
                                and not math.isnan(float(accuracy))
                            ):
                                # Combine with LLM average (70% accuracy, 30% LLM quality)
                                eval_result.metrics["combined_score"] = (
                                    float(accuracy) * 0.7 + llm_average * 0.3
                                )

                # Store artifacts if enabled and present
                if (
                    artifacts_enabled
                    and (
                        eval_result.has_artifacts()
                        or (llm_eval_result and llm_eval_result.has_artifacts())
                    )
                    and program_id
                ):
                    if program_id not in self._pending_artifacts:
                        self._pending_artifacts[program_id] = {}

                    # Merge eval_result artifacts with llm artifacts if they exist
                    if eval_result.has_artifacts():
                        self._pending_artifacts[program_id].update(eval_result.artifacts)
                        logger.debug(
                            f"Program{program_id_str} returned artifacts: "
                            f"{eval_result.artifacts}"
                        )

                    if llm_eval_result and llm_eval_result.has_artifacts():
                        self._pending_artifacts[program_id].update(llm_eval_result.artifacts)
                        logger.debug(
                            f"Program{program_id_str} returned LLM artifacts: "
                            f"{llm_eval_result.artifacts}"
                        )

                elapsed = time.time() - start_time
                logger.info(
                    f"Evaluated program{program_id_str} in {elapsed:.2f}s: "
                    f"{format_metrics_safe(eval_result.metrics)}"
                )

                # Return just metrics for backward compatibility
                return eval_result.metrics

            except asyncio.TimeoutError:
                # Handle timeout specially - don't retry, just return timeout result
                logger.warning(f"Evaluation timed out after {self.config.timeout}s")

                # Capture timeout artifacts if enabled
                if artifacts_enabled and program_id:
                    self._pending_artifacts[program_id] = {
                        "timeout": True,
                        "timeout_duration": self.config.timeout,
                        "failure_stage": "evaluation",
                        "error_type": "timeout",
                    }

                return {"error": 0.0, "timeout": True}

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Evaluation attempt {attempt + 1}/{self.config.max_retries + 1} failed for program{program_id_str}: {str(e)}"
                )
                traceback.print_exc()

                # Capture failure artifacts if enabled
                if artifacts_enabled and program_id:
                    self._pending_artifacts[program_id] = {
                        "stderr": str(e),
                        "traceback": traceback.format_exc(),
                        "failure_stage": "evaluation",
                        "attempt": attempt + 1,
                    }

                # If this is not the last attempt, wait a bit before retrying
                if attempt < self.config.max_retries:
                    await asyncio.sleep(1.0)  # Wait 1 second before retry

            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        # All retries failed
        logger.error(
            f"All evaluation attempts failed for program{program_id_str}. Last error: {str(last_exception)}"
        )
        return {"error": 0.0}

    def _process_evaluation_result(self, result: Any) -> EvaluationResult:
        """
        Process evaluation result to handle both dict and EvaluationResult returns

        Args:
            result: Raw result from evaluation function

        Returns:
            EvaluationResult instance
        """
        if isinstance(result, dict):
            # Backward compatibility - wrap dict in EvaluationResult
            return EvaluationResult.from_dict(result)
        elif isinstance(result, EvaluationResult):
            # New format - use directly
            return result
        else:
            # Error case - return error metrics
            logger.warning(f"Unexpected evaluation result type: {type(result)}")
            return EvaluationResult(metrics={"error": 0.0})

    def get_pending_artifacts(self, program_id: str) -> Optional[Dict[str, Union[str, bytes]]]:
        """
        Get and clear pending artifacts for a program

        Args:
            program_id: Program ID

        Returns:
            Artifacts dictionary or None if not found
        """
        return self._pending_artifacts.pop(program_id, None)

    async def _direct_evaluate(
        self, program_path: str
    ) -> Union[Dict[str, float], EvaluationResult]:
        """
        Directly evaluate a program using the evaluation function with timeout

        Args:
            program_path: Path to the program file

        Returns:
            Dictionary of metrics or EvaluationResult with metrics and artifacts

        Raises:
            asyncio.TimeoutError: If evaluation exceeds timeout
            Exception: If evaluation function raises an exception
        """

        # Let exceptions bubble up for retry handling
        result = await self._run_with_timeout(
            functools.partial(self.evaluate_function, **self._evaluator_kwargs), program_path
        )

        # Return result as-is to be processed by _process_evaluation_result
        # This supports both dict and EvaluationResult returns, just like _cascade_evaluate
        return result

    @staticmethod
    def _is_failed_evaluation(metrics: Dict[str, Any]) -> bool:
        """Return True when metrics represent a failed or timed-out evaluation."""
        return is_failed_evaluation(metrics)

    def _failure_combined_score(self, prior_metrics: Dict[str, Any]) -> float:
        """
        Compute a safe combined_score sentinel when a later cascade stage fails or times out.

        On non-negative fitness scales, 0.0 matches Stage-1 and direct failure scores.
        On negative fitness scales (e.g. minimization objectives returning -MSE), 0.0 would
        outscore valid negative solutions, so -inf is used instead.
        """
        if not prior_metrics:
            return 0.0
        prior_fitness = get_fitness_score(prior_metrics, self._get_feature_dimensions())
        if prior_fitness < 0.0:
            return float("-inf")
        return 0.0

    def _build_cascade_stage_failure(
        self,
        prior_result: EvaluationResult,
        stage_passed_key: str,
        extra_artifacts: Dict[str, Any],
        timed_out: bool = False,
    ) -> EvaluationResult:
        """Build a non-mutating failure EvaluationResult when a later cascade stage fails."""
        failure_metrics: Dict[str, Any] = {}
        for name, value in prior_result.metrics.items():
            if isinstance(value, (int, float)) and name != "error":
                failure_metrics[name] = value if isinstance(value, bool) else float(value)
        failure_metrics["combined_score"] = self._failure_combined_score(prior_result.metrics)
        failure_metrics[stage_passed_key] = 0.0
        failure_metrics["error"] = 0.0
        if timed_out:
            failure_metrics["timeout"] = True

        failure_artifacts = dict(prior_result.artifacts)
        failure_artifacts.update(extra_artifacts)
        return EvaluationResult(metrics=failure_metrics, artifacts=failure_artifacts)

    def _merge_cascade_stage_metrics(
        self,
        prior_metrics: Dict[str, Any],
        stage_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Merge metrics from a cascade stage while preserving failure indicators and
        preventing stale earlier-stage combined_score values from surviving a failed stage.
        """
        merged_metrics: Dict[str, Any] = {}
        for name, value in prior_metrics.items():
            if isinstance(value, (int, float)) and name != "error":
                merged_metrics[name] = value if isinstance(value, bool) else float(value)

        for name, value in stage_metrics.items():
            if isinstance(value, (int, float)) and name != "error":
                merged_metrics[name] = value if isinstance(value, bool) else float(value)

        if self._is_failed_evaluation(stage_metrics):
            if "error" in stage_metrics and stage_metrics["error"] is not None:
                err_val = stage_metrics["error"]
                if isinstance(err_val, (int, float)) and not isinstance(err_val, bool):
                    merged_metrics["error"] = float(err_val)
                else:
                    merged_metrics["error"] = err_val
            else:
                merged_metrics.setdefault("error", 0.0)

            stage_combined = stage_metrics.get("combined_score")
            has_valid_stage_combined = (
                "combined_score" in stage_metrics
                and isinstance(stage_combined, (int, float))
                and not isinstance(stage_combined, bool)
                and not math.isnan(float(stage_combined))
            )
            if not has_valid_stage_combined:
                merged_metrics["combined_score"] = self._failure_combined_score(prior_metrics)

        return merged_metrics

    def _invoke_stage_callable(self, module: Any, stage_name: str) -> Callable:
        """Bind resolved evaluator_args for a cascade stage function."""
        stage_fn = getattr(module, stage_name)
        stage_kwargs = self._stage_evaluator_kwargs.get(stage_name)
        if stage_kwargs is None and self.config.evaluator_args:
            stage_kwargs = self._resolve_stage_kwargs(stage_fn)
        if stage_kwargs:
            return functools.partial(stage_fn, **stage_kwargs)
        return stage_fn

    async def _cascade_evaluate(
        self, program_path: str
    ) -> Union[Dict[str, float], EvaluationResult]:
        """
        Run cascade evaluation with increasingly challenging test cases

        Args:
            program_path: Path to the program file

        Returns:
            Dictionary of metrics or EvaluationResult with metrics and artifacts
        """
        try:
            module = self._get_evaluation_module()
            if module is None:
                return await self._direct_evaluate(program_path)

            # Check if cascade functions exist
            if not hasattr(module, "evaluate_stage1"):
                return await self._direct_evaluate(program_path)

            # Run first stage with timeout
            try:
                stage1_callable = self._invoke_stage_callable(module, "evaluate_stage1")
                stage1_result = await self._run_with_timeout(stage1_callable, program_path)
                stage1_eval_result = self._process_evaluation_result(stage1_result)
            except asyncio.TimeoutError:
                logger.warning(f"Stage 1 evaluation timed out after {self.config.timeout}s")
                return EvaluationResult(
                    metrics={"stage1_passed": 0.0, "error": 0.0, "timeout": True},
                    artifacts={
                        "failure_stage": "stage1",
                        "timeout": True,
                    },
                )
            except Exception as e:
                logger.error(f"Error in stage 1 evaluation: {str(e)}")
                # Capture stage 1 failure with enhanced context
                error_context = self._create_cascade_error_context("stage1", e)
                return EvaluationResult(
                    metrics={"stage1_passed": 0.0, "error": 0.0},
                    artifacts={
                        "stderr": str(e),
                        "traceback": traceback.format_exc(),
                        **error_context,
                    },
                )

            # Check threshold
            if not self.config.cascade_thresholds or not self._passes_threshold(
                stage1_eval_result.metrics, self.config.cascade_thresholds[0]
            ):
                return stage1_eval_result

            # Check if second stage exists
            if not hasattr(module, "evaluate_stage2"):
                return stage1_eval_result

            # Run second stage with timeout
            try:
                stage2_callable = self._invoke_stage_callable(module, "evaluate_stage2")
                stage2_result = await self._run_with_timeout(stage2_callable, program_path)
                stage2_eval_result = self._process_evaluation_result(stage2_result)
            except asyncio.TimeoutError:
                logger.warning(f"Stage 2 evaluation timed out after {self.config.timeout}s")
                return self._build_cascade_stage_failure(
                    stage1_eval_result,
                    "stage2_passed",
                    {
                        "stage2_timeout": True,
                        "failure_stage": "stage2",
                    },
                    timed_out=True,
                )
            except Exception as e:
                logger.error(f"Error in stage 2 evaluation: {str(e)}")
                return self._build_cascade_stage_failure(
                    stage1_eval_result,
                    "stage2_passed",
                    {
                        "stage2_stderr": str(e),
                        "stage2_traceback": traceback.format_exc(),
                        "failure_stage": "stage2",
                    },
                )

            # Merge results from stage 1 and 2
            merged_metrics = self._merge_cascade_stage_metrics(
                stage1_eval_result.metrics, stage2_eval_result.metrics
            )

            # Merge artifacts
            merged_artifacts = {}
            merged_artifacts.update(stage1_eval_result.artifacts)
            merged_artifacts.update(stage2_eval_result.artifacts)

            merged_result = EvaluationResult(metrics=merged_metrics, artifacts=merged_artifacts)

            # Check threshold for stage 3
            if len(self.config.cascade_thresholds) < 2 or not self._passes_threshold(
                merged_result.metrics, self.config.cascade_thresholds[1]
            ):
                return merged_result

            # Check if third stage exists
            if not hasattr(module, "evaluate_stage3"):
                return merged_result

            # Run third stage with timeout
            try:
                stage3_callable = self._invoke_stage_callable(module, "evaluate_stage3")
                stage3_result = await self._run_with_timeout(stage3_callable, program_path)
                stage3_eval_result = self._process_evaluation_result(stage3_result)
            except asyncio.TimeoutError:
                logger.warning(f"Stage 3 evaluation timed out after {self.config.timeout}s")
                return self._build_cascade_stage_failure(
                    merged_result,
                    "stage3_passed",
                    {
                        "stage3_timeout": True,
                        "failure_stage": "stage3",
                    },
                    timed_out=True,
                )
            except Exception as e:
                logger.error(f"Error in stage 3 evaluation: {str(e)}")
                return self._build_cascade_stage_failure(
                    merged_result,
                    "stage3_passed",
                    {
                        "stage3_stderr": str(e),
                        "stage3_traceback": traceback.format_exc(),
                        "failure_stage": "stage3",
                    },
                )

            # Merge stage 3 results
            merged_stage3_metrics = self._merge_cascade_stage_metrics(
                merged_result.metrics, stage3_eval_result.metrics
            )
            merged_stage3_artifacts = dict(merged_result.artifacts)
            merged_stage3_artifacts.update(stage3_eval_result.artifacts)

            return EvaluationResult(
                metrics=merged_stage3_metrics, artifacts=merged_stage3_artifacts
            )

        except Exception as e:
            logger.error(f"Error in cascade evaluation: {str(e)}")
            # Return proper cascade failure result with enhanced context
            error_context = self._create_cascade_error_context("cascade_setup", e)
            return EvaluationResult(
                metrics={"stage1_passed": 0.0, "error": 0.0},
                artifacts={
                    "stderr": str(e),
                    "traceback": traceback.format_exc(),
                    **error_context,
                },
            )

    async def _llm_evaluate(self, program_code: str, program_id: str = "") -> Dict[str, float]:
        """
        Use LLM to evaluate code quality

        Args:
            program_code: Code to evaluate
            program_id: Optional ID for logging

        Returns:
            Dictionary of metric name to score
        """
        if not self.llm_ensemble:
            return {}

        try:
            # Create prompt for LLM
            feature_dimensions = self._get_feature_dimensions()
            prompt = self.prompt_sampler.build_prompt(
                current_program=program_code,
                template_key="evaluation",
                feature_dimensions=feature_dimensions,
            )

            # Get LLM response
            responses = await self.llm_ensemble.generate_all_with_context(
                prompt["system"], [{"role": "user", "content": prompt["user"]}]
            )

            # Log prompt and response to database
            if self.database and program_id:
                self.database.log_prompt(
                    program_id=program_id,
                    template_key="evaluation",
                    prompt=prompt,
                    responses=responses,
                )

            # Extract JSON from response
            try:
                # Try to find JSON block
                json_pattern = r"```json\n(.*?)\n```"
                import re

                artifacts = {}
                weighted_metrics = {}
                metric_weights = {}
                for i, response in enumerate(responses):
                    weight = self.llm_ensemble.weights[i] if self.llm_ensemble.weights else 1.0
                    if weight <= 0:
                        continue

                    try:
                        json_match = re.search(json_pattern, response, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                        else:
                            json_str = response
                            start_idx = json_str.find("{")
                            end_idx = json_str.rfind("}") + 1
                            if start_idx >= 0 and end_idx > start_idx:
                                json_str = json_str[start_idx:end_idx]

                        result = json.loads(json_str)
                        if not isinstance(result, dict):
                            raise ValueError("LLM evaluation must return a JSON object")
                    except (ValueError, TypeError) as error:
                        logger.warning("Skipping invalid LLM judge response %d: %s", i, error)
                        continue

                    for name, value in result.items():
                        # Flags are metadata, and invalid numeric feedback must
                        # not poison valid scores from this or another judge.
                        if isinstance(value, bool) or not isinstance(value, (int, float)):
                            artifacts[name] = value
                            continue
                        try:
                            score = float(value)
                        except (ValueError, TypeError, OverflowError):
                            logger.warning(
                                "Skipping invalid LLM judge metric %s from judge %d", name, i
                            )
                            continue
                        if not math.isfinite(score):
                            logger.warning(
                                "Skipping non-finite LLM judge metric %s from judge %d", name, i
                            )
                            continue
                        weighted_metrics[name] = weighted_metrics.get(name, 0.0) + score * weight
                        metric_weights[name] = metric_weights.get(name, 0.0) + weight

                # A missing or invalid response is not a zero score. Normalize
                # each metric using only the judges that actually supplied it.
                avg_metrics = {
                    name: value / metric_weights[name] for name, value in weighted_metrics.items()
                }

                return EvaluationResult(
                    metrics=avg_metrics,
                    artifacts=artifacts,
                )

            except Exception as e:
                logger.warning(f"Error parsing LLM response: {str(e)}")
                return {}

        except Exception as e:
            logger.error(f"Error in LLM evaluation: {str(e)}")
            traceback.print_exc()
            return {}

    def _create_cascade_error_context(self, stage: str, error: Exception) -> dict:
        """
        Create rich error context for cascade failures

        Args:
            stage: The stage where the error occurred
            error: The exception that was raised

        Returns:
            Dictionary with enhanced error context
        """
        import time

        return {
            "failure_stage": stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": time.time(),
            "cascade_config": self.config.cascade_evaluation,
            "cascade_thresholds": getattr(self.config, "cascade_thresholds", []),
            "timeout_config": self.config.timeout,
            "evaluation_file": self.evaluation_file,
        }

    def _passes_threshold(self, metrics: Dict[str, float], threshold: float) -> bool:
        """
        Check if metrics pass a threshold

        Uses 'combined_score' if available (for consistency with evolution),
        otherwise falls back to averaging all numeric metrics except 'error'

        Args:
            metrics: Dictionary of metric name to score
            threshold: Threshold to pass

        Returns:
            True if metrics pass threshold
        """
        if not metrics:
            return False

        if self._is_failed_evaluation(metrics):
            return False

        # Use combined_score if available - this is what evolution uses.
        # Ensure combined_score is genuine numeric data rather than a boolean failure flag
        # (e.g. {"combined_score": False}), and verify it is not NaN.
        if "combined_score" in metrics:
            score = metrics.get("combined_score")
            # In Python, bool is a subclass of int. A boolean flag (e.g., False)
            # must not be cast to 0.0, which could incorrectly pass a threshold <= 0.0.
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                try:
                    float_score = float(score)
                    if not math.isnan(float_score):
                        return float_score >= threshold
                except (TypeError, ValueError, OverflowError):
                    pass
            return False

        # Fallback: average numeric fitness metrics except 'error' and MAP-Elites
        # feature_dimensions (matching get_fitness_score so behavioral coordinates
        # such as complexity or prompt_length do not skew cascade stage gating).
        feature_dims = set(self._get_feature_dimensions())
        fitness_metrics = []
        all_numeric_metrics = []
        for name, value in metrics.items():
            # Flags are metadata, not scores (bool is a subclass of int).
            if name != "error" and isinstance(value, (int, float)) and not isinstance(value, bool):
                try:
                    fval = float(value)
                except (TypeError, ValueError, OverflowError):
                    logger.warning(f"Skipping non-numeric metric: {name}={value}")
                    continue
                if not math.isnan(fval):
                    all_numeric_metrics.append(fval)
                    if name not in feature_dims:
                        fitness_metrics.append(fval)

        valid_metrics = fitness_metrics if fitness_metrics else all_numeric_metrics
        if not valid_metrics:
            return False

        avg_score = sum(valid_metrics) / len(valid_metrics)
        return avg_score >= threshold

    async def evaluate_multiple(
        self,
        programs: List[Tuple[str, str]],
    ) -> List[Dict[str, float]]:
        """
        Evaluate multiple programs in parallel

        Args:
            programs: List of (program_code, program_id) tuples

        Returns:
            List of metric dictionaries
        """
        tasks = [
            self.task_pool.create_task(self.evaluate_program, program_code, program_id)
            for program_code, program_id in programs
        ]

        return await asyncio.gather(*tasks)
