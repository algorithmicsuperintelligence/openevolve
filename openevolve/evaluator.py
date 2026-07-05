"""
Evaluation system for OpenEvolve
"""

import asyncio
import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import traceback

from openevolve.config import EvaluatorConfig
from openevolve.database import ProgramDatabase
from openevolve.evaluation_result import EvaluationResult, EvaluatorRepairRequest
from openevolve.database import ProgramDatabase
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.utils.async_utils import TaskPool, run_in_executor
from openevolve.prompt.sampler import PromptSampler
from openevolve.utils.format_utils import format_metrics_safe

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
        repair_llm_ensemble: Optional[LLMEnsemble] = None,
    ):
        self.config = config
        self.evaluation_file = evaluation_file
        self.program_suffix = suffix
        self.llm_ensemble = llm_ensemble
        # Separate ensemble for LLM-based code repair; falls back to the main
        # evaluator ensemble (llm_ensemble) when not provided.
        self.repair_llm_ensemble = repair_llm_ensemble or llm_ensemble
        self.prompt_sampler = prompt_sampler
        self.database = database

        # Create a task pool for parallel evaluation
        self.task_pool = TaskPool(max_concurrency=config.parallel_evaluations)

        # Set up evaluation function if file exists
        self._load_evaluation_function()

        # Pending artifacts storage for programs
        self._pending_artifacts: Dict[str, Dict[str, Union[str, bytes]]] = {}

        # Pending repairs: program_id → repaired source code.
        # Populated by _attempt_repair when repair succeeds; consumed by
        # iteration.py / process_parallel.py via get_pending_repair().
        self._pending_repairs: Dict[str, str] = {}

        logger.info(f"Initialized evaluator with {evaluation_file}")

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

            self.evaluate_function = module.evaluate
            logger.info(f"Successfully loaded evaluation function from {self.evaluation_file}")

            # Validate cascade configuration
            self._validate_cascade_configuration(module)
        except Exception as e:
            logger.error(f"Error loading evaluation function: {str(e)}")
            raise

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

                # Add LLM feedback if configured
                llm_eval_result = None
                if self.config.use_llm_feedback and self.llm_ensemble:
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
                            # Combine with LLM average (70% accuracy, 30% LLM quality)
                            eval_result.metrics["combined_score"] = (
                                accuracy * 0.7 + llm_average * 0.3
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

            except EvaluatorRepairRequest as repair_req:
                # The user evaluator signalled that the code needs LLM repair
                # (e.g. a compilation failure).  Attempt repair if configured;
                # otherwise fall through to the standard zero-score path.
                if self.config.repair_on_failure and self.llm_ensemble:
                    repaired_metrics = await self._attempt_repair(repair_req, program_id)
                    if repaired_metrics is not None:
                        return repaired_metrics
                # Repair disabled, not configured, or all attempts exhausted.
                logger.warning(
                    f"Repair {'failed' if self.config.repair_on_failure else 'disabled'} "
                    f"for program{program_id_str}: {repair_req}"
                )
                if artifacts_enabled and program_id:
                    if program_id not in self._pending_artifacts:
                        self._pending_artifacts[program_id] = {}
                    self._pending_artifacts[program_id].update({
                        "compile_error": str(repair_req),
                        "repair_context": repair_req.repair_context,
                    })
                return repair_req.fallback_metrics

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

    def get_pending_repair(self, program_id: str) -> Optional[str]:
        """
        Get and clear the repaired source code for a program, if one exists.

        Returns the repaired code string when a previous ``_attempt_repair``
        call succeeded, or ``None`` when no repair was performed.  The entry is
        removed from the internal store on first read (one-shot).

        Args:
            program_id: Program ID used during evaluation.

        Returns:
            Repaired source code string, or ``None``.
        """
        return self._pending_repairs.pop(program_id, None)

    async def _attempt_repair(
        self,
        repair_req: EvaluatorRepairRequest,
        program_id: str,
    ) -> Optional[Dict[str, float]]:
        """
        Attempt to repair broken code via the LLM, then re-evaluate.

        Loops up to ``config.max_repair_attempts`` times.  On success the
        repaired code is stored in ``_pending_repairs[program_id]`` and the
        repair history is added to ``_pending_artifacts[program_id]`` so that
        ``iteration.py`` can move both into ``Program.metadata``.

        Args:
            repair_req: The ``EvaluatorRepairRequest`` raised by the evaluator.
            program_id: Program ID for artifact/repair storage.

        Returns:
            Metrics dict from the successfully repaired evaluation, or ``None``
            if all repair attempts failed.
        """
        artifacts_enabled = os.environ.get("ENABLE_ARTIFACTS", "true").lower() == "true"
        broken_code = repair_req.broken_code
        error_message = str(repair_req)
        repair_context = repair_req.repair_context
        language = repair_req.language
        repair_history: List[Dict] = []

        for attempt in range(1, self.config.max_repair_attempts + 1):
            logger.info(
                f"Repair attempt {attempt}/{self.config.max_repair_attempts} "
                f"for program {program_id} (language={language})"
            )

            repaired_code = await self._repair_code(
                broken_code=broken_code,
                error_message=error_message,
                repair_context=repair_context,
                language=language,
            )
            if repaired_code is None:
                logger.warning(f"Repair attempt {attempt}: LLM returned no parseable code")
                repair_history.append({
                    "attempt": attempt,
                    "error": error_message,
                    "repair_error": "LLM returned no parseable code",
                    "succeeded": False,
                })
                break

            # Write the repaired code to a temp file and re-evaluate
            with tempfile.NamedTemporaryFile(
                suffix=self.program_suffix, delete=False
            ) as tmp:
                tmp.write(repaired_code.encode("utf-8"))
                tmp_path = tmp.name

            try:
                result = await self._direct_evaluate(tmp_path)
                eval_result = self._process_evaluation_result(result)

                # Success — store the repaired code and history
                repair_history.append({
                    "attempt": attempt,
                    "error": None,
                    "succeeded": True,
                })
                logger.info(
                    f"Repair succeeded on attempt {attempt} for program {program_id}"
                )
                self._pending_repairs[program_id] = repaired_code
                if artifacts_enabled and program_id:
                    if program_id not in self._pending_artifacts:
                        self._pending_artifacts[program_id] = {}
                    self._pending_artifacts[program_id]["repair_history"] = repair_history
                    if eval_result.has_artifacts():
                        self._pending_artifacts[program_id].update(eval_result.artifacts)

                elapsed = 0.0  # timing already handled by outer evaluate_program
                logger.info(
                    f"Repaired program {program_id}: "
                    f"{format_metrics_safe(eval_result.metrics)}"
                )
                return eval_result.metrics

            except EvaluatorRepairRequest as next_req:
                # Re-evaluation raised another repair request — prepare next loop
                error_message = str(next_req)
                repair_context = next_req.repair_context
                broken_code = next_req.broken_code
                repair_history.append({
                    "attempt": attempt,
                    "error": error_message,
                    "succeeded": False,
                })
            except Exception as exc:
                error_message = str(exc)
                repair_history.append({
                    "attempt": attempt,
                    "error": error_message,
                    "succeeded": False,
                })
                logger.warning(f"Repair attempt {attempt} raised exception: {exc}")
                break
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        # All attempts exhausted
        logger.warning(
            f"All {self.config.max_repair_attempts} repair attempt(s) failed "
            f"for program {program_id}"
        )
        if artifacts_enabled and program_id:
            if program_id not in self._pending_artifacts:
                self._pending_artifacts[program_id] = {}
            self._pending_artifacts[program_id].update({
                "compile_error": str(repair_req),
                "repair_context": repair_req.repair_context,
                "repair_history": repair_history,
                "repair_failed": True,
            })
        return None

    async def _repair_code(
        self,
        broken_code: str,
        error_message: str,
        repair_context: str,
        language: str,
    ) -> Optional[str]:
        """
        Ask the LLM to repair broken code and return the fixed source.

        Uses the ``repair_full_rewrite_user`` or ``repair_diff_user`` template
        (depending on ``config.repair_diff_based``) and the ``repair_system_message``
        template (falling back to ``system_message`` if absent).

        Returns the repaired code string on success, or ``None`` if the LLM
        response could not be parsed.
        """
        if not self.repair_llm_ensemble or not self.prompt_sampler:
            logger.warning("_repair_code called but repair_llm_ensemble or prompt_sampler is None")
            return None

        # --- Choose templates ---
        user_template_name = (
            "repair_diff_user" if self.config.repair_diff_based else "repair_full_rewrite_user"
        )
        try:
            user_template = self.prompt_sampler.template_manager.get_template(user_template_name)
        except ValueError:
            logger.warning(
                f"Repair template '{user_template_name}' not found — repair skipped. "
                "Ensure the template file exists in your prompts directory."
            )
            return None

        # Prefer a dedicated repair system message; fall back to the evolution one.
        try:
            system_message = self.prompt_sampler.template_manager.get_template(
                "repair_system_message"
            )
        except ValueError:
            try:
                system_message = self.prompt_sampler.template_manager.get_template(
                    "system_message"
                )
            except ValueError:
                system_message = (
                    "You are an expert software developer. "
                    "Fix all errors in the provided code."
                )

        try:
            # Use sequential replacement instead of str.format() so that braces
            # inside broken_code / error_message / repair_context (e.g. C++ code)
            # do not raise KeyError or corrupt the template.
            user_message = user_template
            for placeholder, value in [
                ("{language}", language),
                ("{error_message}", error_message),
                ("{repair_context}", repair_context),
                ("{broken_code}", broken_code),
            ]:
                user_message = user_message.replace(placeholder, value)
        except Exception as exc:
            logger.warning(f"Repair template substitution error: {exc}")
            return None

        try:
            llm_response = await self.repair_llm_ensemble.generate_with_context(
                system_message=system_message,
                messages=[{"role": "user", "content": user_message}],
            )
        except Exception as exc:
            logger.warning(f"LLM call during repair failed: {exc}")
            return None

        # --- Parse the LLM response ---
        if self.config.repair_diff_based:
            from openevolve.utils.code_utils import apply_diff
            repaired = apply_diff(broken_code, llm_response)
        else:
            from openevolve.utils.code_utils import parse_full_rewrite
            repaired = parse_full_rewrite(llm_response, language)

        if not repaired or not repaired.strip():
            logger.warning("Repair LLM response yielded empty code after parsing")
            return None

        return repaired

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

        # Create a coroutine that runs the evaluation function in an executor
        async def run_evaluation():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.evaluate_function, program_path)

        # Run the evaluation with timeout - let exceptions bubble up for retry handling
        result = await asyncio.wait_for(run_evaluation(), timeout=self.config.timeout)

        # Return result as-is to be processed by _process_evaluation_result
        # This supports both dict and EvaluationResult returns, just like _cascade_evaluate
        return result

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
        # Import the evaluation module to get cascade functions if they exist
        try:
            # Add the evaluation file's directory to Python path so it can import local modules
            eval_dir = os.path.dirname(os.path.abspath(self.evaluation_file))
            if eval_dir not in sys.path:
                sys.path.insert(0, eval_dir)
                logger.debug(f"Added {eval_dir} to Python path for cascade evaluation")

            spec = importlib.util.spec_from_file_location("evaluation_module", self.evaluation_file)
            if spec is None or spec.loader is None:
                return await self._direct_evaluate(program_path)

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Check if cascade functions exist
            if not hasattr(module, "evaluate_stage1"):
                return await self._direct_evaluate(program_path)

            # Run first stage with timeout
            try:

                async def run_stage1():
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, module.evaluate_stage1, program_path)

                stage1_result = await asyncio.wait_for(run_stage1(), timeout=self.config.timeout)
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
            if not self._passes_threshold(
                stage1_eval_result.metrics, self.config.cascade_thresholds[0]
            ):
                return stage1_eval_result

            # Check if second stage exists
            if not hasattr(module, "evaluate_stage2"):
                return stage1_eval_result

            # Run second stage with timeout
            try:

                async def run_stage2():
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, module.evaluate_stage2, program_path)

                stage2_result = await asyncio.wait_for(run_stage2(), timeout=self.config.timeout)
                stage2_eval_result = self._process_evaluation_result(stage2_result)
            except asyncio.TimeoutError:
                logger.warning(f"Stage 2 evaluation timed out after {self.config.timeout}s")
                # Capture stage 2 failure, but keep stage 1 results
                stage1_eval_result.artifacts.update(
                    {
                        "stage2_timeout": True,
                        "failure_stage": "stage2",
                    }
                )
                stage1_eval_result.metrics["stage2_passed"] = 0.0
                stage1_eval_result.metrics["timeout"] = True
                return stage1_eval_result
            except Exception as e:
                logger.error(f"Error in stage 2 evaluation: {str(e)}")
                # Capture stage 2 failure, but keep stage 1 results
                stage1_eval_result.artifacts.update(
                    {
                        "stage2_stderr": str(e),
                        "stage2_traceback": traceback.format_exc(),
                        "failure_stage": "stage2",
                    }
                )
                stage1_eval_result.metrics["stage2_passed"] = 0.0
                return stage1_eval_result

            # Merge results from stage 1 and 2
            merged_metrics = {}
            # Convert all values to float to avoid type errors
            for name, value in stage1_eval_result.metrics.items():
                if isinstance(value, (int, float)) and name != "error":
                    merged_metrics[name] = float(value)

            for name, value in stage2_eval_result.metrics.items():
                if isinstance(value, (int, float)) and name != "error":
                    merged_metrics[name] = float(value)

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

                async def run_stage3():
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, module.evaluate_stage3, program_path)

                stage3_result = await asyncio.wait_for(run_stage3(), timeout=self.config.timeout)
                stage3_eval_result = self._process_evaluation_result(stage3_result)
            except asyncio.TimeoutError:
                logger.warning(f"Stage 3 evaluation timed out after {self.config.timeout}s")
                # Capture stage 3 failure, but keep previous results
                merged_result.artifacts.update(
                    {
                        "stage3_timeout": True,
                        "failure_stage": "stage3",
                    }
                )
                merged_result.metrics["stage3_passed"] = 0.0
                merged_result.metrics["timeout"] = True
                return merged_result
            except Exception as e:
                logger.error(f"Error in stage 3 evaluation: {str(e)}")
                # Capture stage 3 failure, but keep previous results
                merged_result.artifacts.update(
                    {
                        "stage3_stderr": str(e),
                        "stage3_traceback": traceback.format_exc(),
                        "failure_stage": "stage3",
                    }
                )
                merged_result.metrics["stage3_passed"] = 0.0
                return merged_result

            # Merge stage 3 results
            for name, value in stage3_eval_result.metrics.items():
                if isinstance(value, (int, float)) and name != "error":
                    merged_result.metrics[name] = float(value)

            merged_result.artifacts.update(stage3_eval_result.artifacts)

            return merged_result

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
            feature_dimensions = self.database.config.feature_dimensions if self.database else []
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
                avg_metrics = {}
                for i, response in enumerate(responses):
                    json_match = re.search(json_pattern, response, re.DOTALL)

                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        # Try to extract JSON directly
                        json_str = response
                        # Remove non-JSON parts
                        start_idx = json_str.find("{")
                        end_idx = json_str.rfind("}") + 1
                        if start_idx >= 0 and end_idx > start_idx:
                            json_str = json_str[start_idx:end_idx]

                    # Parse JSON
                    result = json.loads(json_str)

                    # All non-numeric values are artifacts, all numeric values are metrics
                    metrics = {}
                    for key, value in result.items():
                        if not isinstance(value, (int, float)):
                            artifacts[key] = value
                        else:
                            metrics[key] = float(value)

                    # Weight of the model in the ensemble
                    weight = self.llm_ensemble.weights[i] if self.llm_ensemble.weights else 1.0

                    # Average the metrics
                    for name, value in metrics.items():
                        if name in avg_metrics:
                            avg_metrics[name] += value * weight
                        else:
                            avg_metrics[name] = value * weight

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

        # Use combined_score if available - this is what evolution uses
        if "combined_score" in metrics:
            score = metrics.get("combined_score")
            if isinstance(score, (int, float)):
                return float(score) >= threshold

        # Fallback: average all numeric metrics except 'error'
        # This maintains backward compatibility
        valid_metrics = []
        for name, value in metrics.items():
            # Skip 'error' keys and ensure values are numeric
            if name != "error" and isinstance(value, (int, float)):
                try:
                    valid_metrics.append(float(value))
                except (TypeError, ValueError):
                    logger.warning(f"Skipping non-numeric metric: {name}={value}")
                    continue

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
