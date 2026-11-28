"""
Regression tests for evaluation gate, cascade stage failure, evaluator_args forwarding,
MAP-Elites feature-dimension threshold isolation, and LLM feedback failure guards.
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "test")

from openevolve import process_parallel as process_parallel_module
from openevolve.config import Config, DatabaseConfig, EvaluatorConfig, LLMModelConfig
from openevolve.database import Program, ProgramDatabase
from openevolve.evaluator import Evaluator
from openevolve.process_parallel import ProcessParallelController, SerializableResult
from openevolve.utils.metrics_utils import get_fitness_score


class CascadeTestMixin:
    """Shared cleanup helpers for temporary evaluation modules."""

    def init_temp_env(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.original_sys_path = sys.path[:]
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), self.original_sys_path))
        self.original_eval_module = sys.modules.get("evaluation_module")
        self.addCleanup(self._restore_eval_module)

    def _restore_eval_module(self):
        if self.original_eval_module is None:
            sys.modules.pop("evaluation_module", None)
        else:
            sys.modules["evaluation_module"] = self.original_eval_module

    def write_eval_file(self, filename: str, source: str) -> Path:
        path = Path(self.directory.name) / filename
        path.write_text(source, encoding="utf-8")
        return path


class TestCascadeStageFailureIntegrity(CascadeTestMixin, unittest.IsolatedAsyncioTestCase):
    """Later-stage failures or timeouts must never retain preliminary combined_score."""

    def setUp(self):
        self.init_temp_env()

    async def test_stage2_exception_neutralizes_stage1_combined_score_without_mutating_cache(self):
        eval_path = self.write_eval_file(
            "eval_stage2_exc.py",
            """
CACHED_STAGE1 = {"combined_score": 0.95, "runs_successfully": 1.0}

def evaluate(program_path):
    return {"combined_score": 0.85}

def evaluate_stage1(program_path):
    return CACHED_STAGE1

def evaluate_stage2(program_path):
    with open(program_path, "r", encoding="utf-8") as fh:
        code = fh.read()
    if "CRASH_STAGE2" in code:
        raise RuntimeError("OOM during full evaluation")
    return {"combined_score": 0.85, "runs_successfully": 1.0}
""",
        )
        evaluator = Evaluator(
            EvaluatorConfig(
                cascade_evaluation=True,
                cascade_thresholds=[0.5],
                timeout=5,
                max_retries=0,
            ),
            str(eval_path),
        )
        self.addCleanup(evaluator._executor.shutdown, wait=False)

        crashed_metrics = await evaluator.evaluate_program("# CRASH_STAGE2\nx = 1", "crashed")
        valid_metrics = await evaluator.evaluate_program("x = 2", "valid")

        self.assertEqual(crashed_metrics["stage2_passed"], 0.0)
        self.assertIn("error", crashed_metrics)
        self.assertEqual(crashed_metrics["combined_score"], 0.0)
        self.assertEqual(get_fitness_score(crashed_metrics), 0.0)
        self.assertLess(get_fitness_score(crashed_metrics), get_fitness_score(valid_metrics))

        # Cached dict inside user module must not have been mutated in place
        self.assertEqual(
            evaluator.evaluation_module.CACHED_STAGE1,
            {"combined_score": 0.95, "runs_successfully": 1.0},
        )

        artifacts = evaluator.get_pending_artifacts("crashed")
        self.assertIsNotNone(artifacts)
        self.assertEqual(artifacts["failure_stage"], "stage2")
        self.assertIn("OOM during full evaluation", artifacts["stage2_stderr"])

    async def test_stage2_and_stage3_timeouts_neutralize_preliminary_combined_score(self):
        eval_path = self.write_eval_file(
            "eval_stage_timeout.py",
            """
import time

def evaluate(program_path):
    return {"combined_score": 0.8}

def evaluate_stage1(program_path):
    return {"combined_score": 0.95, "stage1_score": 0.95}

def evaluate_stage2(program_path):
    with open(program_path, "r", encoding="utf-8") as fh:
        code = fh.read()
    if "TIMEOUT_STAGE2" in code:
        time.sleep(2.0)
    return {"combined_score": 0.90, "stage2_score": 0.90}

def evaluate_stage3(program_path):
    with open(program_path, "r", encoding="utf-8") as fh:
        code = fh.read()
    if "TIMEOUT_STAGE3" in code:
        time.sleep(2.0)
    if "CRASH_STAGE3" in code:
        raise ValueError("stage3 exploded")
    return {"combined_score": 0.85, "stage3_score": 0.85}
""",
        )
        evaluator = Evaluator(
            EvaluatorConfig(
                cascade_evaluation=True,
                cascade_thresholds=[0.5, 0.5],
                timeout=0.15,
                max_retries=0,
            ),
            str(eval_path),
        )
        self.addCleanup(evaluator._executor.shutdown, wait=False)

        s2_timeout = await evaluator.evaluate_program("# TIMEOUT_STAGE2", "s2-to")
        self.assertTrue(s2_timeout["timeout"])
        self.assertEqual(s2_timeout["stage2_passed"], 0.0)
        self.assertIn("error", s2_timeout)
        self.assertEqual(s2_timeout["stage1_score"], 0.95)
        self.assertEqual(s2_timeout["combined_score"], 0.0)
        self.assertEqual(get_fitness_score(s2_timeout), 0.0)

        s3_timeout = await evaluator.evaluate_program("# TIMEOUT_STAGE3", "s3-to")
        self.assertTrue(s3_timeout["timeout"])
        self.assertEqual(s3_timeout["stage3_passed"], 0.0)
        self.assertIn("error", s3_timeout)
        self.assertEqual(s3_timeout["stage2_score"], 0.90)
        self.assertEqual(s3_timeout["combined_score"], 0.0)
        self.assertEqual(get_fitness_score(s3_timeout), 0.0)

        s3_crash = await evaluator.evaluate_program("# CRASH_STAGE3", "s3-crash")
        self.assertEqual(s3_crash["stage3_passed"], 0.0)
        self.assertIn("error", s3_crash)
        self.assertEqual(s3_crash["combined_score"], 0.0)
        self.assertEqual(get_fitness_score(s3_crash), 0.0)

    async def test_negative_scale_stage2_failure_cannot_outscore_valid_negative_solution(self):
        eval_path = self.write_eval_file(
            "eval_negative.py",
            """
def evaluate(program_path):
    return {"combined_score": -12.0}

def evaluate_stage1(program_path):
    return {"combined_score": -8.0}

def evaluate_stage2(program_path):
    with open(program_path, "r", encoding="utf-8") as fh:
        code = fh.read()
    if "FAIL" in code:
        raise RuntimeError("diverged")
    return {"combined_score": -12.0}
""",
        )
        evaluator = Evaluator(
            EvaluatorConfig(
                cascade_evaluation=True,
                cascade_thresholds=[-20.0],
                timeout=5,
                max_retries=0,
            ),
            str(eval_path),
        )
        self.addCleanup(evaluator._executor.shutdown, wait=False)

        valid_metrics = await evaluator.evaluate_program("x = 1", "valid-neg")
        failed_metrics = await evaluator.evaluate_program("# FAIL", "failed-neg")

        self.assertEqual(valid_metrics["combined_score"], -12.0)
        self.assertEqual(failed_metrics["combined_score"], float("-inf"))
        self.assertLess(get_fitness_score(failed_metrics), get_fitness_score(valid_metrics))

        db = ProgramDatabase(DatabaseConfig(in_memory=True))
        db.add(Program(id="valid-neg", code="x = 1", metrics=valid_metrics))
        db.add(Program(id="failed-neg", code="# FAIL", metrics=failed_metrics))
        self.assertEqual(db.get_best_program().id, "valid-neg")

    async def test_stage2_graceful_error_dict_preserves_error_and_neutralizes_stale_score(self):
        eval_path = self.write_eval_file(
            "eval_graceful_err.py",
            """
def evaluate(program_path):
    return {"combined_score": 0.8}

def evaluate_stage1(program_path):
    return {"combined_score": 0.92, "runs_successfully": 1.0}

def evaluate_stage2(program_path):
    with open(program_path, "r", encoding="utf-8") as fh:
        code = fh.read()
    if "OMIT_COMBINED" in code:
        return {"error": "All trials failed"}
    return {"combined_score": 0.0, "error": "All trials failed"}
""",
        )
        evaluator = Evaluator(
            EvaluatorConfig(cascade_evaluation=True, cascade_thresholds=[0.5], max_retries=0),
            str(eval_path),
        )
        self.addCleanup(evaluator._executor.shutdown, wait=False)

        omitted = await evaluator.evaluate_program("# OMIT_COMBINED", "omit")
        self.assertEqual(omitted["error"], "All trials failed")
        self.assertEqual(omitted["combined_score"], 0.0)
        self.assertEqual(get_fitness_score(omitted), 0.0)

        explicit_zero = await evaluator.evaluate_program("x = 1", "explicit")
        self.assertEqual(explicit_zero["error"], "All trials failed")
        self.assertEqual(explicit_zero["combined_score"], 0.0)

    async def test_empty_cascade_thresholds_returns_stage1_cleanly(self):
        eval_path = self.write_eval_file(
            "eval_empty_thresh.py",
            """
def evaluate(program_path):
    return {"combined_score": 0.5}

def evaluate_stage1(program_path):
    return {"combined_score": 0.75}
""",
        )
        evaluator = Evaluator(
            EvaluatorConfig(cascade_evaluation=True, cascade_thresholds=[], max_retries=0),
            str(eval_path),
        )
        self.addCleanup(evaluator._executor.shutdown, wait=False)

        metrics = await evaluator.evaluate_program("x = 1", "empty-thresholds")
        self.assertEqual(metrics, {"combined_score": 0.75})
        self.assertNotIn("error", metrics)


class TestCascadeModuleCachingAndKwargs(CascadeTestMixin, unittest.IsolatedAsyncioTestCase):
    """Cascade evaluation must reuse the loaded module and forward evaluator_args."""

    def setUp(self):
        self.init_temp_env()

    async def test_cascade_reuses_loaded_module_until_file_changes_on_disk(self):
        counter_path = Path(self.directory.name) / "load_count.txt"
        eval_path = self.write_eval_file(
            "eval_cached_mod.py",
            f"""
from pathlib import Path
counter_file = Path({str(counter_path)!r})
count = int(counter_file.read_text()) if counter_file.exists() else 0
counter_file.write_text(str(count + 1))

def evaluate(program_path):
    return {{"combined_score": 0.5}}

def evaluate_stage1(program_path):
    return {{"combined_score": 0.8}}

def evaluate_stage2(program_path):
    return {{"combined_score": 0.9}}
""",
        )
        evaluator = Evaluator(
            EvaluatorConfig(cascade_evaluation=True, cascade_thresholds=[0.5]),
            str(eval_path),
        )
        self.addCleanup(evaluator._executor.shutdown, wait=False)
        self.assertEqual(int(counter_path.read_text()), 1)

        for idx in range(3):
            metrics = await evaluator.evaluate_program(f"x = {idx}", f"p-{idx}")
            self.assertEqual(metrics["combined_score"], 0.9)

        # Module top-level code ran once at startup, not once per candidate
        self.assertEqual(int(counter_path.read_text()), 1)

        # Updating the file on disk invalidates the stat cache and reloads cleanly
        time.sleep(0.01)
        eval_path.write_text(
            eval_path.read_text(encoding="utf-8").replace("0.9", "0.95") + "\n",
            encoding="utf-8",
        )
        updated_metrics = await evaluator.evaluate_program("x = 99", "p-updated")
        self.assertEqual(updated_metrics["combined_score"], 0.95)
        self.assertEqual(int(counter_path.read_text()), 2)

    async def test_cascade_forwards_evaluator_args_including_mixed_stage_signatures(self):
        eval_path = self.write_eval_file(
            "eval_cascade_kwargs.py",
            """
def evaluate(program_path):
    return {"combined_score": 0.0}

def evaluate_stage1(program_path):
    # Smoke test takes only program_path
    return {"combined_score": 0.8}

def evaluate_stage2(program_path, weight=1.0):
    return {"combined_score": 0.4 * weight}

def evaluate_stage3(program_path, **kwargs):
    return {"combined_score": 0.4 * kwargs["weight"] + kwargs["bonus"]}
""",
        )
        evaluator = Evaluator(
            EvaluatorConfig(
                cascade_evaluation=True,
                cascade_thresholds=[0.5, 0.5],
                evaluator_args={"weight": 2.0, "bonus": 0.15},
            ),
            str(eval_path),
        )
        self.addCleanup(evaluator._executor.shutdown, wait=False)

        metrics = await evaluator.evaluate_program("x = 1", "cascade-kwargs")
        self.assertAlmostEqual(metrics["combined_score"], 0.95)

    def test_cascade_rejects_unaccepted_or_colliding_evaluator_args_at_startup(self):
        eval_path = self.write_eval_file(
            "eval_cascade_bad_kwargs.py",
            """
def evaluate(program_path):
    return {"combined_score": 1.0}

def evaluate_stage1(program_path):
    return {"combined_score": 1.0}

def evaluate_stage2(program_path, weight=1.0):
    return {"combined_score": weight}
""",
        )
        with self.assertRaisesRegex(ValueError, "unknown_arg"):
            Evaluator(
                EvaluatorConfig(
                    cascade_evaluation=True,
                    evaluator_args={"weight": 2.0, "unknown_arg": 5},
                ),
                str(eval_path),
            )

        with self.assertRaisesRegex(ValueError, "program_path"):
            Evaluator(
                EvaluatorConfig(
                    cascade_evaluation=True,
                    evaluator_args={"program_path": "shadow.py"},
                ),
                str(eval_path),
            )


class TestThresholdAndWorkerFeatureDimensions(CascadeTestMixin, unittest.TestCase):
    """Cascade threshold checks and worker evaluators must respect feature_dimensions."""

    def setUp(self):
        self.init_temp_env()

    def test_passes_threshold_excludes_feature_dimensions_and_rejects_non_numeric_combined(self):
        eval_path = self.write_eval_file(
            "eval_dims.py",
            "def evaluate(program_path):\n    return {'accuracy': 0.8}\n",
        )
        db = ProgramDatabase(
            DatabaseConfig(in_memory=True, feature_dimensions=["prompt_length", "execution_time"])
        )
        evaluator = Evaluator(
            EvaluatorConfig(cascade_evaluation=False),
            str(eval_path),
            database=db,
        )
        self.addCleanup(evaluator._executor.shutdown, wait=False)

        # Large feature dimension (prompt_length=1500) must not pass a zero-accuracy program
        self.assertFalse(
            evaluator._passes_threshold({"accuracy": 0.0, "prompt_length": 1500.0}, 0.5)
        )
        # Small feature dimension (execution_time=0.001) must not choke a 0.8-accuracy program
        self.assertTrue(
            evaluator._passes_threshold({"accuracy": 0.8, "execution_time": 0.001}, 0.5)
        )

        # Non-numeric / boolean combined_score must never fall through to other metrics
        bare_evaluator = Evaluator.__new__(Evaluator)
        for bad_combined in (False, True, None, "invalid", float("nan")):
            with self.subTest(combined_score=bad_combined):
                self.assertFalse(
                    bare_evaluator._passes_threshold(
                        {"combined_score": bad_combined, "runs_successfully": 1.0}, 0.0
                    )
                )

    def test_worker_lazy_init_forwards_feature_dimensions_to_evaluator(self):
        eval_path = self.write_eval_file(
            "eval_worker_dims.py",
            "def evaluate(program_path):\n    return {'reward': 0.1}\n",
        )
        saved_globals = {
            name: getattr(process_parallel_module, name, None)
            for name in (
                "_worker_config",
                "_worker_evaluation_file",
                "_worker_evaluator",
                "_worker_llm_ensemble",
                "_worker_prompt_sampler",
            )
        }

        def restore_worker_globals():
            worker_eval = getattr(process_parallel_module, "_worker_evaluator", None)
            if worker_eval is not None and hasattr(worker_eval, "_executor"):
                worker_eval._executor.shutdown(wait=False)
            for name, val in saved_globals.items():
                setattr(process_parallel_module, name, val)

        self.addCleanup(restore_worker_globals)

        cfg = Config()
        cfg.llm.models = [LLMModelConfig(name="local", api_key="test")]
        cfg.llm.evaluator_models = [LLMModelConfig(name="local", api_key="test")]
        cfg.database.feature_dimensions = ["complexity"]
        cfg.evaluator.cascade_evaluation = False

        controller = ProcessParallelController(cfg, str(eval_path), ProgramDatabase(cfg.database))
        cfg_dict = controller._serialize_config(cfg)
        process_parallel_module._worker_init(cfg_dict, str(eval_path), None)
        process_parallel_module._lazy_init_worker_components()

        worker_evaluator = process_parallel_module._worker_evaluator
        self.assertEqual(worker_evaluator._get_feature_dimensions(), ["complexity"])
        self.assertFalse(
            worker_evaluator._passes_threshold({"reward": 0.1, "complexity": 1000.0}, 0.5)
        )


class TestLLMFeedbackOnFailedEvaluations(CascadeTestMixin, unittest.IsolatedAsyncioTestCase):
    """LLM feedback must not inflate failed or timed-out evaluations."""

    def setUp(self):
        self.init_temp_env()

    async def test_llm_feedback_skipped_on_failed_evaluations_but_applied_on_valid_ones(self):
        eval_path = self.write_eval_file(
            "eval_llm_guard.py",
            """
def evaluate(program_path):
    with open(program_path, "r", encoding="utf-8") as fh:
        code = fh.read()
    if "SYNTAX_ERR" in code:
        return {"combined_score": 0.0, "error": "SyntaxError: invalid syntax"}
    if "RAISE_ERR" in code:
        raise RuntimeError("boom")
    return {"combined_score": 0.5}
""",
        )
        ensemble = Mock(weights=[1.0])
        ensemble.generate_all_with_context = AsyncMock(return_value=['{"quality": 0.9}'])
        sampler = Mock()
        sampler.build_prompt.return_value = {"system": "judge", "user": "code"}

        evaluator = Evaluator(
            EvaluatorConfig(
                cascade_evaluation=False,
                use_llm_feedback=True,
                max_retries=0,
            ),
            str(eval_path),
            llm_ensemble=ensemble,
            prompt_sampler=sampler,
        )
        self.addCleanup(evaluator._executor.shutdown, wait=False)

        syntax_metrics = await evaluator.evaluate_program("# SYNTAX_ERR", "syntax-fail")
        self.assertEqual(syntax_metrics["combined_score"], 0.0)
        self.assertNotIn("llm_quality", syntax_metrics)
        ensemble.generate_all_with_context.assert_not_called()

        raised_metrics = await evaluator.evaluate_program("# RAISE_ERR", "raise-fail")
        self.assertEqual(raised_metrics, {"error": 0.0})
        self.assertEqual(get_fitness_score(raised_metrics), 0.0)
        ensemble.generate_all_with_context.assert_not_called()

        valid_metrics = await evaluator.evaluate_program("x = 1", "valid")
        ensemble.generate_all_with_context.assert_called_once()
        self.assertAlmostEqual(valid_metrics["combined_score"], 0.5 * 0.7 + 0.9 * 0.3)


class TestProcessParallelGateIntegrity(CascadeTestMixin, unittest.IsolatedAsyncioTestCase):
    """ProcessParallelController target_score, early_stopping, and config serialization guards."""

    def setUp(self):
        self.init_temp_env()
        self.eval_path = self.write_eval_file(
            "eval_pp.py",
            "def evaluate(program_path):\n    return {'combined_score': -10.0}\n",
        )

    async def test_failed_evaluations_do_not_trigger_negative_target_score(self):
        cfg = Config()
        cfg.llm.models = [LLMModelConfig(name="local", api_key="test")]
        cfg.database.in_memory = True
        cfg.database.num_islands = 1
        cfg.evaluator.cascade_evaluation = False
        cfg.evaluator.parallel_evaluations = 1

        db = ProgramDatabase(cfg.database)
        db.add(Program(id="seed", code="x = 0", metrics={"combined_score": -20.0}), target_island=0)
        controller = ProcessParallelController(cfg, str(self.eval_path), db)

        iteration_metrics = {
            1: {"combined_score": 0.0, "error": "SyntaxError"},
            2: {"combined_score": 0.0, "stage2_passed": 0.0, "error": 0.0},
            3: {"combined_score": -4.0},
        }

        def make_future(iteration, island_id):
            fut = MagicMock()
            fut.done.return_value = True
            fut.cancel.return_value = True
            fut.result.return_value = SerializableResult(
                child_program_dict={
                    "id": f"child_{iteration}",
                    "code": f"x = {iteration}",
                    "language": "python",
                    "parent_id": "seed",
                    "generation": iteration,
                    "metrics": iteration_metrics.get(iteration, {"combined_score": -10.0}),
                    "iteration_found": iteration,
                    "metadata": {"changes": "edit", "island": island_id},
                },
                parent_id="seed",
                iteration_time=0.01,
                iteration=iteration,
                target_island=island_id,
            )
            return fut

        controller.executor = Mock()
        with patch.object(controller, "_submit_iteration", side_effect=make_future):
            best = await controller.run_evolution(
                start_iteration=1, max_iterations=6, target_score=-5.0
            )

        # Iterations 1 and 2 had combined_score=0.0 >= -5.0, but both failed and must not exit early;
        # iteration 3 genuinely reached -4.0 >= -5.0 and stops the loop before iterations 4..6 run.
        self.assertEqual(db.last_iteration, 3)
        self.assertEqual(best.id, "child_3")

    async def test_missing_custom_early_stopping_metric_excludes_feature_dimensions(self):
        cfg = Config()
        cfg.llm.models = [LLMModelConfig(name="local", api_key="test")]
        cfg.database.in_memory = True
        cfg.database.num_islands = 3
        cfg.database.feature_dimensions = ["complexity"]
        cfg.evaluator.cascade_evaluation = False
        cfg.evaluator.parallel_evaluations = 1
        cfg.early_stopping_patience = 2
        cfg.early_stopping_metric = "custom_missing_metric"
        cfg.convergence_threshold = 0.01

        db = ProgramDatabase(cfg.database)
        for idx in range(3):
            db.add(
                Program(
                    id=f"seed_{idx}",
                    code=f"x = {idx}",
                    metrics={"reward": 0.5, "complexity": 10.0 * (idx + 1)},
                ),
                target_island=idx,
            )
        controller = ProcessParallelController(cfg, str(self.eval_path), db)

        def make_future(iteration, island_id):
            fut = MagicMock()
            fut.done.return_value = True
            fut.cancel.return_value = True
            # reward stays flat at 0.5 while complexity grows rapidly
            fut.result.return_value = SerializableResult(
                child_program_dict={
                    "id": f"c_{iteration}",
                    "code": f"y = {iteration}",
                    "language": "python",
                    "parent_id": f"seed_{island_id}",
                    "generation": iteration,
                    "metrics": {"reward": 0.5, "complexity": 100.0 * iteration},
                    "iteration_found": iteration,
                    "metadata": {"changes": "edit", "island": island_id},
                },
                parent_id=f"seed_{island_id}",
                iteration_time=0.01,
                iteration=iteration,
                target_island=island_id,
            )
            return fut

        controller.executor = Mock()
        with patch.object(controller, "_submit_iteration", side_effect=make_future):
            await controller.run_evolution(start_iteration=1, max_iterations=6, target_score=None)

        self.assertTrue(controller.early_stopping_triggered)

    def test_serialize_config_restores_caller_novelty_llm(self):
        cfg = Config()
        cfg.llm.models = [LLMModelConfig(name="local", api_key="test")]
        sentinel_llm = object()
        cfg.database.novelty_llm = sentinel_llm

        controller = ProcessParallelController(
            cfg, str(self.eval_path), ProgramDatabase(cfg.database)
        )
        serialized = controller._serialize_config(cfg)

        self.assertIsNone(serialized["database"]["novelty_llm"])
        self.assertIs(cfg.database.novelty_llm, sentinel_llm)


if __name__ == "__main__":
    unittest.main()
