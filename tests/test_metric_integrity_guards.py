"""Regression tests for boolean-as-int and non-finite metric integrity guards."""

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from openevolve import Population, run_coevolution
from openevolve.api import run_evolution
from openevolve.config import Config, DatabaseConfig, EvaluatorConfig, LLMModelConfig, PromptConfig
from openevolve.database import Program, ProgramDatabase, _safe_avg_metrics, _safe_sum_metrics
from openevolve.evaluator import Evaluator
from openevolve.evolution_trace import EvolutionTrace, extract_full_lineage_traces
from openevolve.process_parallel import ProcessParallelController
from openevolve.prompt.sampler import PromptSampler


class TestPromptSamplerMetricGuards(unittest.TestCase):
    def setUp(self):
        self.sampler = PromptSampler(PromptConfig())

    def test_format_metrics_preserves_booleans_as_flags(self):
        formatted = self.sampler._format_metrics({"score": 0.5, "timeout": True, "valid": False})
        self.assertIn("- score: 0.5000", formatted)
        self.assertIn("- timeout: True", formatted)
        self.assertIn("- valid: False", formatted)
        self.assertNotIn("timeout: 1.0000", formatted)

    def test_timeout_flip_does_not_report_improvement_or_excellent_timeout(self):
        parent_metrics = {"error": 0.0, "timeout": False}
        child_metrics = {"error": 0.0, "timeout": True}
        previous = [
            {
                "id": "child-1",
                "code": "x = 1",
                "metrics": child_metrics,
                "changes_description": "test change",
                "metadata": {"parent_metrics": parent_metrics},
            }
        ]
        history = self.sampler._format_evolution_history(
            previous_programs=previous,
            top_programs=[],
            inspirations=[],
            language="python",
        )
        self.assertNotIn("Improvement in all metrics", history)
        self.assertNotIn("timeout: 1.0000", history)
        self.assertIn("timeout: True", history)

        features = self.sampler._extract_unique_features(
            {"code": "x = 1", "metrics": {"timeout": True, "valid": False, "aux": float("nan")}}
        )
        self.assertNotIn("Excellent timeout", features)
        self.assertNotIn("Alternative valid", features)


class TestEvaluatorMetricGuards(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        eval_file = Path(self.directory.name) / "eval.py"
        eval_file.write_text('def evaluate(path):\n    return {"combined_score": True}\n')
        self.ensemble = Mock(weights=[1.0])
        self.sampler = Mock()
        self.sampler.build_prompt.return_value = {"system": "judge", "user": "code"}
        self.evaluator = Evaluator(
            EvaluatorConfig(cascade_evaluation=False, use_llm_feedback=True),
            str(eval_file),
            llm_ensemble=self.ensemble,
            prompt_sampler=self.sampler,
        )
        self.addCleanup(self.evaluator._executor.shutdown)

    def test_passes_threshold_rejects_boolean_combined_score_and_nan(self):
        self.assertFalse(
            self.evaluator._passes_threshold({"combined_score": True, "score": 0.1}, 0.5)
        )
        self.assertFalse(self.evaluator._passes_threshold({"combined_score": float("nan")}, 0.0))
        self.assertTrue(self.evaluator._passes_threshold({"score": 0.8, "aux": float("nan")}, 0.5))

    async def test_llm_feedback_skips_boolean_combined_score_without_crashing(self):
        self.ensemble.generate_all_with_context = AsyncMock(return_value=['{"quality": 0.8}'])
        metrics = await self.evaluator.evaluate_program("x = 1", "bool-combined")
        self.assertIs(metrics["combined_score"], True)
        self.assertAlmostEqual(metrics["llm_quality"], 0.08)


class TestCoevolutionMetricGuards(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.specs = {}
        for name in ("blue", "red"):
            src = self.root / f"{name}.py"
            src.write_text("0")
            cfg = Config()
            cfg.language = "python"
            cfg.llm.models = [LLMModelConfig(name="local", api_key="test")]
            cfg.llm.evaluator_models = []
            cfg.evaluator.cascade_evaluation = False
            cfg.evaluator.max_retries = 0
            self.specs[name] = Population(src, cfg)

    def test_boolean_combined_score_rejected_on_refresh(self):
        def eval_fn(path, name, opponents):
            return {"combined_score": True}

        with self.assertRaisesRegex(ValueError, "Re-evaluation failed"):
            run_coevolution(
                self.specs,
                eval_fn,
                output_dir=str(self.root / "out_bool_refresh"),
                evaluation_id="bool-refresh",
                rounds=1,
                iterations_per_phase=1,
            )

    def test_non_numeric_and_boolean_champion_raise_value_error(self):
        def eval_fn(path, name, opponents):
            return {"combined_score": 1.0}

        for bad_metrics in (
            {"combined_score": True},
            {"combined_score": None},
            {"combined_score": "bad"},
            {"combined_score": float("nan")},
        ):

            async def fake_run(engine, iterations, bad=bad_metrics):
                return Program(id="bad-champ", code="1", metrics=bad)

            with (
                self.subTest(bad_metrics=bad_metrics),
                patch("openevolve.coevolution.OpenEvolve.run", fake_run),
                self.assertRaisesRegex(ValueError, "No valid champion"),
            ):
                run_coevolution(
                    self.specs,
                    eval_fn,
                    output_dir=str(self.root / f"out_{id(bad_metrics)}"),
                    evaluation_id="bad-champ",
                    rounds=1,
                    iterations_per_phase=1,
                )


class TestProcessParallelMetricGuards(unittest.IsolatedAsyncioTestCase):
    async def test_target_score_and_early_stopping_ignore_booleans_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            eval_file = root / "eval.py"
            eval_file.write_text('def evaluate(path):\n    return {"combined_score": 0.1}\n')
            cfg = Config()
            cfg.max_iterations = 2
            cfg.early_stopping_patience = 1
            cfg.early_stopping_metric = "combined_score"
            cfg.llm.models = [LLMModelConfig(name="local", api_key="test")]
            cfg.evaluator.cascade_evaluation = False
            db = ProgramDatabase(cfg.database)
            parent = Program(id="seed", code="x = 0", metrics={"combined_score": 0.1})
            db.add(parent)

            controller = ProcessParallelController(cfg, str(eval_file), db)

            class FakeResult:
                def __init__(self, iteration, metrics):
                    self.iteration = iteration
                    self.parent_id = "seed"
                    self.child_program = Program(
                        id=f"c-{iteration}",
                        code=f"x = {iteration}",
                        parent_id="seed",
                        metrics=metrics,
                    )
                    self.error = None
                    self.prompt = None
                    self.llm_response = None
                    self.artifacts = None
                    self.iteration_time = 0.01
                    self.token_usage = None
                    self.evolution_mode = "full_rewrite"

            fake_futures = []
            for idx, m in (
                (1, {"combined_score": True, "score": 0.0}),
                (2, {"combined_score": None, "score": float("nan")}),
            ):
                fut = Mock()
                fut.done.return_value = True
                fut.result.return_value = FakeResult(idx, m)
                fake_futures.append(fut)

            submit_iter = iter(fake_futures)
            controller.executor = Mock()
            controller._submit_iteration = Mock(side_effect=lambda *a, **kw: next(submit_iter))

            await controller.run_evolution(start_iteration=1, max_iterations=2, target_score=0.9)
            self.assertEqual(controller.completed_iterations, 2)
            self.assertFalse(controller.early_stopping_triggered)


class TestDatabaseTraceAndAPIGuards(unittest.TestCase):
    def test_empty_feature_dimensions_and_nan_safe_helpers(self):
        db = ProgramDatabase(DatabaseConfig(feature_dimensions=[]))
        self.assertGreaterEqual(db.feature_bins, 1)
        self.assertAlmostEqual(_safe_sum_metrics({"a": 1.0, "b": True, "c": float("nan")}), 1.0)
        self.assertAlmostEqual(_safe_avg_metrics({"a": 2.0, "b": False, "c": float("nan")}), 2.0)

    def test_evolution_trace_and_lineage_ignore_booleans_and_nonfinite(self):
        trace = EvolutionTrace(
            iteration=1,
            timestamp=0.0,
            parent_id="p",
            child_id="c",
            parent_metrics={"score": 0.2, "timeout": False, "nan_m": float("nan")},
            child_metrics={"score": 0.7, "timeout": True, "nan_m": float("inf")},
        )
        imp = trace.calculate_improvement()
        self.assertEqual(set(imp.keys()), {"score"})
        self.assertAlmostEqual(imp["score"], 0.5)

        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            programs_dir = Path(tmpdir) / "programs"
            programs_dir.mkdir()
            (programs_dir / "p.json").write_text(
                json.dumps(
                    {
                        "id": "p",
                        "parent_id": None,
                        "metrics": {"score": 0.2, "timeout": False},
                        "generation": 0,
                        "iteration_found": 0,
                        "prompts": {},
                    }
                )
            )
            (programs_dir / "c.json").write_text(
                json.dumps(
                    {
                        "id": "c",
                        "parent_id": "p",
                        "metrics": {"score": 0.7, "timeout": True},
                        "generation": 1,
                        "iteration_found": 1,
                        "prompts": {},
                    }
                )
            )
            traces = extract_full_lineage_traces(tmpdir)
        child_trace = next(t for t in traces if t["final_program_id"] == "c")
        self.assertEqual(len(child_trace["improvement_steps"]), 1)
        self.assertEqual(set(child_trace["improvement_steps"][0]["improvement"].keys()), {"score"})

    def test_api_run_evolution_timeout_best_score_is_zero(self):
        cfg = Config()
        cfg.llm.models = [LLMModelConfig(name="local", api_key="test")]
        cfg.evaluator.cascade_evaluation = False
        timed_out = Program(id="p0", code="x=1", metrics={"error": 0.0, "timeout": True})
        with patch("openevolve.api.OpenEvolve.run", new=AsyncMock(return_value=timed_out)):
            res = run_evolution(
                initial_program="x = 1",
                evaluator=lambda p: {"error": 0.0, "timeout": True},
                config=cfg,
                iterations=1,
            )
        self.assertEqual(res.best_score, 0.0)
        self.assertFalse(math.isnan(res.best_score))


if __name__ == "__main__":
    unittest.main()
