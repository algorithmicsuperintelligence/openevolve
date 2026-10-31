"""Offline tests for the opt-in AlphaEvolve evaluator contract (#468)."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from openevolve.api import _prepare_evaluator, run_evolution
from openevolve.compat.alphaevolve import adapt_evaluator
from openevolve.config import Config, EvaluatorConfig, LLMConfig, LLMModelConfig
from openevolve.evaluator import Evaluator


def evaluation(scores, insights=None):
    return {
        "scores": {"scores": [{"metric": key, "score": value} for key, value in scores]},
        "insights": insights,
    }


class ScriptedModel:
    """An importable, deterministic provider, including inside spawned workers."""

    def __init__(self, config):
        self.model = config.name

    async def generate_with_context(self, system_message, messages, **kwargs):
        return "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE"


def create_scripted_model(config):
    return ScriptedModel(config)


class TestAlphaEvolveAdapter(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.program = Path(self.temp.name) / "program.py"
        self.program.write_text("# 中文\nx = 1\n", encoding="utf-8")

    def adapt(self, result, **kwargs):
        return adapt_evaluator(lambda candidate: result, objective_metric="fitness", **kwargs)

    def test_candidate_contains_original_source_and_file_name(self):
        def score(candidate):
            self.assertEqual(
                candidate,
                {"content": {"files": [{"path": "program.py", "content": "# 中文\nx = 1\n"}]}},
            )
            return evaluation([("fitness", 0.8), ("diversity", 2)])

        result = adapt_evaluator(score, objective_metric="fitness")(self.program)
        self.assertEqual(result.metrics, {"fitness": 0.8, "diversity": 2.0, "combined_score": 0.8})

    def test_accepts_sdk_style_objects_without_importing_sdk(self):
        result = SimpleNamespace(
            scores=SimpleNamespace(scores=[SimpleNamespace(metric="fitness", score=-12.5)]),
            insights=SimpleNamespace(insights=[SimpleNamespace(label="Hint", text="Try again")]),
        )
        converted = self.adapt(result)(self.program)
        self.assertEqual(converted.metrics["combined_score"], -12.5)
        self.assertEqual(
            json.loads(converted.artifacts["alphaevolve_insights"]),
            [{"label": "Hint", "text": "Try again"}],
        )

    def test_dictionary_insights_are_preserved(self):
        insights = {"insights": [{"label": "Runtime Error", "text": "bad input"}]}
        result = self.adapt(evaluation([("fitness", 1.0)], insights))(self.program)
        self.assertEqual(json.loads(result.artifacts["alphaevolve_insights"]), insights["insights"])

    def test_objective_is_not_average_of_auxiliary_metrics(self):
        result = self.adapt(evaluation([("fitness", -10), ("length", 1000)]))(self.program)
        self.assertEqual(result.metrics["combined_score"], -10)

    def test_failure_sentinel_null_and_nonfinite_scores_fail_closed(self):
        for value in (-1e12, None, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                result = self.adapt(evaluation([("fitness", value), ("length", 100)]))(self.program)
                self.assertEqual(result.metrics["combined_score"], -1e12)
                self.assertEqual(result.metrics["fitness"], -1e12)
                self.assertEqual(result.metrics["length"], 100)
                self.assertEqual(result.metrics["error"], 0.0)
                self.assertIn("fitness", result.artifacts["error"])

    def test_failed_auxiliary_metric_also_marks_evaluation_failed(self):
        result = self.adapt(evaluation([("fitness", 1), ("length", None)]))(self.program)
        self.assertEqual(result.metrics["combined_score"], -1e12)

    def test_failure_mapping_is_configurable(self):
        result = self.adapt(
            evaluation([("fitness", -999)]), failure_sentinel=-999, failure_score=-1e20
        )(self.program)
        self.assertEqual(result.metrics["combined_score"], -1e20)

    def test_sentinel_can_be_disabled_for_a_valid_negative_score(self):
        result = self.adapt(evaluation([("fitness", -1e12)]), failure_sentinel=None)(self.program)
        self.assertNotIn("error", result.metrics)

    def test_missing_duplicate_or_reserved_metrics_raise(self):
        for scores in (
            [],
            [("other", 1)],
            [("fitness", 1), ("fitness", 2)],
            [("fitness", 1), ("combined_score", 2)],
            [("error", 1), ("fitness", 2)],
        ):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                self.adapt(evaluation(scores))(self.program)

    def test_malformed_names_scores_and_envelopes_raise(self):
        for result in (
            {},
            {"scores": {}},
            {"scores": {"scores": "not a list"}},
            evaluation([(None, 1)]),
            evaluation([("", 1)]),
            evaluation([("fitness", True)]),
            evaluation([("fitness", "1.0")]),
        ):
            with self.subTest(result=result), self.assertRaises((TypeError, ValueError)):
                self.adapt(result)(self.program)

    def test_combined_score_can_be_the_original_objective(self):
        adapter = adapt_evaluator(
            lambda candidate: evaluation([("combined_score", 0.5)]),
            objective_metric="combined_score",
        )
        self.assertEqual(adapter(self.program).metrics, {"combined_score": 0.5})

    def test_configuration_validation(self):
        for options in (
            {"objective_metric": ""},
            {"objective_metric": "error"},
            {"objective_metric": "fitness", "failure_score": float("nan")},
            {"objective_metric": "fitness", "failure_sentinel": float("inf")},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                adapt_evaluator(lambda candidate: {}, **options)
        with self.assertRaises(TypeError):
            adapt_evaluator(None, objective_metric="fitness")

    def test_evaluator_errors_are_not_hidden(self):
        def broken(candidate):
            raise RuntimeError("broken evaluator")

        with self.assertRaisesRegex(RuntimeError, "broken evaluator"):
            adapt_evaluator(broken, objective_metric="fitness")(self.program)

    def test_existing_callable_serialization_and_evaluator_pipeline(self):
        adapter = self.adapt(
            evaluation([("fitness", -3.0)], {"insights": [{"label": "Hint", "text": "Keep"}]})
        )
        files = []
        path = _prepare_evaluator(adapter, self.temp.name, files)
        evaluator = Evaluator(EvaluatorConfig(cascade_evaluation=False, max_retries=0), path)
        self.addCleanup(evaluator._executor.shutdown, wait=True)
        result = asyncio.run(evaluator.evaluate_program("x = 1", "candidate"))
        self.assertEqual(result, {"fitness": -3.0, "combined_score": -3.0})
        self.assertIn("alphaevolve_insights", evaluator.get_pending_artifacts("candidate"))

    def test_one_generation_through_real_controller_and_worker(self):
        def score(candidate):
            namespace = {}
            exec(candidate["content"]["files"][0]["content"], namespace)
            return evaluation([("fitness", namespace["x"])])

        config = Config()
        config.llm = LLMConfig(
            models=[LLMModelConfig(name="offline-script", init_client=create_scripted_model)]
        )
        config.database.num_islands = 1
        config.evaluator.cascade_evaluation = False
        config.evaluator.max_retries = 0
        config.evaluator.parallel_evaluations = 1
        config.evaluator.use_llm_feedback = False
        result = run_evolution(
            "# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END\n",
            adapt_evaluator(score, objective_metric="fitness"),
            config=config,
            iterations=1,
            output_dir=str(Path(self.temp.name) / "evolution"),
        )
        self.assertEqual(result.best_score, 2.0)
        self.assertEqual(result.best_program.generation, 1)
        self.assertIn("x = 2", result.best_code)


if __name__ == "__main__":
    unittest.main()
