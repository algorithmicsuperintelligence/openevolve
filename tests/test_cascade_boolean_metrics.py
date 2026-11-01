"""Boolean flags must remain metadata throughout successful cascade evaluations."""

import sys
import tempfile
import unittest
from pathlib import Path

from openevolve.config import EvaluatorConfig
from openevolve.evaluator import Evaluator
from openevolve.utils.metrics_utils import get_fitness_score


class TestCascadeBooleanMetrics(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.original_path = sys.path[:]
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), self.original_path))
        self.original_module = sys.modules.get("evaluation_module")
        self.addCleanup(self.restore_module)

    def restore_module(self):
        if self.original_module is None:
            sys.modules.pop("evaluation_module", None)
        else:
            sys.modules["evaluation_module"] = self.original_module

    def make_evaluator(self, stages, thresholds=(0.5, 0.5)):
        path = Path(self.directory.name) / "evaluator.py"
        source = "def evaluate(path):\n    return {}\n"
        for i, metrics in enumerate(stages, 1):
            source += f"def evaluate_stage{i}(path):\n    return {metrics!r}\n"
        path.write_text(source, encoding="utf-8")
        evaluator = Evaluator(
            EvaluatorConfig(cascade_evaluation=True, cascade_thresholds=list(thresholds)),
            str(path),
        )
        self.addCleanup(evaluator._executor.shutdown, wait=True)
        return evaluator

    async def test_true_flag_does_not_advance_a_low_scoring_first_stage(self):
        first = {"score": 0.1, "valid": True}
        evaluator = self.make_evaluator([first, {"score": 0.9, "later_stage": 1.0}])
        result = await evaluator._cascade_evaluate("unused.py")
        self.assertEqual(result.metrics, first)

    async def test_false_flag_does_not_block_a_high_scoring_first_stage(self):
        evaluator = self.make_evaluator(
            [{"score": 0.8, "flag": False}, {"score": 0.9, "later_stage": 1.0}]
        )
        result = await evaluator._cascade_evaluate("unused.py")
        self.assertEqual(result.metrics["later_stage"], 1.0)

    async def test_true_flags_do_not_advance_a_low_scoring_second_stage(self):
        evaluator = self.make_evaluator(
            [{"score": 0.8, "valid": True}, {"score": 0.1}, {"score": 0.9, "third": 1.0}]
        )
        result = await evaluator._cascade_evaluate("unused.py")
        self.assertEqual(result.metrics["score"], 0.1)
        self.assertNotIn("third", result.metrics)

    async def test_merges_preserve_flags_and_fallback_fitness(self):
        for count in (2, 3):
            with self.subTest(stages=count):
                stages = [
                    {"score": 0.2, f"valid_{i}": True, f"optional_{i}": False} for i in range(count)
                ]
                evaluator = self.make_evaluator(stages, thresholds=(0.0, 0.0))
                result = await evaluator._cascade_evaluate("unused.py")
                self.assertAlmostEqual(get_fitness_score(result.metrics), 0.2)
                for i in range(count):
                    self.assertIs(result.metrics[f"valid_{i}"], True)
                    self.assertIs(result.metrics[f"optional_{i}"], False)

    def test_flag_only_metrics_do_not_pass_even_a_zero_threshold(self):
        evaluator = self.make_evaluator([{}, {}])
        for metrics in ({"valid": True}, {"valid": False}, {"error": 0.0, "valid": True}):
            with self.subTest(metrics=metrics):
                self.assertFalse(evaluator._passes_threshold(metrics, 0.0))

    def test_numeric_combined_score_still_takes_precedence(self):
        evaluator = self.make_evaluator([{}, {}])
        for combined, other, expected in ((0.0, 1.0, False), (0.5, 0.0, True)):
            with self.subTest(combined=combined):
                self.assertEqual(
                    evaluator._passes_threshold(
                        {"combined_score": combined, "score": other, "flag": True}, 0.5
                    ),
                    expected,
                )

    async def test_numeric_scores_and_error_filtering_are_unchanged(self):
        evaluator = self.make_evaluator(
            [{"score": 1, "error": 0.0}, {"score": 0.8}, {"score": 0.9}]
        )
        result = await evaluator._cascade_evaluate("unused.py")
        self.assertEqual(result.metrics, {"score": 0.9})
        self.assertIsInstance(result.metrics["score"], float)


if __name__ == "__main__":
    unittest.main()
