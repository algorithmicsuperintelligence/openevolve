"""
Booleans in a metrics dict are FLAGS, not scores.

`bool` is a subclass of `int`, so a naive `isinstance(value, (int, float))` check
silently treats True/False as 1.0/0.0. That matters in two places:

  * display  - `timeout=1.0000` instead of `timeout=True`
  * FITNESS  - a program that timed out returns {"error": 0.0, "timeout": True}
               (openevolve/evaluator.py), which averaged to 0.5 instead of 0.0,
               handing a failed program a mid-range score.

OpenEvolve's own evaluator emits `timeout: True` in several places, so this is not
a hypothetical input.
"""

import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test")

from openevolve.utils.format_utils import format_improvement_safe, format_metrics_safe
from openevolve.utils.metrics_utils import get_fitness_score, safe_numeric_average


class TestBooleanMetricsFormatting(unittest.TestCase):
    def test_bool_rendered_as_true_false(self):
        self.assertEqual(
            format_metrics_safe({"valid": True, "timeout": False, "score": 0.25}),
            "valid=True, timeout=False, score=0.2500",
        )

    def test_bool_excluded_from_improvement(self):
        """A boolean flipping False->True is not a '+1.0000' improvement."""
        self.assertEqual(
            format_improvement_safe(
                {"valid": False, "score": 0.25},
                {"valid": True, "score": 0.5},
            ),
            "score=+0.2500",
        )

    def test_numeric_formatting_unchanged(self):
        self.assertEqual(format_metrics_safe({"score": 0.5, "n": 3}), "score=0.5000, n=3.0000")


class TestBooleanMetricsExcludedFromFitness(unittest.TestCase):
    def test_timed_out_program_scores_zero(self):
        """The exact dict openevolve/evaluator.py returns on timeout."""
        metrics = {"error": 0.0, "timeout": True}
        # Before the fix both of these returned 0.5.
        self.assertEqual(safe_numeric_average(metrics), 0.0)
        self.assertEqual(get_fitness_score(metrics), 0.0)

    def test_bool_does_not_inflate_average(self):
        # Without the guard this would be (0.4 + 1.0) / 2 = 0.7
        self.assertAlmostEqual(safe_numeric_average({"score": 0.4, "valid": True}), 0.4)

    def test_all_boolean_metrics_average_to_zero(self):
        self.assertEqual(safe_numeric_average({"valid": True, "timeout": False}), 0.0)

    def test_combined_score_still_takes_precedence(self):
        self.assertAlmostEqual(get_fitness_score({"combined_score": 0.9, "timeout": True}), 0.9)

    def test_ordinary_metrics_unaffected(self):
        self.assertAlmostEqual(safe_numeric_average({"a": 0.8, "b": 0.6}), 0.7)
        self.assertAlmostEqual(get_fitness_score({"a": 0.8, "b": 0.6}), 0.7)

    def test_feature_dimensions_still_excluded(self):
        """Bool handling must not disturb the existing feature-dimension exclusion."""
        metrics = {"score": 0.8, "complexity": 100.0}
        self.assertAlmostEqual(get_fitness_score(metrics, ["complexity"]), 0.8)


if __name__ == "__main__":
    unittest.main()
