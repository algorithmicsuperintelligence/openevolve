"""Finite fitness averages should not overflow while summing their inputs."""

import math
import sys
import unittest

from openevolve.utils.metrics_utils import get_fitness_score, safe_numeric_average


class TestNumericAverageOverflow(unittest.TestCase):
    def test_large_same_sign_metrics_have_a_finite_average(self):
        for value in (1e308, -1e308, sys.float_info.max, -sys.float_info.max):
            for count in (2, 3, 10):
                with self.subTest(value=value, count=count):
                    metrics = {str(i): value for i in range(count)}
                    self.assertEqual(safe_numeric_average(metrics), value)

    def test_finite_average_survives_intermediate_overflow_and_cancellation(self):
        maximum = sys.float_info.max
        for values in (
            [maximum, maximum, -maximum],
            [-maximum, -maximum, maximum],
            [maximum, maximum, -maximum, -maximum],
        ):
            with self.subTest(values=values):
                metrics = dict(enumerate(values))
                expected = (sum(value / maximum for value in values) / len(values)) * maximum
                self.assertEqual(safe_numeric_average(metrics), expected)

    def test_fallback_fitness_excludes_features_and_flags(self):
        metrics = {"a": 1e308, "b": 1e308, "complexity": 3.0, "valid": True}
        self.assertEqual(get_fitness_score(metrics, ["complexity"]), 1e308)
        self.assertEqual(get_fitness_score({**metrics, "combined_score": 0.25}), 0.25)

    def test_ordinary_metrics_and_invalid_values_keep_existing_behavior(self):
        self.assertEqual(safe_numeric_average({"a": 1.0, "b": 3.0}), 2.0)
        self.assertEqual(safe_numeric_average({"nan": math.nan, "a": 2, "valid": True}), 2.0)
        self.assertEqual(safe_numeric_average({"a": 0.0, "b": -0.0}), 0.0)
        self.assertEqual(safe_numeric_average({}), 0.0)
        self.assertEqual(safe_numeric_average({"too_large": 10**400, "a": 2}), 2.0)

    def test_explicit_infinity_policy_is_unchanged(self):
        self.assertEqual(safe_numeric_average({"a": math.inf, "b": 1}), math.inf)
        self.assertEqual(safe_numeric_average({"a": -math.inf, "b": 1}), -math.inf)
        self.assertTrue(math.isnan(safe_numeric_average({"a": math.inf, "b": -math.inf})))


if __name__ == "__main__":
    unittest.main()
