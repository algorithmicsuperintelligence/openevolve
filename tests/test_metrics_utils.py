import unittest

from openevolve.utils.metrics_utils import get_fitness_score, safe_numeric_average


class TestMetricsUtils(unittest.TestCase):
    def test_safe_numeric_average_excludes_boolean_values(self):
        metrics = {
            "combined_score": 0.0,
            "timeout": True,
            "stage1_passed": False,
            "latency_ms": 2.0,
        }

        self.assertEqual(safe_numeric_average(metrics), 1.0)

    def test_get_fitness_score_excludes_boolean_values_without_combined_score(self):
        metrics = {
            "error": 0.0,
            "timeout": True,
            "ranking_passed": False,
        }

        self.assertEqual(get_fitness_score(metrics), 0.0)


if __name__ == "__main__":
    unittest.main()
