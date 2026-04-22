"""
Test for tuple and list combined_score handling
"""

import unittest

from openevolve.utils.metrics_utils import get_fitness_score


class TestCombinedScoreTypes(unittest.TestCase):
    def test_combined_score_float(self):
        metrics = {"combined_score": 1.23}
        result = get_fitness_score(metrics)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 1.23)

    def test_combined_score_list(self):
        metrics = {"combined_score": [1.0, 2.0, 3.0]}
        result = get_fitness_score(metrics)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [1.0, 2.0, 3.0])

    def test_combined_score_tuple(self):
        metrics = {"combined_score": (4.0, 5.0, 6.0)}
        result = get_fitness_score(metrics)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, (4.0, 5.0, 6.0))

    def test_combined_score_invalid(self):
        metrics = {"combined_score": "not_a_number"}
        result = get_fitness_score(metrics)
        # fallback to 0.0 for invalid types when no other valid metrics are present
        self.assertIsInstance(result, float)
        self.assertEqual(result, 0.0)


if __name__ == "__main__":
    unittest.main()
