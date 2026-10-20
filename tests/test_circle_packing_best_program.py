"""
Regression test for the circle packing best program record (issue #156).

Verifies that examples/circle_packing/best_program.py reproduces the record result
reported in issue #156 (sum of radii ~2.635977 with validity 1.0) using the
repository's own evaluator. The program is deterministic (no random initialization),
so the check is stable across runs.

Requires scipy (see examples/circle_packing/requirements.txt). A full evaluation
runs the program twice (once via evaluate(), once for the shape checks) and takes
roughly 40-60 seconds in total.
"""

import importlib.util
import os
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVALUATOR_PATH = os.path.join(REPO_ROOT, "examples", "circle_packing", "evaluator.py")
BEST_PROGRAM_PATH = os.path.join(REPO_ROOT, "examples", "circle_packing", "best_program.py")

# Lower bound for the recorded result, with headroom for floating point differences
# across scipy/BLAS versions (the exact value varies by ~1e-11).
EXPECTED_SUM_RADII = 2.6359

try:
    import scipy  # noqa: F401

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_evaluator():
    """Load the circle packing evaluator by path (it is not part of the package)"""
    spec = importlib.util.spec_from_file_location("circle_packing_evaluator", EVALUATOR_PATH)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    return evaluator


@unittest.skipUnless(HAS_SCIPY, "scipy is required to evaluate the circle packing program")
class TestCirclePackingBestProgram(unittest.TestCase):
    """Check the recorded best program still achieves the issue #156 result"""

    @classmethod
    def setUpClass(cls):
        cls.evaluator = load_evaluator()

    def test_best_program_reproduces_record_result(self):
        """The deterministic best program must reproduce the issue #156 result"""
        metrics = self.evaluator.evaluate(BEST_PROGRAM_PATH)

        self.assertEqual(metrics["validity"], 1.0)
        self.assertGreaterEqual(metrics["sum_radii"], EXPECTED_SUM_RADII)
        self.assertAlmostEqual(metrics["combined_score"], metrics["target_ratio"])

    def test_best_program_solution_shapes_and_validity(self):
        """The solution must contain 26 circles, all valid inside the unit square"""
        centers, radii, reported_sum = self.evaluator.run_with_timeout(
            BEST_PROGRAM_PATH, timeout_seconds=600
        )

        self.assertEqual(centers.shape, (26, 2))
        self.assertEqual(radii.shape, (26,))
        self.assertTrue(self.evaluator.validate_packing(centers, radii))
        self.assertGreaterEqual(float(reported_sum), EXPECTED_SUM_RADII)


if __name__ == "__main__":
    unittest.main()
