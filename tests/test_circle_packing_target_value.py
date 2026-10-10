"""
Tests for the configurable circle packing target value (issue #117).

The evaluator's TARGET_VALUE is only a scaling reference used to compute
`target_ratio`/`combined_score` (the evolutionary pressure). It does not leak a
solution, and any positive value works. These tests verify that:

1. By default the evaluator keeps the historical AlphaEvolve target (2.635).
2. Setting CIRCLE_PACKING_TARGET_VALUE in the environment changes the scaling of
   `target_ratio` but not the validity of a solution.
3. `evaluate()` and `evaluate_stage1()` share the same target.

The evaluator module is loaded by path (it is not part of the openevolve package)
with a fresh module instance per test, so environment changes are picked up at
module import time and no global state leaks between tests.

Each evaluation runs a tiny deterministic program (26 fixed circles on a grid, no
scipy), so the whole test module finishes in seconds.
"""

import importlib.util
import os
import tempfile
import textwrap
import unittest
from unittest.mock import patch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVALUATOR_PATH = os.path.join(REPO_ROOT, "examples", "circle_packing", "evaluator.py")

DETERMINISTIC_PROGRAM = textwrap.dedent("""
    import numpy as np


    def run_packing():
        centers = np.zeros((26, 2))
        radii = np.zeros(26)
        k = 0
        for x in np.linspace(0.12, 0.88, 6):
            for y in np.linspace(0.12, 0.88, 5):
                if k < 26:
                    centers[k] = [x, y]
                    radii[k] = 0.075
                    k += 1
        return centers, radii, float(radii.sum())
    """)


def load_evaluator():
    """Load a fresh evaluator module instance by path"""
    spec = importlib.util.spec_from_file_location("circle_packing_evaluator", EVALUATOR_PATH)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    return evaluator


class TestCirclePackingTargetValue(unittest.TestCase):
    """Check the evaluator target value is configurable and only affects scaling"""

    def setUp(self):
        fd, self.program_path = tempfile.mkstemp(suffix=".py")
        with os.fdopen(fd, "w") as f:
            f.write(DETERMINISTIC_PROGRAM)
        self.addCleanup(os.unlink, self.program_path)

    def test_default_target_value(self):
        """Without the environment variable, the historical 2.635 target is used"""
        evaluator = load_evaluator()

        self.assertEqual(evaluator.TARGET_VALUE, 2.635)

        metrics = evaluator.evaluate(self.program_path)
        self.assertEqual(metrics["validity"], 1.0)
        self.assertAlmostEqual(metrics["target_ratio"], metrics["sum_radii"] / 2.635)

    def test_target_value_from_environment(self):
        """CIRCLE_PACKING_TARGET_VALUE rescales target_ratio but validity is unchanged"""
        with patch.dict(os.environ, {"CIRCLE_PACKING_TARGET_VALUE": "3.0"}):
            evaluator = load_evaluator()

            self.assertEqual(evaluator.TARGET_VALUE, 3.0)

            metrics = evaluator.evaluate(self.program_path)

        self.assertEqual(metrics["validity"], 1.0)
        self.assertAlmostEqual(metrics["sum_radii"], 26 * 0.075)
        self.assertAlmostEqual(metrics["target_ratio"], metrics["sum_radii"] / 3.0)

    def test_stage1_and_evaluate_share_target(self):
        """evaluate() and evaluate_stage1() must use the same (configured) target"""
        with patch.dict(os.environ, {"CIRCLE_PACKING_TARGET_VALUE": "3.0"}):
            evaluator = load_evaluator()
            metrics = evaluator.evaluate(self.program_path)
            stage1 = evaluator.evaluate_stage1(self.program_path)

        self.assertEqual(metrics["validity"], stage1["validity"])
        self.assertAlmostEqual(metrics["sum_radii"], stage1["sum_radii"])
        self.assertAlmostEqual(metrics["target_ratio"], stage1["target_ratio"])


if __name__ == "__main__":
    unittest.main()
