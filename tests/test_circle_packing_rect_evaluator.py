"""Regression tests for non-finite circles in the circle_packing_rect evaluator (issue #548)."""

import importlib.util
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EVALUATOR_PATH = EXAMPLES_DIR / "alphaevolve_math_problems" / "circle_packing_rect" / "evaluator.py"

# 21 equal circles on a 7x3 grid: neighbours are 0.1 apart (>= 2 * 0.04) and the
# circumscribing rectangle is 0.68 x 0.28, well inside a perimeter of 4.
VALID_GRID = "np.array([[0.1 * i, 0.1 * j, 0.04] for i in range(7) for j in range(3)])"


def load_evaluator():
    """Load the example evaluator module directly from its file path."""
    spec = importlib.util.spec_from_file_location("circle_packing_rect_evaluator", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCirclePackingRectNonFinite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = load_evaluator()

    def evaluate_circles(self, body):
        """Evaluate a candidate whose circle_packing21() runs the statements in `body`."""
        module_name = f"rect_candidate_{self._testMethodName}"
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        # The evaluator imports the program via __import__, which caches in sys.modules.
        self.addCleanup(sys.modules.pop, module_name, None)
        body = textwrap.indent(textwrap.dedent(body).strip(), "    ")
        path = Path(tmpdir.name) / f"{module_name}.py"
        path.write_text(f"import numpy as np\n\n\ndef circle_packing21():\n{body}\n")
        return self.evaluator.evaluate(str(path))

    def test_nan_centres_rejected(self):
        """The issue's repro: NaN centres make every overlap/containment check False."""
        result = self.evaluate_circles("""
            circles = np.full((21, 3), np.nan)
            circles[:, 2] = 1.0
            return circles
            """)
        self.assertEqual(result["combined_score"], 0.0)
        self.assertIn("finite", result["error"])

    def test_infinite_centres_rejected(self):
        """inf - inf is NaN, so centres at infinity slip past the same checks."""
        result = self.evaluate_circles("""
            circles = np.full((21, 3), np.inf)
            circles[:, 2] = 1.0
            return circles
            """)
        self.assertEqual(result["combined_score"], 0.0)
        self.assertIn("finite", result["error"])

    def test_infinite_radius_rejected(self):
        """np.isnan lets an infinite radius through; next to a NaN centre it scored inf."""
        result = self.evaluate_circles(f"""
            circles = {VALID_GRID}
            circles[0] = [np.nan, np.nan, np.inf]
            return circles
            """)
        self.assertEqual(result["combined_score"], 0.0)
        self.assertIn("finite", result["error"])

    def test_nan_radius_message_unchanged(self):
        """The radius check runs first, so a NaN radius keeps its specific message."""
        result = self.evaluate_circles(f"""
            circles = {VALID_GRID}
            circles[0, 2] = np.nan
            return circles
            """)
        self.assertEqual(result["combined_score"], 0.0)
        self.assertIn("nan radius", result["error"])

    def test_finite_packing_still_scores(self):
        result = self.evaluate_circles(f"return {VALID_GRID}")
        self.assertNotIn("error", result)
        self.assertAlmostEqual(result["radii_sum"], 21 * 0.04)
        self.assertGreater(result["combined_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
