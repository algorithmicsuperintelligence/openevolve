"""Regression tests for issue #549: first_autocorr_ineq must reject NaN/inf function values."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EVALUATOR_PATH = EXAMPLES_DIR / "alphaevolve_math_problems" / "first_autocorr_ineq" / "evaluator.py"

PROGRAM_TEMPLATE = """\
import numpy as np


def run():
    return {f_values}, {c1}, 0.0, 10
"""


def load_evaluator():
    """Load the example evaluator module directly from its file path."""
    spec = importlib.util.spec_from_file_location("first_autocorr_ineq_evaluator", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestFirstAutocorrIneqEvaluator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = load_evaluator()

    def evaluate_candidate(self, f_values, c1):
        """Evaluate a program whose run() returns the given f_values expression and claimed c1."""
        module_name = f"candidate_{self._testMethodName}"
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        # The evaluator imports the program via __import__, which caches in sys.modules.
        self.addCleanup(sys.modules.pop, module_name, None)
        path = Path(tmpdir.name) / f"{module_name}.py"
        path.write_text(PROGRAM_TEMPLATE.format(f_values=f_values, c1=c1))
        return self.evaluator.evaluate(str(path))

    def assert_rejected_as_non_finite(self, result):
        self.assertEqual(result["combined_score"], 0.0)
        self.assertIn("error", result)
        self.assertIn("finite", result["error"])

    def test_nan_values_rejected(self):
        """The issue's repro: all-NaN values with a claimed c1 must not be scored."""
        result = self.evaluate_candidate("np.full(10, np.nan)", 0.5)
        self.assert_rejected_as_non_finite(result)

    def test_positive_infinity_rejected(self):
        result = self.evaluate_candidate("np.full(10, np.inf)", 0.5)
        self.assert_rejected_as_non_finite(result)

    def test_nan_in_one_entry_rejected(self):
        result = self.evaluate_candidate("np.array([1.0] * 3 + [np.nan] + [1.0] * 6)", 2.0)
        self.assert_rejected_as_non_finite(result)

    def test_negative_infinity_still_rejected(self):
        result = self.evaluate_candidate("np.array([1.0] * 3 + [-np.inf] + [1.0] * 6)", 2.0)
        self.assertEqual(result["combined_score"], 0.0)
        self.assertIn("error", result)

    def test_finite_function_still_scores(self):
        """Ten ones have autoconvolution peak 0.5 and integral^2 0.25, so c1 is exactly 2."""
        result = self.evaluate_candidate("np.ones(10)", 2.0)
        self.assertNotIn("error", result)
        self.assertEqual(result["c1"], 2.0)
        self.assertAlmostEqual(result["combined_score"], self.evaluator.BENCHMARK / 2.0)


if __name__ == "__main__":
    unittest.main()
