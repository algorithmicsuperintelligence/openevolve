"""Regression tests for numeric validation in the 11/12-hexagon evaluators."""

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

EXAMPLES = (
    Path(__file__).resolve().parent.parent
    / "examples"
    / "alphaevolve_math_problems"
    / "hexagon_packing"
)


class TestHexagonPackingValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluators = []
        for count in (11, 12):
            path = EXAMPLES / str(count) / "evaluator.py"
            spec = importlib.util.spec_from_file_location(f"hexagon_evaluator_{count}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls.evaluators.append(module)

    def evaluate_candidate(self, evaluator, mutation=""):
        """Execute an actual temporary candidate through the public evaluator."""
        with tempfile.TemporaryDirectory() as directory:
            name = f"hexagon_candidate_{Path(directory).name}"
            path = Path(directory) / f"{name}.py"
            source = (
                "import numpy as np\n"
                f"def hexagon_packing_{evaluator.N_HEX}():\n"
                f"    inner = np.zeros(({evaluator.N_HEX}, 3))\n"
                f"    inner[:, 0] = 3 * (np.arange({evaluator.N_HEX})"
                f" - {(evaluator.N_HEX - 1) / 2})\n"
                "    outer = np.zeros(3)\n"
                "    side = 100.0\n"
            )
            source += "".join(f"    {line}\n" for line in mutation.splitlines())
            source += "    return inner, outer, side\n"
            path.write_text(source, encoding="utf-8")
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    return evaluator.evaluate(str(path))
            finally:
                sys.modules.pop(name, None)

    def assert_invalid_candidate(self, evaluator, mutation, diagnostic):
        result = self.evaluate_candidate(evaluator, mutation)
        self.assertEqual(result["combined_score"], 0.0)
        self.assertIn(diagnostic, result["error"])
        self.assertTrue(np.isfinite(result["combined_score"]))

    def test_nan_packing_from_issue_scores_zero(self):
        for evaluator in self.evaluators:
            with self.subTest(count=evaluator.N_HEX):
                self.assert_invalid_candidate(evaluator, "inner[:] = np.nan\nside = 0.1", "finite")

    def test_nonfinite_inner_coordinates_and_angles_score_zero(self):
        for evaluator in self.evaluators:
            for column in range(3):
                for value in ("np.nan", "np.inf", "-np.inf"):
                    with self.subTest(count=evaluator.N_HEX, column=column, value=value):
                        self.assert_invalid_candidate(
                            evaluator, f"inner[0, {column}] = {value}", "finite"
                        )

    def test_nonfinite_outer_coordinates_and_angles_score_zero(self):
        for evaluator in self.evaluators:
            for column in range(3):
                for value in ("np.nan", "np.inf", "-np.inf"):
                    with self.subTest(count=evaluator.N_HEX, column=column, value=value):
                        self.assert_invalid_candidate(
                            evaluator, f"outer[{column}] = {value}", "finite"
                        )

    def test_nonfinite_outer_side_scores_zero(self):
        for evaluator in self.evaluators:
            for value in ("np.nan", "np.inf", "-np.inf"):
                with self.subTest(count=evaluator.N_HEX, value=value):
                    self.assert_invalid_candidate(evaluator, f"side = {value}", "finite")

    def test_nonpositive_outer_side_scores_zero(self):
        for evaluator in self.evaluators:
            for value in ("0.0", "-0.0", "-100.0"):
                with self.subTest(count=evaluator.N_HEX, value=value):
                    self.assert_invalid_candidate(evaluator, f"side = {value}", "positive")

    def test_valid_packing_keeps_expected_score(self):
        for evaluator in self.evaluators:
            with self.subTest(count=evaluator.N_HEX):
                result = self.evaluate_candidate(evaluator)
                self.assertNotIn("error", result)
                self.assertEqual(result["inv_outer_hex_side_length"], 0.01)
                self.assertEqual(result["combined_score"], 0.01 / evaluator.BENCHMARK)

    def test_translated_rotated_packing_keeps_expected_score(self):
        for evaluator in self.evaluators:
            with self.subTest(count=evaluator.N_HEX):
                result = self.evaluate_candidate(
                    evaluator,
                    "angle = np.deg2rad(30.0)\n"
                    "rotation = np.array([[np.cos(angle), -np.sin(angle)], "
                    "[np.sin(angle), np.cos(angle)]])\n"
                    "outer = np.array([4.0, -3.0, 30.0])\n"
                    "inner[:, :2] = inner[:, :2] @ rotation.T + outer[:2]\n"
                    "inner[:, 2] = 30.0",
                )
                self.assertNotIn("error", result)
                self.assertEqual(result["combined_score"], 0.01 / evaluator.BENCHMARK)

    def test_finite_overlapping_packing_still_rejected(self):
        for evaluator in self.evaluators:
            with self.subTest(count=evaluator.N_HEX):
                self.assert_invalid_candidate(evaluator, "inner[1] = inner[0]", "intersect")

    def test_finite_out_of_bounds_packing_still_rejected(self):
        for evaluator in self.evaluators:
            with self.subTest(count=evaluator.N_HEX):
                self.assert_invalid_candidate(evaluator, "side = 1.0", "contained")


if __name__ == "__main__":
    unittest.main()
