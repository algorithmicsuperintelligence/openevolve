"""Regression tests for integer-domain validation and exact C6 set arithmetic."""

import contextlib
import importlib.util
import io
import math
import sys
import tempfile
import unittest
from pathlib import Path

EVALUATOR_PATH = (
    Path(__file__).resolve().parent.parent
    / "examples"
    / "alphaevolve_math_problems"
    / "sums_diffs_finite_sets"
    / "evaluator.py"
)


class TestSumsDiffsFiniteSetsValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("c6_evaluator", EVALUATOR_PATH)
        cls.evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.evaluator)

    def evaluate_candidate(self, values, bound, dtype="np.int64"):
        """Exercise the public evaluator with an actual candidate program."""
        with tempfile.TemporaryDirectory() as directory:
            name = f"c6_candidate_{Path(directory).name}"
            path = Path(directory) / f"{name}.py"
            path.write_text(
                "import numpy as np\n"
                "def run():\n"
                f"    return np.array({values}, dtype={dtype}), {bound}\n",
                encoding="utf-8",
            )
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    return self.evaluator.evaluate(str(path))
            finally:
                sys.modules.pop(name, None)

    def assert_zero_score(self, result):
        self.assertEqual(result["combined_score"], 0.0)
        self.assertIn("error", result)
        self.assertTrue(math.isfinite(result["combined_score"]))

    def assert_valid_score(self, result, bound, max_value):
        self.assertNotIn("error", result)
        self.assertAlmostEqual(result["c6_bound"], bound)
        self.assertAlmostEqual(result["combined_score"], bound / self.evaluator.BENCHMARK)
        self.assertEqual(result["set_size"], 3)
        self.assertEqual(result["max_val"], max_value)

    def test_fractional_set_cannot_claim_a_record_bound(self):
        # This non-integer set was accepted with combined_score > 2.25.
        result = self.evaluate_candidate("[0, 0.02, 0.05]", "2.617357979428168", "float")
        self.assert_zero_score(result)
        self.assertIn("integers", result["error"])

    def test_nonfinite_set_values_are_rejected(self):
        for value in ("np.nan", "np.inf", "-np.inf"):
            with self.subTest(value=value):
                self.assert_zero_score(self.evaluate_candidate(f"[0, 2, {value}]", "1.0", "float"))

    def test_integer_sets_keep_the_expected_score_across_dtypes(self):
        # U = {0, 2, 5}: six distinct sums, seven distinct differences.
        bound = 1 + math.log(7 / 6) / math.log(11)
        for dtype in ("int8", "uint8", "int32", "uint32", "int64", "uint64", "object_"):
            with self.subTest(dtype=dtype):
                result = self.evaluate_candidate("[0, 2, 5]", repr(bound), f"np.{dtype}")
                self.assert_valid_score(result, bound, 5)

    def test_integer_valued_floats_remain_supported(self):
        bound = 1 + math.log(7 / 6) / math.log(11)
        for dtype in ("float32", "float64"):
            with self.subTest(dtype=dtype):
                result = self.evaluate_candidate("[0, 2, 5]", repr(bound), f"np.{dtype}")
                self.assert_valid_score(result, bound, 5)

    def test_small_integer_arithmetic_does_not_wrap(self):
        for dtype, max_value in (("int8", 64), ("uint8", 128)):
            # Both sets have six sums and seven differences over the integers.
            with self.subTest(dtype=dtype):
                bound = 1 + math.log(7 / 6) / math.log(2 * max_value + 1)
                result = self.evaluate_candidate(f"[0, 1, {max_value}]", repr(bound), f"np.{dtype}")
                self.assert_valid_score(result, bound, max_value)

    def test_large_integer_arithmetic_does_not_wrap(self):
        for dtype, max_value in (("int64", 2**62), ("uint64", 2**63)):
            with self.subTest(dtype=dtype):
                bound = 1 + math.log(7 / 6) / math.log(2 * max_value + 1)
                result = self.evaluate_candidate(f"[0, 1, {max_value}]", repr(bound), f"np.{dtype}")
                self.assert_valid_score(result, bound, max_value)

    def test_existing_set_constraints_still_apply(self):
        for values in ("[1, 2, 5]", "[0, -2, 5]", "[]", "[[0, 2, 5]]"):
            with self.subTest(values=values):
                self.assert_zero_score(self.evaluate_candidate(values, "1.0"))

    def test_trivial_set_cannot_be_scored(self):
        self.assert_zero_score(self.evaluate_candidate("[0]", "1.0"))

    def test_incorrect_or_nonfinite_reported_bounds_still_fail(self):
        for bound in ("2.0", "np.nan", "np.inf", "-np.inf"):
            for values, dtype in (("[0, 2, 5]", "np.int64"), ("[0, 1, 128]", "np.uint8")):
                with self.subTest(bound=bound, dtype=dtype):
                    self.assert_zero_score(self.evaluate_candidate(values, bound, dtype))


if __name__ == "__main__":
    unittest.main()
