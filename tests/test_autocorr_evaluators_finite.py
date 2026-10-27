"""The second/third autocorrelation example evaluators must reject non-finite inputs and still score valid ones."""

import importlib.util
import os
import tempfile
import unittest

import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "..", "examples", "alphaevolve_math_problems")


def _load(name):
    spec = importlib.util.spec_from_file_location("ev_" + name, os.path.join(BASE, name, "evaluator.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_program(directory, name, f_values, claimed, n_points):
    path = os.path.join(directory, name + ".py")
    with open(path, "w") as handle:
        handle.write(
            "import numpy as np\n"
            "def run():\n"
            f"    return np.array({[float(x) for x in f_values]!r}), {float(claimed)!r}, 0.0, {int(n_points)}\n"
        )
    return path


def _c3(f_values):
    dx = 0.5 / len(f_values)
    return np.max(np.abs(np.convolve(f_values, f_values))) * dx / (np.sum(f_values) * dx) ** 2


class TestAutocorrEvaluatorsFinite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.second = _load("second_autocorr_ineq")
        cls.third = _load("third_autocorr_ineq")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name

    def test_second_rejects_nan_and_inf(self):
        for i, bad in enumerate((np.nan, np.inf, -np.inf)):
            f = np.ones(10)
            f[3] = bad
            result = self.second.evaluate(_write_program(self.dir, f"s_bad{i}", f, 0.9, 10))
            self.assertIn("error", result)
            self.assertEqual(result["combined_score"], 0.0)

    def test_second_accepts_valid(self):
        result = self.second.evaluate(_write_program(self.dir, "s_ok", np.ones(10), 0.9, 10))
        self.assertNotIn("error", result)
        self.assertGreater(result["combined_score"], 0.0)
        self.assertTrue(np.isfinite(result["c2"]))

    def test_third_rejects_nan_inf_and_nonfinite_claim(self):
        for i, bad in enumerate((np.nan, np.inf, -np.inf)):
            f = np.ones(10)
            f[3] = bad
            result = self.third.evaluate(_write_program(self.dir, f"t_bad{i}", f, 1.5, 10))
            self.assertIn("error", result)
            self.assertEqual(result["combined_score"], 0.0)
        result = self.third.evaluate(_write_program(self.dir, "t_nanclaim", np.ones(10), np.nan, 10))
        self.assertIn("error", result)
        self.assertEqual(result["combined_score"], 0.0)

    def test_third_accepts_valid(self):
        f = np.ones(10)
        result = self.third.evaluate(_write_program(self.dir, "t_ok", f, _c3(f), 10))
        self.assertNotIn("error", result)
        self.assertGreater(result["combined_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
