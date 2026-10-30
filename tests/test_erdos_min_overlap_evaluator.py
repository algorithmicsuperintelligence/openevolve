"""The erdos_min_overlap evaluator scores the recomputed C5 bound, not the reported one (#562)."""

import importlib.util
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EVALUATOR = EXAMPLES_DIR / "alphaevolve_math_problems" / "erdos_min_overlap" / "evaluator.py"

N_POINTS = 100

# h = 0.5 everywhere on a width-2 interval: integral 1, and the maximum
# correlation of h with 1 - h is N_POINTS * 0.25 * (2 / N_POINTS) = 0.5.
TRUE_C5 = 0.5


def load_evaluator(name, path):
    """Load an example evaluator module directly from its file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestErdosMinOverlapEvaluator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = load_evaluator("erdos_min_overlap_evaluator", EVALUATOR)

    def write_program(self, module_name, claimed_c5):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        # The evaluator imports the program via __import__, which caches in
        # sys.modules: make sure a re-run never sees a stale module.
        self.addCleanup(sys.modules.pop, module_name, None)
        path = Path(tmpdir.name) / f"{module_name}.py"
        path.write_text(textwrap.dedent(f"""
                import numpy as np

                def run():
                    return np.full({N_POINTS}, 0.5), {claimed_c5!r}, {N_POINTS}
                """))
        return str(path)

    def test_understated_bound_is_scored_with_computed_value(self):
        """Claiming a bound just inside the consistency tolerance must not
        improve the score: it is computed from h, not taken from the claim."""
        program = self.write_program("understated_erdos", TRUE_C5 - 0.99e-4)
        result = self.evaluator.evaluate(program)
        self.assertNotIn("error", result)
        self.assertAlmostEqual(result["c5_bound"], TRUE_C5, places=12)
        self.assertAlmostEqual(
            result["combined_score"], self.evaluator.BENCHMARK / TRUE_C5, places=12
        )

    def test_honest_bound_scores_the_same(self):
        program = self.write_program("honest_erdos", TRUE_C5)
        result = self.evaluator.evaluate(program)
        self.assertNotIn("error", result)
        self.assertAlmostEqual(
            result["combined_score"], self.evaluator.BENCHMARK / TRUE_C5, places=12
        )

    def test_bound_outside_tolerance_still_rejected(self):
        program = self.write_program("outside_tolerance_erdos", TRUE_C5 - 2e-4)
        result = self.evaluator.evaluate(program)
        self.assertEqual(result["combined_score"], 0.0)
        self.assertIn("C5 mismatch", result["error"])


if __name__ == "__main__":
    unittest.main()
