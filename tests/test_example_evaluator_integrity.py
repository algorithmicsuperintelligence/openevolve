"""
Regression tests for example evaluator integrity.

Example evaluators must never trust values the candidate program reports about
itself: those values are produced inside the EVOLVE-BLOCK, i.e. under the LLM's
control during evolution, so a candidate could otherwise claim an arbitrary
score without solving the task.

See https://github.com/algorithmicsuperintelligence/openevolve/issues/514
"""

import importlib.util
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
MATMUL_EVALUATOR = EXAMPLES_DIR / "alphaevolve_math_problems" / "matmul" / "evaluator.py"
FUNCMIN_EVALUATOR = EXAMPLES_DIR / "function_minimization" / "evaluator.py"


def load_evaluator(name, path):
    """Load an example evaluator module directly from its file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TempProgramTestCase(unittest.TestCase):
    """Writes candidate programs to temp files and evaluates them."""

    def write_program(self, module_name, source):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        # The matmul evaluator imports the program via __import__, which caches
        # in sys.modules: make sure a re-run never sees a stale module.
        self.addCleanup(sys.modules.pop, module_name, None)
        path = Path(self._tmpdir.name) / f"{module_name}.py"
        path.write_text(textwrap.dedent(source))
        return str(path)


# An exact rank-40 decomposition of the <2, 4, 5> matmul tensor, built from
# indicator outer products: T[a, b, c] = sum_r A[a, r] * B[b, r] * C[c, r].
_RANK_40_PROGRAM = """
    import numpy as np

    N, M, P = 2, 4, 5

    def _decomposition():
        n_cols = N * M * P
        A = np.zeros((N * M, n_cols))
        B = np.zeros((M * P, n_cols))
        C = np.zeros((N * P, n_cols))
        r = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    A[i * M + j, r] = 1.0
                    B[j * P + k, r] = 1.0
                    C[k * N + i, r] = 1.0
                    r += 1
        return A, B, C

    def run():
        # Lie about the rank: the true rank is 40, claim 2.
        return _decomposition(), N, M, P, 0.0, 2
"""


class TestMatmulEvaluatorIntegrity(TempProgramTestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = load_evaluator("matmul_evaluator", MATMUL_EVALUATOR)

    def test_empty_decomposition_scores_zero(self):
        """The issue's exact repro: empty arrays + claimed rank must score 0,
        not skip verification and score 32 / claimed_rank."""
        program = self.write_program(
            "cheat_empty_matmul",
            """
            import numpy as np

            def run():
                return (np.zeros(0), np.zeros(0), np.zeros(0)), 2, 4, 5, 0.0, 2
            """,
        )
        result = self.evaluator.evaluate(program)
        self.assertEqual(result["combined_score"], 0.0)
        self.assertIn("error", result)

    def test_reported_rank_is_ignored(self):
        """The score must reflect the actual decomposition (rank 40 -> 32/40),
        not the candidate-claimed rank (which would give 32/2 = 16)."""
        program = self.write_program("cheat_rank_matmul", _RANK_40_PROGRAM)
        result = self.evaluator.evaluate(program)
        self.assertNotIn("error", result)
        self.assertEqual(result["rank"], 40)
        self.assertAlmostEqual(result["combined_score"], 32 / 40)

    def test_reported_dimensions_are_checked(self):
        """A valid decomposition for <2, 4, 5> must not score when the
        candidate reports dimensions of a different problem."""
        program = self.write_program(
            "cheat_dims_matmul",
            _RANK_40_PROGRAM.replace(
                "return _decomposition(), N, M, P, 0.0, 2",
                "return _decomposition(), 3, 4, 5, 0.0, 40",
            ),
        )
        result = self.evaluator.evaluate(program)
        self.assertEqual(result["combined_score"], 0.0)
        self.assertIn("error", result)


class TestFunctionMinimizationEvaluatorIntegrity(TempProgramTestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = load_evaluator("funcmin_evaluator", FUNCMIN_EVALUATOR)

    def _metrics(self, result):
        return result.metrics if hasattr(result, "metrics") else result

    def test_reported_value_is_recomputed(self):
        """Returning (x, y, claimed_value) must score exactly like returning
        (x, y): the claimed value is recomputed from the point."""
        honest = self.write_program(
            "honest_funcmin",
            """
            def run_search():
                return (0.5, 0.5)
            """,
        )
        cheat = self.write_program(
            "cheat_funcmin",
            """
            def run_search():
                # Claim the global-minimum value at a non-minimum point.
                return (0.5, 0.5, -1.519)
            """,
        )
        honest_metrics = self._metrics(self.evaluator.evaluate(honest))
        cheat_metrics = self._metrics(self.evaluator.evaluate(cheat))
        self.assertEqual(cheat_metrics["combined_score"], honest_metrics["combined_score"])
        # The claimed global-minimum value must not leak into the score:
        # f(0.5, 0.5) != -1.519, so value_score cannot be 1.0.
        self.assertLess(cheat_metrics["value_score"], 1.0)

    def test_stage1_recomputes_reported_value(self):
        """The cascade stage-1 evaluator must not trust the reported value
        either."""
        cheat = self.write_program(
            "cheat_funcmin_stage1",
            """
            def run_search():
                return (0.5, 0.5, -1.519)
            """,
        )
        metrics = self._metrics(self.evaluator.evaluate_stage1(cheat))
        self.assertLess(metrics["value_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
