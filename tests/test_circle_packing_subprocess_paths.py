"""
Regression test for circle-packing evaluators embedding paths in generated Python.

A Windows temp path contains \\Users. Putting that path inside single quotes makes
\\U a unicode escape, so the child process dies with SyntaxError and every
evaluation is scored as a failure.
"""

import importlib.util
import os
import tempfile
import textwrap
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVALUATOR_PATHS = [
    os.path.join(REPO_ROOT, "examples", "circle_packing", "evaluator.py"),
    os.path.join(REPO_ROOT, "examples", "circle_packing_with_artifacts", "evaluator.py"),
]


def load_evaluator(path):
    """Load an example evaluator by path (examples are not part of the package)"""
    name = "circle_packing_eval_" + os.path.splitext(os.path.basename(os.path.dirname(path)))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    return evaluator


class TestCirclePackingSubprocessPaths(unittest.TestCase):
    """Generated runner source must be valid Python for Windows paths"""

    @classmethod
    def setUpClass(cls):
        cls.evaluators = [load_evaluator(path) for path in EVALUATOR_PATHS]

    def test_windows_user_temp_path_compiles(self):
        """C:\\Users\\... must not be parsed as a unicode escape"""
        program_path = r"C:\Users\alice\AppData\Local\Temp\prog.py"
        results_path = program_path + ".results"
        for evaluator in self.evaluators:
            script = evaluator._subprocess_script(program_path, results_path)
            compile(script, "<circle-packing-runner>", "exec")
            self.assertIn(repr(program_path), script)
            self.assertIn(repr(results_path), script)

    def test_apostrophe_in_path_compiles(self):
        """A quote in a directory name must stay inside the string literal"""
        program_path = r"C:\Users\o'brien\prog.py"
        results_path = program_path + ".results"
        for evaluator in self.evaluators:
            script = evaluator._subprocess_script(program_path, results_path)
            compile(script, "<circle-packing-runner>", "exec")

    def test_run_with_timeout_executes_program_from_user_temp(self):
        """The documented subprocess path must return the program's packing"""
        program = textwrap.dedent("""
            import numpy as np

            def run_packing():
                centers = np.array([[0.5, 0.5]])
                radii = np.array([0.1])
                return centers, radii, float(radii.sum())
            """)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(program)
            program_path = handle.name

        try:
            for evaluator in self.evaluators:
                centers, radii, reported_sum = evaluator.run_with_timeout(
                    program_path, timeout_seconds=30
                )
                self.assertEqual(centers.shape, (1, 2))
                self.assertEqual(float(radii[0]), 0.1)
                self.assertAlmostEqual(float(reported_sum), 0.1)
        finally:
            os.unlink(program_path)


if __name__ == "__main__":
    unittest.main()
