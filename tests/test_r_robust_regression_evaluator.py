"""
The R robust-regression evaluator must hand R the CSV files it writes.

The script used to interpolate the numpy arrays into read.csv(), and the CSV
files were created afterwards and never referenced.
"""

import asyncio
import importlib.util
import os
import re
import subprocess
import unittest
from unittest.mock import patch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVALUATOR_PATH = os.path.join(REPO_ROOT, "examples", "r_robust_regression", "evaluator.py")


def load_evaluator():
    spec = importlib.util.spec_from_file_location("r_robust_regression_evaluator", EVALUATOR_PATH)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    return evaluator


class TestRRobustRegressionRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = load_evaluator()

    def test_windows_paths_are_quoted_for_r(self):
        script = self.evaluator._r_runner_script(
            r"C:\Users\alice\program.r",
            r"C:\Users\alice\X.csv",
            r"C:\Users\alice\y.csv",
        )
        self.assertIn('source("C:/Users/alice/program.r")', script)
        self.assertIn('read.csv("C:/Users/alice/X.csv", header=FALSE)', script)
        self.assertIn('read.csv("C:/Users/alice/y.csv", header=FALSE)', script)

    def test_evaluate_script_reads_the_csv_files(self):
        """The script passed to Rscript must name the CSV paths, not the arrays."""
        scripts = []

        def fake_run(cmd, **kwargs):
            script_path = cmd[1]
            with open(script_path, "r", encoding="utf-8") as handle:
                script = handle.read()
            scripts.append(script)
            for csv_path in re.findall(r'read\.csv\("([^"]+\.csv)"', script):
                self.assertTrue(os.path.isfile(csv_path), csv_path)
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Rscript not run")

        program_path = r"C:\Users\alice\program.r"
        with patch.object(self.evaluator.subprocess, "run", fake_run):
            result = asyncio.run(self.evaluator.evaluate(program_path))

        self.assertEqual(len(scripts), 4)
        self.assertIn('source("C:/Users/alice/program.r")', scripts[0])
        self.assertNotIn("[[", scripts[0])
        self.assertEqual(result.metrics["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
