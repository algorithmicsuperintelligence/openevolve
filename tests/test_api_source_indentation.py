"""Convenience APIs must emit importable standalone candidate definitions."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openevolve.api import (
    EvolutionResult,
    _prepare_evaluator,
    _prepare_program,
    evolve_algorithm,
    evolve_function,
)


def top_level_double(value):
    if value < 0:
        return -(abs(value) * 2)
    return value * 2


class TopLevelSum:
    def process(self, values):
        total = 0
        for value in values:
            total += value
        return total


def benchmark_sum(instance):
    return {"score": 1.0 if instance.process([1, 2, 3]) == 6 else 0.0}


class TestGeneratedCandidateSource(unittest.TestCase):
    def assert_generated_candidate_scores(self, subject, *, algorithm=False):
        # Capture only orchestration, as in test_api.py. Candidate generation,
        # file preparation, subprocess import and generated scoring stay real.
        with patch("openevolve.api.run_evolution") as run:
            run.return_value = EvolutionResult(None, 0.0, "", {}, None)
            if algorithm:
                evolve_algorithm(subject, benchmark_sum, iterations=1)
            else:
                evolve_function(subject, [(3, 6), (-2, -4)], iterations=1)
        arguments = run.call_args.kwargs
        source = arguments["initial_program"]
        self.assertEqual(source.count("# EVOLVE-BLOCK-START"), 1)
        self.assertEqual(source.count("# EVOLVE-BLOCK-END"), 1)

        with tempfile.TemporaryDirectory() as directory:
            files = []
            program = _prepare_program(source, directory, files)
            evaluator = _prepare_evaluator(arguments["evaluator"], directory, files)
            self.assertEqual(Path(program).read_text(), source)
            process = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import importlib.util, json, sys; "
                    "spec = importlib.util.spec_from_file_location('evaluator', sys.argv[1]); "
                    "module = importlib.util.module_from_spec(spec); "
                    "spec.loader.exec_module(module); "
                    "print(json.dumps(module.evaluate(sys.argv[2])))",
                    evaluator,
                    program,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        self.assertEqual(process.returncode, 0, process.stderr)
        metrics = json.loads(process.stdout)
        self.assertEqual(metrics["combined_score"], 1.0, metrics)
        self.assertNotIn("error", metrics)
        if not algorithm:
            self.assertEqual(metrics["tests_passed"], 2)

    def test_nested_function(self):
        def double(value):
            if value < 0:
                return -(abs(value) * 2)
            return value * 2

        self.assert_generated_candidate_scores(double)

    def test_nested_function_with_markers(self):
        def double(value):
            # EVOLVE-BLOCK-START
            if value < 0:
                return -(abs(value) * 2)
            return value * 2
            # EVOLVE-BLOCK-END

        self.assert_generated_candidate_scores(double)

    def test_top_level_function(self):
        self.assert_generated_candidate_scores(top_level_double)

    def test_nested_class(self):
        class Sum:
            def process(self, values):
                total = 0
                for value in values:
                    total += value
                return total

        self.assert_generated_candidate_scores(Sum, algorithm=True)

    def test_nested_class_with_markers(self):
        class Sum:
            # EVOLVE-BLOCK-START
            def process(self, values):
                total = 0
                for value in values:
                    total += value
                return total

            # EVOLVE-BLOCK-END

        self.assert_generated_candidate_scores(Sum, algorithm=True)

    def test_top_level_class(self):
        self.assert_generated_candidate_scores(TopLevelSum, algorithm=True)
