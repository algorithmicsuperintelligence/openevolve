"""
Test that artifacts produced while evaluating the INITIAL program are stored.

Regression test for the bug where the controller evaluated the initial program
(populating the evaluator's pending artifacts) but never retrieved/stored them,
unlike evolved programs. All initial-program artifacts (stderr, failure logs, LLM
feedback, etc.) were therefore silently dropped.

See: https://github.com/algorithmicsuperintelligence/openevolve/pull/462
"""

import asyncio
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

# Dummy API key so LLM ensembles initialize without a real key
os.environ.setdefault("OPENAI_API_KEY", "test")

from openevolve.config import Config
from openevolve.controller import OpenEvolve


class TestInitialProgramArtifacts(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        # Initial program with an evolve block
        self.program_file = os.path.join(self.test_dir, "initial_program.py")
        with open(self.program_file, "w") as f:
            f.write(
                "# EVOLVE-BLOCK-START\n"
                "def solve(x):\n"
                "    return x + 1\n"
                "# EVOLVE-BLOCK-END\n"
            )

        # Evaluator returns metrics AND artifacts via EvaluationResult
        self.eval_file = os.path.join(self.test_dir, "evaluator.py")
        with open(self.eval_file, "w") as f:
            f.write(
                "from openevolve.evaluation_result import EvaluationResult\n"
                "\n"
                "def evaluate(program_path):\n"
                "    return EvaluationResult(\n"
                "        metrics={'combined_score': 0.7},\n"
                "        artifacts={'stderr': 'initial-program-warning', 'note': 'hello'},\n"
                "    )\n"
            )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _make_config(self):
        config = Config()
        config.max_iterations = 1
        config.database.in_memory = True
        config.database.db_path = None
        # Keep artifacts inline in the DB for easy retrieval
        config.prompt.include_artifacts = True
        return config

    def test_initial_program_artifacts_are_stored(self):
        config = self._make_config()
        controller = OpenEvolve(
            initial_program_path=self.program_file,
            evaluation_file=self.eval_file,
            config=config,
            output_dir=os.path.join(self.test_dir, "out"),
        )

        # Neutralize the parallel evolution loop - we only want the initial-program
        # handling in run() to execute.
        with (
            patch.object(
                controller, "_run_evolution_with_checkpoints", new=AsyncMock(return_value=None)
            ),
            patch("openevolve.controller.ProcessParallelController") as mock_ppc,
        ):
            # start()/stop()/request_shutdown() are called on the instance
            instance = mock_ppc.return_value
            instance.start.return_value = None
            instance.stop.return_value = None

            asyncio.run(controller.run(iterations=1))

        # Exactly one (initial) program should have been added
        self.assertEqual(len(controller.database.programs), 1)
        initial_id = next(iter(controller.database.programs))

        artifacts = controller.database.get_artifacts(initial_id)
        self.assertIsNotNone(artifacts, "Initial program artifacts must be stored, not dropped")
        self.assertEqual(artifacts.get("stderr"), "initial-program-warning")
        self.assertEqual(artifacts.get("note"), "hello")

    def test_no_artifacts_when_evaluator_returns_none(self):
        """When the evaluator returns no artifacts, nothing is stored (and no crash)."""
        # Overwrite evaluator to return a bare metrics dict (no artifacts)
        with open(self.eval_file, "w") as f:
            f.write("def evaluate(program_path):\n    return {'combined_score': 0.7}\n")

        config = self._make_config()
        controller = OpenEvolve(
            initial_program_path=self.program_file,
            evaluation_file=self.eval_file,
            config=config,
            output_dir=os.path.join(self.test_dir, "out"),
        )

        with (
            patch.object(
                controller, "_run_evolution_with_checkpoints", new=AsyncMock(return_value=None)
            ),
            patch("openevolve.controller.ProcessParallelController") as mock_ppc,
        ):
            mock_ppc.return_value.start.return_value = None
            mock_ppc.return_value.stop.return_value = None
            asyncio.run(controller.run(iterations=1))

        self.assertEqual(len(controller.database.programs), 1)
        initial_id = next(iter(controller.database.programs))
        artifacts = controller.database.get_artifacts(initial_id)
        self.assertEqual(artifacts, {}, "No artifacts should be stored when evaluator returns none")


if __name__ == "__main__":
    unittest.main()
