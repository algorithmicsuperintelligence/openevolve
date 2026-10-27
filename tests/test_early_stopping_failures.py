"""
Tests for early stopping interaction with failed evaluations (issue #294)

Failed or timed-out evaluations surface as metrics carrying an "error" or
"timeout" key. Those are not scores: they must not feed the early-stopping
plateau counter, otherwise a run that never succeeds once can exit through
the "successful evaluations plateaued" path.
"""

import asyncio
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Set dummy API key for testing
os.environ["OPENAI_API_KEY"] = "test"

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.process_parallel import ProcessParallelController, SerializableResult


class TestEarlyStoppingWithFailedEvaluations(unittest.TestCase):
    """Early stopping must only count evaluations that produced a score"""

    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()

        # Create test config
        self.config = Config()
        self.config.max_iterations = 10
        self.config.evaluator.parallel_evaluations = 2
        self.config.evaluator.timeout = 10
        # One island per test program so each owns its MAP-Elites cell and none is
        # displaced (MAP-Elites removes programs displaced from their cell).
        self.config.database.num_islands = 3
        self.config.database.in_memory = True
        self.config.checkpoint_interval = 5
        # Issue #294 scenario: early stopping enabled, reporter's config only set
        # patience (convergence_threshold stays at its default of 0.001).
        self.config.early_stopping_patience = 3

        # Create test evaluation file
        self.eval_content = """
def evaluate(program_path):
    return {"score": 0.5, "performance": 0.6}
"""
        self.eval_file = os.path.join(self.test_dir, "evaluator.py")
        with open(self.eval_file, "w") as f:
            f.write(self.eval_content)

        # Create test database
        self.database = ProgramDatabase(self.config.database)

        # Add some test programs, one per island so each survives as its cell owner
        for i in range(3):
            program = Program(
                id=f"test_{i}",
                code=f"def func_{i}(): return {i}",
                language="python",
                metrics={"score": 0.5 + i * 0.1, "performance": 0.4 + i * 0.1},
                iteration_found=0,
            )
            self.database.add(program, target_island=i)

    def tearDown(self):
        """Clean up test environment"""
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _run_evolution(self, metrics_for_iteration, max_iterations=5):
        """
        Run the controller against mocked per-iteration results.

        metrics_for_iteration is a callable mapping the iteration number to the
        metrics dict carried by that iteration's child program. Returns
        (controller, submit_mock) after run_evolution completes.
        """

        async def run_test():
            controller = ProcessParallelController(self.config, self.eval_file, self.database)

            def make_result(iteration, island_id):
                future = MagicMock()
                result = SerializableResult(
                    child_program_dict={
                        "id": f"child_{iteration}",
                        "code": f"def evolved_{iteration}(): return {iteration}",
                        "language": "python",
                        "parent_id": f"test_{island_id}",
                        "generation": iteration,
                        "metrics": metrics_for_iteration(iteration),
                        "iteration_found": iteration,
                        "metadata": {"changes": "test", "island": island_id},
                    },
                    parent_id=f"test_{island_id}",
                    iteration_time=0.1,
                    iteration=iteration,
                )
                future.done.return_value = True
                future.result.return_value = result
                future.cancel.return_value = True
                return future

            with patch.object(controller, "_submit_iteration", side_effect=make_result):
                mock_submit = controller._submit_iteration

                controller.start()
                await controller.run_evolution(
                    start_iteration=1, max_iterations=max_iterations, target_score=None
                )

            return controller, mock_submit

        return asyncio.run(run_test())

    def test_failed_evaluations_do_not_trigger_early_stop(self):
        """Repeated evaluator failures (codelion's {"error": 1.0, ...} convention)
        must not be counted as a plateau, and every iteration must still run"""
        controller, mock_submit = self._run_evolution(
            lambda i: {"combined_score": 0.0, "error": "syntax error"}
        )

        self.assertFalse(controller.early_stopping_triggered)
        self.assertEqual(mock_submit.call_count, 5)

    def test_timeout_shaped_metrics_do_not_trigger_early_stop(self):
        """The evaluator's timeout shape ({"error": 0.0, "timeout": True}) must
        also be excluded from the plateau counter"""
        controller, mock_submit = self._run_evolution(lambda i: {"error": 0.0, "timeout": True})

        self.assertFalse(controller.early_stopping_triggered)
        self.assertEqual(mock_submit.call_count, 5)

    def test_genuine_plateau_still_stops(self):
        """Successful evaluations that plateau must still trigger early stopping"""
        controller, _ = self._run_evolution(lambda i: {"combined_score": 0.5})

        self.assertTrue(controller.early_stopping_triggered)

    def test_negative_best_not_poisoned_by_failures(self):
        """After a real (negative) best score, failures scoring 0.0 must not look
        like improvements that reset the patience counter"""
        controller, mock_submit = self._run_evolution(
            lambda i: (
                {"combined_score": -0.5}
                if i == 1
                else {"combined_score": 0.0, "error": "runtime failure"}
            )
        )

        self.assertFalse(controller.early_stopping_triggered)
        self.assertEqual(mock_submit.call_count, 5)


if __name__ == "__main__":
    unittest.main()
