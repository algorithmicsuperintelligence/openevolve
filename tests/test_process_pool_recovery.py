"""
Tests for process pool crash recovery
"""

import asyncio
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from concurrent.futures import BrokenExecutor, Future

# Set dummy API key for testing
os.environ["OPENAI_API_KEY"] = "test"

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.process_parallel import ProcessParallelController, SerializableResult


class TestProcessPoolRecovery(unittest.TestCase):
    """Tests for process pool crash recovery"""

    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()

        # Create test config
        self.config = Config()
        self.config.max_iterations = 10
        self.config.evaluator.parallel_evaluations = 2
        self.config.evaluator.timeout = 10
        self.config.database.num_islands = 2
        self.config.database.in_memory = True
        self.config.checkpoint_interval = 5

        # Create test evaluation file
        self.eval_content = """
def evaluate(program_path):
    return {"score": 0.5}
"""
        self.eval_file = os.path.join(self.test_dir, "evaluator.py")
        with open(self.eval_file, "w") as f:
            f.write(self.eval_content)

        # Create test database
        self.database = ProgramDatabase(self.config.database)

        # Add some test programs
        for i in range(2):
            program = Program(
                id=f"test_{i}",
                code=f"def func_{i}(): return {i}",
                language="python",
                metrics={"score": 0.5},
                iteration_found=0,
            )
            self.database.add(program)

    def tearDown(self):
        """Clean up test environment"""
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_controller_has_recovery_tracking(self):
        """Test that controller initializes with recovery tracking attributes"""
        controller = ProcessParallelController(self.config, self.eval_file, self.database)

        self.assertEqual(controller.recovery_attempts, 0)
        self.assertEqual(controller.max_recovery_attempts, 3)

    def test_recover_process_pool_recreates_executor(self):
        """Test that _recover_process_pool recreates the executor"""
        controller = ProcessParallelController(self.config, self.eval_file, self.database)

        # Start the controller to create initial executor
        controller.start()
        self.assertIsNotNone(controller.executor)
        original_executor = controller.executor

        # Simulate recovery
        with patch("time.sleep"):
            controller._recover_process_pool()

        # Verify executor was recreated
        self.assertIsNotNone(controller.executor)
        self.assertIsNot(controller.executor, original_executor)

        # Clean up
        controller.stop()

    def test_broken_executor_triggers_recovery_and_resets_on_success(self):
        """Test that BrokenExecutor triggers recovery and counter resets on success"""

        async def run_test():
            controller = ProcessParallelController(self.config, self.eval_file, self.database)

            # Track recovery calls
            recovery_called = []

            def mock_recover(failed_iterations=None):
                recovery_called.append(failed_iterations)

            controller._recover_process_pool = mock_recover

            # First call raises BrokenExecutor, subsequent calls succeed
            call_count = [0]

            def mock_submit(iteration, island_id):
                call_count[0] += 1
                mock_future = MagicMock(spec=Future)

                if call_count[0] == 1:
                    # First future raises BrokenExecutor when result() is called
                    mock_future.done.return_value = True
                    mock_future.result.side_effect = BrokenExecutor("Pool crashed")
                else:
                    # Subsequent calls succeed
                    mock_result = SerializableResult(
                        child_program_dict={
                            "id": f"child_{call_count[0]}",
                            "code": "def evolved(): return 1",
                            "language": "python",
                            "parent_id": "test_0",
                            "generation": 1,
                            "metrics": {"score": 0.7},
                            "iteration_found": iteration,
                            "metadata": {"island": island_id},
                        },
                        parent_id="test_0",
                        iteration_time=0.1,
                        iteration=iteration,
                    )
                    mock_future.done.return_value = True
                    mock_future.result.return_value = mock_result
                    mock_future.cancel.return_value = True

                return mock_future

            with patch.object(controller, "_submit_iteration", side_effect=mock_submit):
                controller.start()

                # Run evolution - should recover from crash and reset counter on success
                await controller.run_evolution(
                    start_iteration=1, max_iterations=2, target_score=None
                )

            # Verify recovery was triggered
            self.assertEqual(len(recovery_called), 1)
            # Verify counter was reset after successful iteration
            self.assertEqual(controller.recovery_attempts, 0)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
