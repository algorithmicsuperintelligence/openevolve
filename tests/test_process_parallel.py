"""
Tests for process-based parallel controller
"""

import asyncio
import os
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
import time
from concurrent.futures import Future

# Set dummy API key for testing
os.environ["OPENAI_API_KEY"] = "test"

from openevolve.config import Config, DatabaseConfig, EvaluatorConfig, LLMConfig, PromptConfig
from openevolve.database import Program, ProgramDatabase
from openevolve.process_parallel import ProcessParallelController, SerializableResult


class TestProcessParallel(unittest.TestCase):
    """Tests for process-based parallel controller"""

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

    def test_controller_initialization(self):
        """Test that controller initializes correctly"""
        controller = ProcessParallelController(self.config, self.eval_file, self.database)

        self.assertEqual(controller.num_workers, 2)
        self.assertIsNone(controller.executor)
        self.assertIsNotNone(controller.shutdown_event)

    def test_controller_start_stop(self):
        """Test starting and stopping the controller"""
        controller = ProcessParallelController(self.config, self.eval_file, self.database)

        # Start controller
        controller.start()
        self.assertIsNotNone(controller.executor)

        # Stop controller
        controller.stop()
        self.assertIsNone(controller.executor)
        self.assertTrue(controller.shutdown_event.is_set())

    def test_database_snapshot_creation(self):
        """Test creating database snapshot for workers"""
        controller = ProcessParallelController(self.config, self.eval_file, self.database)

        snapshot = controller._create_database_snapshot()

        # Verify snapshot structure
        self.assertIn("programs", snapshot)
        self.assertIn("islands", snapshot)
        self.assertIn("current_island", snapshot)
        self.assertIn("artifacts", snapshot)

        # Verify programs are serialized
        self.assertEqual(len(snapshot["programs"]), 3)
        for pid, prog_dict in snapshot["programs"].items():
            self.assertIsInstance(prog_dict, dict)
            self.assertIn("id", prog_dict)
            self.assertIn("code", prog_dict)

    def test_run_evolution_basic(self):
        """Test basic evolution run"""

        async def run_test():
            controller = ProcessParallelController(self.config, self.eval_file, self.database)

            # Mock the executor to avoid actually spawning processes
            with patch.object(controller, "_submit_iteration") as mock_submit:
                # Create mock futures that complete immediately
                mock_future1 = MagicMock()
                mock_result1 = SerializableResult(
                    child_program_dict={
                        "id": "child_1",
                        "code": "def evolved(): return 1",
                        "language": "python",
                        "parent_id": "test_0",
                        "generation": 1,
                        "metrics": {"score": 0.7, "performance": 0.8},
                        "iteration_found": 1,
                        "metadata": {"changes": "test", "island": 0},
                    },
                    parent_id="test_0",
                    iteration_time=0.1,
                    iteration=1,
                )
                mock_future1.done.return_value = True
                mock_future1.result.return_value = mock_result1
                mock_future1.cancel.return_value = True

                mock_submit.return_value = mock_future1

                # Start controller
                controller.start()

                # Run evolution for 1 iteration
                result = await controller.run_evolution(
                    start_iteration=1, max_iterations=1, target_score=None
                )

                # Verify iteration was submitted with island_id
                mock_submit.assert_called_once_with(1, 0)

                # Verify program was added to database
                self.assertIn("child_1", self.database.programs)
                child = self.database.get("child_1")
                self.assertEqual(child.metrics["score"], 0.7)

        # Run the async test
        asyncio.run(run_test())

    def test_request_shutdown(self):
        """Test graceful shutdown request"""
        controller = ProcessParallelController(self.config, self.eval_file, self.database)

        # Request shutdown
        controller.request_shutdown()

        # Verify shutdown event is set
        self.assertTrue(controller.shutdown_event.is_set())

    def test_serializable_result(self):
        """Test SerializableResult dataclass"""
        result = SerializableResult(
            child_program_dict={"id": "test", "code": "pass"},
            parent_id="parent",
            iteration_time=1.5,
            iteration=10,
            error=None,
        )

        # Verify attributes
        self.assertEqual(result.child_program_dict["id"], "test")
        self.assertEqual(result.parent_id, "parent")
        self.assertEqual(result.iteration_time, 1.5)
        self.assertEqual(result.iteration, 10)
        self.assertIsNone(result.error)

        # Test with error
        error_result = SerializableResult(error="Test error", iteration=5)
        self.assertEqual(error_result.error, "Test error")
        self.assertIsNone(error_result.child_program_dict)


if __name__ == "__main__":
    unittest.main()
