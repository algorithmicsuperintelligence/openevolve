"""
Tests for checkpoint resume functionality and initial program deduplication
"""

import asyncio
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import json
import time

# Set dummy API key for testing to prevent OpenAI SDK import failures
os.environ["OPENAI_API_KEY"] = "test"

from openevolve.config import Config
from openevolve.controller import OpenEvolve
from openevolve.database import Program, ProgramDatabase


class MockEvaluator:
    """Mock evaluator for testing"""

    def __init__(self):
        self.call_count = 0

    async def evaluate_program(self, code, program_id):
        """Mock evaluation that returns predictable metrics"""
        self.call_count += 1
        # Return slightly different metrics each time to simulate real evaluation
        return {
            "score": 0.5 + (self.call_count * 0.1) % 0.5,
            "combined_score": 0.6 + (self.call_count * 0.05) % 0.4,
        }

    def get_pending_artifacts(self, program_id):
        """Mock artifact retrieval - mirrors the real Evaluator interface.

        The controller calls this for the initial program; the real Evaluator
        always provides it, so the mock must too.
        """
        return None


class TestCheckpointResume(unittest.TestCase):
    """Tests for checkpoint resume functionality"""

    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()

        # Create a simple test program
        self.test_program_content = """# EVOLVE-BLOCK-START
def test_function():
    return "test"
# EVOLVE-BLOCK-END

def main():
    return test_function()
"""

        self.test_program_path = os.path.join(self.test_dir, "test_program.py")
        with open(self.test_program_path, "w") as f:
            f.write(self.test_program_content)

        # Create a simple evaluator file
        self.evaluator_content = """
def evaluate(program_path):
    return {"score": 0.5, "combined_score": 0.6}
"""

        self.evaluator_path = os.path.join(self.test_dir, "evaluator.py")
        with open(self.evaluator_path, "w") as f:
            f.write(self.evaluator_content)

        # Create test config
        self.config = Config()
        self.config.max_iterations = 2  # Keep tests fast
        self.config.checkpoint_interval = 1
        self.config.database.in_memory = True

    def tearDown(self):
        """Clean up test environment"""
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_fresh_start_adds_initial_program(self):
        """Test that initial program is added when starting fresh"""

        async def run_test():
            with patch("openevolve.controller.Evaluator") as mock_evaluator_class:
                mock_evaluator = MockEvaluator()
                mock_evaluator_class.return_value = mock_evaluator

                controller = OpenEvolve(
                    initial_program_path=self.test_program_path,
                    evaluation_file=self.evaluator_path,
                    config=self.config,
                    output_dir=self.test_dir,
                )

                # Verify database is empty initially
                self.assertEqual(len(controller.database.programs), 0)
                self.assertEqual(controller.database.last_iteration, 0)

                # Mock the parallel controller to avoid API calls
                with patch(
                    "openevolve.controller.ProcessParallelController"
                ) as mock_controller_class:
                    mock_controller = Mock()
                    mock_controller.run_evolution = AsyncMock(return_value=None)
                    mock_controller.start = Mock(return_value=None)
                    mock_controller.stop = Mock(return_value=None)
                    mock_controller.shutdown_event = Mock()
                    mock_controller.shutdown_event.is_set.return_value = False
                    mock_controller_class.return_value = mock_controller

                    # Run for 0 iterations (just initialization)
                    result = await controller.run(iterations=0)

                # Verify initial program was added
                self.assertEqual(len(controller.database.programs), 1)

                # Verify the initial program has the correct content
                programs = list(controller.database.programs.values())
                initial_program = programs[0]
                self.assertEqual(initial_program.code, self.test_program_content)
                self.assertEqual(initial_program.iteration_found, 0)

                # Verify evaluator was called exactly once for initial program
                self.assertEqual(mock_evaluator.call_count, 1)

        # Run the async test
        asyncio.run(run_test())

    def test_duplicate_content_prevention(self):
        """Test that programs with identical content are not added multiple times"""

        async def run_test():
            with patch("openevolve.controller.Evaluator") as mock_evaluator_class:
                mock_evaluator = MockEvaluator()
                mock_evaluator_class.return_value = mock_evaluator

                controller = OpenEvolve(
                    initial_program_path=self.test_program_path,
                    evaluation_file=self.evaluator_path,
                    config=self.config,
                    output_dir=self.test_dir,
                )

                # Add a program with different ID but same content as initial program
                existing_program = Program(
                    id="different_id",
                    code=self.test_program_content,  # Same content as initial program
                    language="python",
                    metrics={"score": 0.7, "combined_score": 0.8},
                    iteration_found=0,
                )

                controller.database.add(existing_program)

                # Mock the parallel controller to avoid API calls
                with patch(
                    "openevolve.controller.ProcessParallelController"
                ) as mock_controller_class:
                    mock_controller = Mock()
                    mock_controller.run_evolution = AsyncMock(return_value=None)
                    mock_controller.start = Mock(return_value=None)
                    mock_controller.stop = Mock(return_value=None)
                    mock_controller.shutdown_event = Mock()
                    mock_controller.shutdown_event.is_set.return_value = False
                    mock_controller_class.return_value = mock_controller

                    # Run for 0 iterations (just initialization)
                    result = await controller.run(iterations=0)

                # Verify no additional program was added (still only 1 program)
                self.assertEqual(len(controller.database.programs), 1)

                # Verify the existing program is still there
                self.assertIn("different_id", controller.database.programs)

                # Verify evaluator was not called for initial program
                self.assertEqual(mock_evaluator.call_count, 0)

    def test_checkpoint_resume_skips_initial_program(self):
        """Test that initial program is not re-added when resuming from checkpoint"""

        async def run_test():
            with patch("openevolve.controller.Evaluator") as mock_evaluator_class:
                mock_evaluator = MockEvaluator()
                mock_evaluator_class.return_value = mock_evaluator

                controller = OpenEvolve(
                    initial_program_path=self.test_program_path,
                    evaluation_file=self.evaluator_path,
                    config=self.config,
                    output_dir=self.test_dir,
                )

                # Simulate existing database state (as if loaded from checkpoint)
                existing_program = Program(
                    id="existing_program_id",
                    code=self.test_program_content,  # Same content as initial program
                    language="python",
                    metrics={"score": 0.7, "combined_score": 0.8},
                    iteration_found=5,
                )

                controller.database.add(existing_program)
                controller.database.last_iteration = 10  # Simulate resuming from iteration 10

                # Verify database has the existing program
                self.assertEqual(len(controller.database.programs), 1)
                self.assertEqual(controller.database.last_iteration, 10)

                # Mock the parallel controller to avoid API calls
                with patch(
                    "openevolve.controller.ProcessParallelController"
                ) as mock_controller_class:
                    mock_controller = Mock()
                    mock_controller.run_evolution = AsyncMock(return_value=None)
                    mock_controller.start = Mock(return_value=None)
                    mock_controller.stop = Mock(return_value=None)
                    mock_controller.shutdown_event = Mock()
                    mock_controller.shutdown_event.is_set.return_value = False
                    mock_controller_class.return_value = mock_controller

                    # Run for 0 iterations (just initialization)
                    result = await controller.run(iterations=0)

                # Verify no additional program was added (still only 1 program)
                self.assertEqual(len(controller.database.programs), 1)

                # Verify the existing program is still there with original ID
                self.assertIn("existing_program_id", controller.database.programs)

                # Verify evaluator was not called for initial program (count should be 0)
                self.assertEqual(mock_evaluator.call_count, 0)

        # Run the async test
        asyncio.run(run_test())

    def test_non_empty_database_at_iteration_zero(self):
        """Test that initial program is not added when database already has programs at iteration 0"""

        async def run_test():
            with patch("openevolve.controller.Evaluator") as mock_evaluator_class:
                mock_evaluator = MockEvaluator()
                mock_evaluator_class.return_value = mock_evaluator

                controller = OpenEvolve(
                    initial_program_path=self.test_program_path,
                    evaluation_file=self.evaluator_path,
                    config=self.config,
                    output_dir=self.test_dir,
                )

                # Add a program with different content to simulate pre-populated database
                different_program = Program(
                    id="different_id",
                    code="def different_function(): pass",  # Different content
                    language="python",
                    metrics={"score": 0.6, "combined_score": 0.7},
                    iteration_found=0,
                )

                controller.database.add(different_program)
                # Keep last_iteration at 0 to simulate fresh start with pre-populated DB

                # Verify database has the different program
                self.assertEqual(len(controller.database.programs), 1)
                self.assertEqual(controller.database.last_iteration, 0)

                # Mock the parallel controller to avoid API calls
                with patch(
                    "openevolve.controller.ProcessParallelController"
                ) as mock_controller_class:
                    mock_controller = Mock()
                    mock_controller.run_evolution = AsyncMock(return_value=None)
                    mock_controller.start = Mock(return_value=None)
                    mock_controller.stop = Mock(return_value=None)
                    mock_controller.shutdown_event = Mock()
                    mock_controller.shutdown_event.is_set.return_value = False
                    mock_controller_class.return_value = mock_controller

                    # Run for 0 iterations (just initialization)
                    result = await controller.run(iterations=0)

                # Verify no additional program was added (still only 1 program)
                self.assertEqual(len(controller.database.programs), 1)

                # Verify the existing program is still there
                self.assertIn("different_id", controller.database.programs)

                # Verify evaluator was not called for initial program
                self.assertEqual(mock_evaluator.call_count, 0)

        # Run the async test
        asyncio.run(run_test())

    def test_multiple_run_calls_no_pollution(self):
        """Test that calling run() multiple times doesn't pollute the database"""

        async def run_test():
            with patch("openevolve.controller.Evaluator") as mock_evaluator_class:
                mock_evaluator = MockEvaluator()
                mock_evaluator_class.return_value = mock_evaluator

                controller = OpenEvolve(
                    initial_program_path=self.test_program_path,
                    evaluation_file=self.evaluator_path,
                    config=self.config,
                    output_dir=self.test_dir,
                )

                # Mock the parallel controller to avoid API calls
                with patch(
                    "openevolve.controller.ProcessParallelController"
                ) as mock_parallel_class:
                    mock_parallel = MagicMock()
                    mock_parallel.run_evolution = AsyncMock(return_value=None)
                    mock_parallel.start = MagicMock()
                    mock_parallel.stop = MagicMock()
                    mock_parallel.shutdown_event.is_set.return_value = False
                    mock_parallel_class.return_value = mock_parallel

                    # Run first time
                    result1 = await controller.run(iterations=0)
                    initial_count = len(controller.database.programs)
                    evaluator_calls_after_first = mock_evaluator.call_count

                    # Run second time (simulating resume or restart)
                    result2 = await controller.run(iterations=0)

                    # Run third time
                    result3 = await controller.run(iterations=0)

                # Verify database size didn't grow
                self.assertEqual(len(controller.database.programs), initial_count)

                # Verify evaluator was only called once (for the initial program in first run)
                self.assertEqual(mock_evaluator.call_count, evaluator_calls_after_first)

        # Run the async test
        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
