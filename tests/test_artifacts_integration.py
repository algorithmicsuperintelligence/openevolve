"""
Integration tests for artifacts functionality
"""

import asyncio
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from openevolve.config import Config, DatabaseConfig, EvaluatorConfig, PromptConfig
from openevolve.database import Program
from openevolve.database_memory import InMemoryProgramDatabase as ProgramDatabase
from openevolve.evaluation_result import EvaluationResult
from openevolve.evaluator import Evaluator
from openevolve.prompt.sampler import PromptSampler


class TestArtifactsIntegration(unittest.TestCase):
    """Test full integration of artifacts feature"""

    def setUp(self):
        # Set up event loop for async operations in tests
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        # Create temporary directory for database
        self.temp_dir = tempfile.mkdtemp()

        # Create evaluation file that can return EvaluationResult
        self.eval_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        self.eval_file.write(
            """
import traceback
from openevolve.evaluation_result import EvaluationResult

def evaluate(program_path):
    try:
        # Try to compile the program
        with open(program_path, 'r') as f:
            code = f.read()
        
        compile(code, program_path, 'exec')
        
        # If compilation succeeds, return good metrics
        return EvaluationResult(
            metrics={"compile_ok": 1.0, "score": 0.8},
            artifacts={"stdout": "Compilation successful"}
        )
    except Exception as e:
        # If compilation fails, capture the error
        return EvaluationResult(
            metrics={"compile_ok": 0.0, "score": 0.0},
            artifacts={
                "stderr": str(e),
                "traceback": traceback.format_exc(),
                "failure_stage": "compilation"
            }
        )

def evaluate_stage1(program_path):
    # Basic compilation check
    try:
        with open(program_path, 'r') as f:
            code = f.read()
        
        compile(code, program_path, 'exec')
        return {"stage1_passed": 1.0, "compile_ok": 1.0}
    except Exception as e:
        return EvaluationResult(
            metrics={"stage1_passed": 0.0, "compile_ok": 0.0},
            artifacts={
                "stderr": str(e),
                "failure_stage": "stage1_compilation"
            }
        )
"""
        )
        self.eval_file.close()

        # Set up config
        self.config = Config()
        self.config.database.db_path = self.temp_dir
        self.config.evaluator.cascade_evaluation = True
        self.config.prompt.include_artifacts = True

        # Initialize components
        self.database = ProgramDatabase(self.config.database)
        self.evaluator = Evaluator(self.config.evaluator, self.eval_file.name)
        self.prompt_sampler = PromptSampler(self.config.prompt)

    def tearDown(self):
        os.unlink(self.eval_file.name)
        # Clean up event loop if we created one
        if hasattr(self, "loop") and self.loop and not self.loop.is_closed():
            # Cancel any pending tasks
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()
            # Run the loop briefly to let cancellations process
            if pending:
                self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

    def test_compile_failure_artifact_capture(self):
        """Test that compilation failures are captured as artifacts"""

        async def run_test():
            # Program with syntax error
            bad_code = "print('hello'\n  # Missing closing parenthesis"
            program_id = "bad_program_1"

            # Evaluate the program
            metrics = await self.evaluator.evaluate_program(bad_code, program_id)

            # Should get failure metrics
            self.assertEqual(metrics.get("compile_ok"), 0.0)

            # Should have pending artifacts
            artifacts = self.evaluator.get_pending_artifacts(program_id)
            self.assertIsNotNone(artifacts)
            self.assertIn("stderr", artifacts)
            # Note: stage1 evaluation doesn't include traceback, only stderr and failure_stage
            self.assertIn("failure_stage", artifacts)

            return artifacts

        artifacts = asyncio.run(run_test())

        # Verify artifact content - should have captured the compilation error
        self.assertIn("stderr", artifacts)
        self.assertTrue(len(artifacts["stderr"]) > 0, "stderr should not be empty")
        self.assertIn("failure_stage", artifacts)
        self.assertEqual(artifacts["failure_stage"], "stage1_compilation")

    def test_end_to_end_artifact_flow(self):
        """Test full flow: eval failure -> artifact -> prompt -> next gen"""

        async def run_test():
            # 1. Create a program with compilation error
            bad_code = "def broken_function(\n    return 'incomplete'"
            program_id = "flow_test_1"

            # 2. Evaluate and get artifacts
            metrics = await self.evaluator.evaluate_program(bad_code, program_id)
            artifacts = self.evaluator.get_pending_artifacts(program_id)

            # 3. Create program and store in database
            program = Program(id=program_id, code=bad_code, language="python", metrics=metrics)
            self.database.add(program)

            # 4. Store artifacts
            if artifacts:
                self.database.store_artifacts(program_id, artifacts)

            # 5. Retrieve artifacts and build prompt
            stored_artifacts = self.database.get_artifacts(program_id)

            prompt = self.prompt_sampler.build_prompt(
                current_program=bad_code,
                parent_program=bad_code,
                program_metrics=metrics,
                previous_programs=[],
                top_programs=[],
                program_artifacts=stored_artifacts,
            )

            return prompt, stored_artifacts

        prompt, artifacts = asyncio.run(run_test())

        # Verify artifacts appear in prompt
        self.assertIn("stderr", prompt["user"].lower())
        self.assertIn("Last Execution Output", prompt["user"])

        # Verify artifacts were stored and retrieved correctly
        self.assertIn("stderr", artifacts)
        self.assertTrue(len(artifacts["stderr"]) > 0, "stderr should not be empty")

    def test_cascade_evaluation_with_artifacts(self):
        """Test cascade evaluation captures artifacts at each stage"""

        async def run_test():
            # Program that will fail at stage 1
            invalid_code = "invalid syntax here"
            program_id = "cascade_test_1"

            # Run cascade evaluation
            result = await self.evaluator._cascade_evaluate(f"/tmp/test_program.py")

            # Should be an EvaluationResult with artifacts
            if isinstance(result, EvaluationResult):
                return result
            else:
                # If it returns a dict, wrap it
                return EvaluationResult.from_dict(result)

        # Mock the actual file operations since we're testing the cascade logic
        with patch("openevolve.evaluator.run_in_executor") as mock_executor:
            # Mock stage1 to return an error with artifacts
            mock_executor.return_value = EvaluationResult(
                metrics={"stage1_passed": 0.0}, artifacts={"stderr": "Stage 1 compilation error"}
            )

            result = asyncio.run(run_test())

            # Should have failure metrics and artifacts
            self.assertEqual(result.metrics.get("stage1_passed"), 0.0)
            self.assertIn("stderr", result.artifacts)

    def test_artifacts_disabled_integration(self):
        """Test that the full system works with artifacts disabled"""

        with patch.dict(os.environ, {"ENABLE_ARTIFACTS": "false"}):

            async def run_test():
                # Program with error
                bad_code = "invalid syntax"
                program_id = "disabled_test_1"

                # Evaluate
                metrics = await self.evaluator.evaluate_program(bad_code, program_id)

                # Should not have pending artifacts when disabled
                artifacts = self.evaluator.get_pending_artifacts(program_id)
                return metrics, artifacts

            metrics, artifacts = asyncio.run(run_test())

            # Should still get metrics but no artifacts
            self.assertIsInstance(metrics, dict)
            self.assertIsNone(artifacts)

    def test_successful_evaluation_with_artifacts(self):
        """Test that successful evaluations can also have artifacts"""

        async def run_test():
            # Valid Python code
            good_code = "print('Hello, world!')"
            program_id = "success_test_1"

            # Evaluate
            metrics = await self.evaluator.evaluate_program(good_code, program_id)
            artifacts = self.evaluator.get_pending_artifacts(program_id)

            return metrics, artifacts

        metrics, artifacts = asyncio.run(run_test())

        # Should get successful metrics
        self.assertEqual(metrics.get("compile_ok"), 1.0)

        # Should have artifacts from successful compilation
        if artifacts:
            self.assertIn("stdout", artifacts)
            self.assertIn("successful", artifacts["stdout"].lower())


class TestArtifactsRoundTrip(unittest.TestCase):
    """Test that artifacts persist correctly across save/load cycles"""

    def setUp(self):
        # Set up event loop for async operations in tests
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        self.temp_dir = tempfile.mkdtemp()
        config = DatabaseConfig(db_path=self.temp_dir)
        self.database = ProgramDatabase(config)

    def tearDown(self):
        # Clean up event loop if we created one
        if hasattr(self, "loop") and self.loop and not self.loop.is_closed():
            # Cancel any pending tasks
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()
            # Run the loop briefly to let cancellations process
            if pending:
                self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

    def test_small_and_large_artifacts_round_trip(self):
        """Test that stored artifacts can be fetched without saving the database"""
        # Create program with artifacts
        program = Program(id="persist_test_1", code="print('test')", metrics={"score": 0.8})

        artifacts = {
            "stderr": "error message",
            "stdout": "output message",
            "large_log": "x" * (50 * 1024),  # Large artifact
        }

        # Add program and artifacts
        self.database.add(program)
        self.database.store_artifacts(program.id, artifacts)

        loaded_artifacts = self.database.get_artifacts(program.id)

        self.assertEqual(loaded_artifacts["stderr"], artifacts["stderr"])
        self.assertEqual(loaded_artifacts["stdout"], artifacts["stdout"])
        self.assertEqual(loaded_artifacts["large_log"], artifacts["large_log"])


if __name__ == "__main__":
    unittest.main()
