"""
Test suite for artifacts functionality
"""

import asyncio
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from openevolve.config import DatabaseConfig, EvaluatorConfig, PromptConfig
from openevolve.database import Program
from openevolve.database_memory import InMemoryProgramDatabase as ProgramDatabase
from openevolve.evaluation_result import EvaluationResult
from openevolve.evaluator import Evaluator
from openevolve.prompt.sampler import PromptSampler


class TestEvaluationResult(unittest.TestCase):
    """Test the EvaluationResult dataclass"""

    def test_from_dict_compatibility(self):
        """Test that dict -> EvaluationResult -> dict roundtrip works"""
        original_dict = {"accuracy": 0.95, "speed": 0.8}

        # Convert to EvaluationResult
        eval_result = EvaluationResult.from_dict(original_dict)

        # Check structure
        self.assertEqual(eval_result.metrics, original_dict)
        self.assertEqual(eval_result.artifacts, {})

        # Convert back to dict
        result_dict = eval_result.to_dict()
        self.assertEqual(result_dict, original_dict)

    def test_evaluation_result_with_artifacts(self):
        """Test EvaluationResult with artifacts"""
        metrics = {"accuracy": 0.95}
        artifacts = {"stderr": "compilation error", "stdout": "test output"}

        eval_result = EvaluationResult(metrics=metrics, artifacts=artifacts)

        self.assertEqual(eval_result.metrics, metrics)
        self.assertEqual(eval_result.artifacts, artifacts)
        self.assertTrue(eval_result.has_artifacts())
        self.assertEqual(eval_result.get_artifact_keys(), ["stderr", "stdout"])

    def test_artifact_size_calculation(self):
        """Test artifact size calculation"""
        eval_result = EvaluationResult(
            metrics={"score": 1.0}, artifacts={"text": "hello world", "binary": b"binary data"}
        )

        # Text should be encoded to bytes for size calculation
        text_size = eval_result.get_artifact_size("text")
        self.assertEqual(text_size, len("hello world".encode("utf-8")))

        # Binary should return length directly
        binary_size = eval_result.get_artifact_size("binary")
        self.assertEqual(binary_size, len(b"binary data"))

        # Total size
        total_size = eval_result.get_total_artifact_size()
        self.assertEqual(total_size, text_size + binary_size)


class TestDatabaseArtifacts(unittest.TestCase):
    """Test artifact storage in the database"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        config = DatabaseConfig(db_path=self.temp_dir)
        self.database = ProgramDatabase(config)

        # Create a test program
        self.program = Program(id="test_program_1", code="print('hello')", metrics={"score": 0.5})
        self.database.add(self.program)

    def test_store_small_artifacts(self):
        """Test storing small artifacts in JSON"""
        artifacts = {"stderr": "small error message", "stdout": "small output"}

        self.database.store_artifacts(self.program.id, artifacts)

        # Retrieve artifacts
        retrieved = self.database.get_artifacts(self.program.id)
        self.assertEqual(retrieved, artifacts)

        # Check that program has artifacts_json set
        program = self.database.get(self.program.id)
        self.assertIsNotNone(program.artifacts_json)

    def test_store_large_artifacts(self):
        """Test storing large artifacts to disk"""
        large_content = "x" * (50 * 1024)  # 50KB
        artifacts = {"large_output": large_content}

        self.database.store_artifacts(self.program.id, artifacts)

        # Retrieve artifacts
        retrieved = self.database.get_artifacts(self.program.id)
        self.assertEqual(retrieved["large_output"], large_content)

        # Check that program has artifact_dir set
        program = self.database.get(self.program.id)
        self.assertIsNotNone(program.artifact_dir)

    def test_store_mixed_artifacts(self):
        """Test storing both small and large artifacts"""
        small_content = "small message"
        large_content = "y" * (50 * 1024)  # 50KB

        artifacts = {"stderr": small_content, "large_log": large_content}

        self.database.store_artifacts(self.program.id, artifacts)

        # Retrieve all artifacts
        retrieved = self.database.get_artifacts(self.program.id)
        self.assertEqual(retrieved["stderr"], small_content)
        self.assertEqual(retrieved["large_log"], large_content)

    def test_artifacts_disabled(self):
        """Test that artifacts are skipped when disabled"""
        with patch.dict(os.environ, {"ENABLE_ARTIFACTS": "false"}):
            artifacts = {"stderr": "error message"}

            # Should not store artifacts when disabled
            self.database.store_artifacts(self.program.id, artifacts)

            # Should return empty dict
            retrieved = self.database.get_artifacts(self.program.id)
            self.assertEqual(retrieved, {})


class TestEvaluatorArtifacts(unittest.TestCase):
    """Test artifact handling in the evaluator"""

    def setUp(self):
        # Set up event loop for async operations in tests
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        # Create a mock evaluation file
        self.temp_eval_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        self.temp_eval_file.write(
            """
def evaluate(program_path):
    return {"score": 0.5}
"""
        )
        self.temp_eval_file.close()

        config = EvaluatorConfig()
        self.evaluator = Evaluator(config, self.temp_eval_file.name)

    def tearDown(self):
        os.unlink(self.temp_eval_file.name)
        # Clean up event loop if we created one
        if hasattr(self, "loop") and self.loop and not self.loop.is_closed():
            # Cancel any pending tasks
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()
            # Run the loop briefly to let cancellations process
            if pending:
                self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

    def test_evaluate_program_backward_compatibility(self):
        """Test that old evaluators still work unchanged"""

        async def run_test():
            result = await self.evaluator.evaluate_program("print('test')", "test_id")
            self.assertIsInstance(result, dict)
            self.assertIn("score", result)

        asyncio.run(run_test())

    def test_process_evaluation_result_dict(self):
        """Test processing dict results"""
        dict_result = {"accuracy": 0.9, "speed": 0.7}
        eval_result = self.evaluator._process_evaluation_result(dict_result)

        self.assertIsInstance(eval_result, EvaluationResult)
        self.assertEqual(eval_result.metrics, dict_result)
        self.assertEqual(eval_result.artifacts, {})

    def test_process_evaluation_result_evaluation_result(self):
        """Test processing EvaluationResult objects"""
        original = EvaluationResult(metrics={"score": 0.8}, artifacts={"stderr": "warning message"})

        result = self.evaluator._process_evaluation_result(original)
        self.assertEqual(result, original)

    def test_pending_artifacts(self):
        """Test pending artifacts storage and retrieval"""
        program_id = "test_program"
        artifacts = {"stderr": "error", "stdout": "output"}

        # Store artifacts
        self.evaluator._pending_artifacts[program_id] = artifacts

        # Retrieve and check that they're cleared
        retrieved = self.evaluator.get_pending_artifacts(program_id)
        self.assertEqual(retrieved, artifacts)

        # Should be None after retrieval
        second_retrieval = self.evaluator.get_pending_artifacts(program_id)
        self.assertIsNone(second_retrieval)


class TestPromptArtifacts(unittest.TestCase):
    """Test artifact rendering in prompts"""

    def setUp(self):
        config = PromptConfig()
        self.sampler = PromptSampler(config)

    def test_render_artifacts_all_items(self):
        """Test that all artifacts are included using .items() without prioritization"""
        artifacts = {
            "stderr": "error message",
            "stdout": "output message",
            "traceback": "stack trace",
            "other": "other data",
        }

        rendered = self.sampler._render_artifacts(artifacts)

        # All artifacts should be present (no prioritization)
        for key in artifacts.keys():
            self.assertIn(key, rendered)

        # Check that all content is included
        for value in artifacts.values():
            self.assertIn(value, rendered)

    def test_render_artifacts_generic(self):
        """Test that all artifacts are included using .items()"""
        artifacts = {"log1": "first log", "log2": "second log", "config": "configuration data"}

        rendered = self.sampler._render_artifacts(artifacts)

        # All artifacts should be present
        for key in artifacts.keys():
            self.assertIn(key, rendered)

    def test_render_artifacts_truncation(self):
        """Test artifact truncation for large content"""
        # Create content larger than 20KB to trigger truncation
        large_content = "This is a very long log message. " * 700  # Creates ~23KB of text
        artifacts = {"large_log": large_content}

        rendered = self.sampler._render_artifacts(artifacts)

        # Should contain truncation indicator
        self.assertIn("(truncated)", rendered)

    def test_render_artifacts_security_filter(self):
        """Test that security filter redacts potential tokens"""
        # Create content that looks like a token
        token_like_content = "x" * 40  # 40 character string that looks like a token
        artifacts = {"suspicious_log": token_like_content}

        rendered = self.sampler._render_artifacts(artifacts)

        # Should be redacted by security filter
        self.assertIn("<REDACTED_TOKEN>", rendered)

    def test_safe_decode_artifact_string(self):
        """Test safe decoding of string artifacts"""
        text = "hello world"
        decoded = self.sampler._safe_decode_artifact(text)
        self.assertEqual(decoded, text)

    def test_safe_decode_artifact_bytes(self):
        """Test safe decoding of bytes artifacts"""
        text = "hello world"
        binary = text.encode("utf-8")
        decoded = self.sampler._safe_decode_artifact(binary)
        self.assertEqual(decoded, text)

    def test_safe_decode_artifact_invalid_bytes(self):
        """Test safe decoding of invalid bytes"""
        invalid_bytes = b"\xff\xfe\xfd"
        decoded = self.sampler._safe_decode_artifact(invalid_bytes)
        # Should not raise exception and should contain some indication of binary data
        self.assertIsInstance(decoded, str)

    def test_build_prompt_with_artifacts(self):
        """Test building prompt with artifacts"""
        artifacts = {"stderr": "compilation error"}

        prompt = self.sampler.build_prompt(
            current_program="print('test')",
            parent_program="print('test')",
            program_metrics={"score": 0.5},
            previous_programs=[],
            top_programs=[],
            program_artifacts=artifacts,
        )

        # Artifacts should be included in the user message (case-insensitive check)
        self.assertIn("stderr", prompt["user"].lower())
        self.assertIn("compilation error", prompt["user"])

    def test_build_prompt_without_artifacts(self):
        """Test building prompt without artifacts"""
        prompt = self.sampler.build_prompt(
            current_program="print('test')",
            parent_program="print('test')",
            program_metrics={"score": 0.5},
            previous_programs=[],
            top_programs=[],
        )

        # Should work normally without artifacts
        self.assertIn("system", prompt)
        self.assertIn("user", prompt)


if __name__ == "__main__":
    unittest.main()
