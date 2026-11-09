"""
Tests for cascade evaluation validation functionality in openevolve.evaluator
"""

import os
import sys
import tempfile
import unittest
from types import ModuleType
from unittest.mock import patch

from openevolve.config import Config
from openevolve.evaluation_result import EvaluationResult
from openevolve.evaluator import Evaluator


class TestCascadeValidation(unittest.IsolatedAsyncioTestCase):
    """Tests for cascade evaluation configuration validation"""

    def setUp(self):
        """Set up test evaluator with cascade validation"""
        self.config = Config()

        # Create temporary evaluator files for testing
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files"""
        # Clean up temp files more safely
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_evaluator_file(self, filename: str, content: str) -> str:
        """Helper to create temporary evaluator file"""
        file_path = os.path.join(self.temp_dir, filename)
        with open(file_path, "w") as f:
            f.write(content)
        return file_path

    def test_cascade_validation_with_valid_evaluator(self):
        """Test cascade validation with evaluator that has cascade functions"""
        # Create evaluator with cascade functions
        evaluator_content = """
def evaluate_stage1(program_path):
    return {"stage1_score": 0.5}

def evaluate_stage2(program_path):
    return {"stage2_score": 0.7}

def evaluate_stage3(program_path):
    return {"stage3_score": 0.9}

def evaluate(program_path):
    return {"final_score": 1.0}
"""
        evaluator_path = self._create_evaluator_file("valid_cascade.py", evaluator_content)

        # Configure for cascade evaluation
        self.config.evaluator.cascade_evaluation = True
        self.config.evaluator.evaluation_file = evaluator_path

        # Should not raise warnings for valid cascade evaluator
        with patch("openevolve.evaluator.logger") as mock_logger:
            evaluator = Evaluator(self.config.evaluator, evaluator_path)

            # Should not have called warning
            mock_logger.warning.assert_not_called()

    def test_cascade_validation_warning_for_missing_functions(self):
        """Test cascade validation warns when cascade functions are missing"""
        # Create evaluator without cascade functions
        evaluator_content = """
def evaluate(program_path):
    return {"score": 0.5}
"""
        evaluator_path = self._create_evaluator_file("no_cascade.py", evaluator_content)

        # Configure for cascade evaluation
        self.config.evaluator.cascade_evaluation = True
        self.config.evaluator.evaluation_file = evaluator_path

        # Should warn about missing cascade functions
        with patch("openevolve.evaluator.logger") as mock_logger:
            evaluator = Evaluator(self.config.evaluator, evaluator_path)

            # Should have warned about missing stage functions
            mock_logger.warning.assert_called()
            warning_call = mock_logger.warning.call_args[0][0]
            self.assertIn("cascade_evaluation: true", warning_call)
            self.assertIn("evaluate_stage1", warning_call)

    def test_cascade_validation_partial_functions(self):
        """Test cascade validation with only some cascade functions"""
        # Create evaluator with only stage1
        evaluator_content = """
def evaluate_stage1(program_path):
    return {"stage1_score": 0.5}

def evaluate(program_path):
    return {"score": 0.5}
"""
        evaluator_path = self._create_evaluator_file("partial_cascade.py", evaluator_content)

        # Configure for cascade evaluation
        self.config.evaluator.cascade_evaluation = True
        self.config.evaluator.evaluation_file = evaluator_path

        # Should warn about missing additional stages
        with patch("openevolve.evaluator.logger") as mock_logger:
            evaluator = Evaluator(self.config.evaluator, evaluator_path)

            # Should warn about missing stage2/stage3
            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args[0][0]
            self.assertIn(
                "defines 'evaluate_stage1' but no additional cascade stages", warning_call
            )

    def test_no_cascade_validation_when_disabled(self):
        """Test no validation when cascade evaluation is disabled"""
        # Create evaluator without cascade functions
        evaluator_content = """
def evaluate(program_path):
    return {"score": 0.5}
"""
        evaluator_path = self._create_evaluator_file("no_cascade.py", evaluator_content)

        # Configure WITHOUT cascade evaluation
        self.config.evaluator.cascade_evaluation = False
        self.config.evaluator.evaluation_file = evaluator_path

        # Should not perform validation or warn
        with patch("openevolve.evaluator.logger") as mock_logger:
            evaluator = Evaluator(self.config.evaluator, evaluator_path)

            # Should not warn when cascade evaluation is disabled
            mock_logger.warning.assert_not_called()

    async def test_direct_evaluate_supports_evaluation_result(self):
        """Test that _direct_evaluate supports EvaluationResult returns"""
        # Create evaluator that returns EvaluationResult
        evaluator_content = """
from openevolve.evaluation_result import EvaluationResult

def evaluate(program_path):
    return EvaluationResult(
        metrics={"score": 0.8, "accuracy": 0.9},
        artifacts={"debug_info": "test data"}
    )
"""
        evaluator_path = self._create_evaluator_file("result_evaluator.py", evaluator_content)

        self.config.evaluator.cascade_evaluation = False
        self.config.evaluator.evaluation_file = evaluator_path
        self.config.evaluator.timeout = 10

        evaluator = Evaluator(self.config.evaluator, evaluator_path)

        # Create a dummy program file
        program_path = self._create_evaluator_file("test_program.py", "def test(): pass")

        # Mock the evaluation function
        def mock_evaluate(path):
            return EvaluationResult(
                metrics={"score": 0.8, "accuracy": 0.9}, artifacts={"debug_info": "test data"}
            )

        evaluator.evaluate_function = mock_evaluate

        # Should handle EvaluationResult without issues
        result = await evaluator._direct_evaluate(program_path)

        # Should return the EvaluationResult as-is
        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(result.metrics["score"], 0.8)
        self.assertEqual(result.artifacts["debug_info"], "test data")

    async def test_direct_evaluate_supports_dict_result(self):
        """Test that _direct_evaluate still supports dict returns"""
        # Create evaluator that returns dict
        evaluator_content = """
def evaluate(program_path):
    return {"score": 0.7, "performance": 0.85}
"""
        evaluator_path = self._create_evaluator_file("dict_evaluator.py", evaluator_content)

        self.config.evaluator.cascade_evaluation = False
        self.config.evaluator.evaluation_file = evaluator_path
        self.config.evaluator.timeout = 10

        evaluator = Evaluator(self.config.evaluator, evaluator_path)

        # Create a dummy program file
        program_path = self._create_evaluator_file("test_program.py", "def test(): pass")

        # Mock the evaluation function directly
        def mock_evaluate(path):
            return {"score": 0.7, "performance": 0.85}

        evaluator.evaluate_function = mock_evaluate

        # Should handle dict result without issues
        result = await evaluator._direct_evaluate(program_path)

        # Should return the dict as-is
        self.assertIsInstance(result, dict)
        self.assertEqual(result["score"], 0.7)
        self.assertEqual(result["performance"], 0.85)

    def test_cascade_validation_with_class_based_evaluator(self):
        """Test cascade validation with class-based evaluator"""
        # Create class-based evaluator with all stages
        evaluator_content = """
class Evaluator:
    def evaluate_stage1(self, program_path):
        return {"stage1_score": 0.5}
    
    def evaluate_stage2(self, program_path):
        return {"stage2_score": 0.7}
    
    def evaluate(self, program_path):
        return {"score": 0.5}

# Module-level functions (what validation looks for)
def evaluate_stage1(program_path):
    evaluator = Evaluator()
    return evaluator.evaluate_stage1(program_path)

def evaluate_stage2(program_path):
    evaluator = Evaluator()
    return evaluator.evaluate_stage2(program_path)

def evaluate(program_path):
    evaluator = Evaluator()
    return evaluator.evaluate(program_path)
"""
        evaluator_path = self._create_evaluator_file("class_cascade.py", evaluator_content)

        # Configure for cascade evaluation
        self.config.evaluator.cascade_evaluation = True
        self.config.evaluator.evaluation_file = evaluator_path

        # Should not warn since module-level functions exist
        with patch("openevolve.evaluator.logger") as mock_logger:
            evaluator = Evaluator(self.config.evaluator, evaluator_path)

            mock_logger.warning.assert_not_called()

    def test_cascade_validation_with_syntax_error(self):
        """Test cascade validation handles syntax errors gracefully"""
        # Create evaluator with syntax error
        evaluator_content = """
def evaluate_stage1(program_path)  # Missing colon
    return {"stage1_score": 0.5}
"""
        evaluator_path = self._create_evaluator_file("syntax_error.py", evaluator_content)

        # Configure for cascade evaluation
        self.config.evaluator.cascade_evaluation = True
        self.config.evaluator.evaluation_file = evaluator_path

        # Should raise an error due to syntax error
        with self.assertRaises(Exception):  # Could be SyntaxError or other import error
            evaluator = Evaluator(self.config.evaluator, evaluator_path)

    def test_cascade_validation_nonexistent_file(self):
        """Test cascade validation with nonexistent evaluator file"""
        # Configure with nonexistent file
        nonexistent_path = "/nonexistent/path.py"
        self.config.evaluator.cascade_evaluation = True
        self.config.evaluator.evaluation_file = nonexistent_path

        # Should raise ValueError for missing file
        with self.assertRaises(ValueError) as context:
            evaluator = Evaluator(self.config.evaluator, nonexistent_path)

        self.assertIn("not found", str(context.exception))

    def test_process_evaluation_result_with_artifacts(self):
        """Test that _process_evaluation_result handles artifacts correctly"""
        evaluator_content = """
def evaluate(program_path):
    return {"score": 0.5}
"""
        evaluator_path = self._create_evaluator_file("dummy.py", evaluator_content)

        self.config.evaluator.cascade_evaluation = False  # Disable cascade to avoid warnings
        self.config.evaluator.evaluation_file = evaluator_path
        evaluator = Evaluator(self.config.evaluator, evaluator_path)

        # Test with EvaluationResult containing artifacts
        eval_result = EvaluationResult(
            metrics={"score": 0.9}, artifacts={"log": "test log", "data": [1, 2, 3]}
        )

        result = evaluator._process_evaluation_result(eval_result)

        self.assertEqual(result.metrics, {"score": 0.9})
        self.assertEqual(result.artifacts, {"log": "test log", "data": [1, 2, 3]})

    def test_process_evaluation_result_with_dict(self):
        """Test that _process_evaluation_result handles dict results correctly"""
        evaluator_content = """
def evaluate(program_path):
    return {"score": 0.5}
"""
        evaluator_path = self._create_evaluator_file("dummy.py", evaluator_content)

        self.config.evaluator.cascade_evaluation = False  # Disable cascade to avoid warnings
        self.config.evaluator.evaluation_file = evaluator_path
        evaluator = Evaluator(self.config.evaluator, evaluator_path)

        # Test with dict result
        dict_result = {"score": 0.7, "accuracy": 0.8}

        result = evaluator._process_evaluation_result(dict_result)

        self.assertEqual(result.metrics, {"score": 0.7, "accuracy": 0.8})
        self.assertEqual(result.artifacts, {})

    async def test_reused_stage_result_does_not_carry_failure_into_next_candidate(self):
        evaluator_path = self._create_evaluator_file(
            "cached_cascade.py",
            """
from asyncio import TimeoutError
from pathlib import Path

import _openevolve_test_cascade_cache as cache

def evaluate(program_path):
    return {}

def evaluate_stage1(program_path):
    return cache.result

def evaluate_stage2(program_path):
    code = Path(program_path).read_text(encoding="utf-8")
    if code == "# timeout":
        raise TimeoutError("stage 2 timed out")
    if code == "# exception":
        raise RuntimeError("stage 2 failed")
    return {"combined_score": 0.9}
""",
        )
        shared_module = ModuleType("_openevolve_test_cascade_cache")
        self.config.evaluator.cascade_evaluation = True
        self.config.evaluator.cascade_thresholds = [0.5]
        with (
            patch.dict(sys.modules, {shared_module.__name__: shared_module}),
            patch.dict(os.environ, {"ENABLE_ARTIFACTS": "true"}),
        ):
            evaluator = Evaluator(self.config.evaluator, evaluator_path)
            self.addCleanup(evaluator._executor.shutdown)
            for failure, artifact in (
                ("timeout", "stage2_timeout"),
                ("exception", "stage2_stderr"),
            ):
                with self.subTest(failure=failure):
                    measured = {"combined_score": 0.8}
                    artifacts = {"stdout": "stage 1 completed"}
                    shared_module.result = EvaluationResult(measured, artifacts)

                    failed = await evaluator.evaluate_program(f"# {failure}", "failed")
                    self.assertEqual(failed["stage2_passed"], 0.0)
                    self.assertIn(artifact, evaluator.get_pending_artifacts("failed"))

                    succeeded = await evaluator.evaluate_program("# success", "succeeded")
                    self.assertEqual(succeeded, {"combined_score": 0.9})
                    self.assertEqual(evaluator.get_pending_artifacts("succeeded"), artifacts)
                    self.assertEqual(measured, {"combined_score": 0.8})
                    self.assertEqual(artifacts, {"stdout": "stage 1 completed"})


if __name__ == "__main__":
    unittest.main()
