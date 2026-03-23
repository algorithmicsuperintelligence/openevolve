"""
Tests for evaluation timeout functionality in openevolve.evaluator
"""

import asyncio
import os
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

from openevolve.config import EvaluatorConfig
from openevolve.evaluator import Evaluator


class TestEvaluatorTimeout(unittest.TestCase):
    """Tests for evaluator timeout functionality"""

    def setUp(self):
        """Set up test evaluation file"""
        # Create a test evaluation file
        self.test_eval_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)

        # Write test evaluation functions with shorter sleep times for faster tests
        self.test_eval_file.write(
            """
import time

def evaluate(program_path):
    # Read the program to determine behavior
    with open(program_path, 'r') as f:
        code = f.read()
    
    if 'SLEEP_LONG' in code:
        # Sleep for a long time to trigger timeout (reduced for faster tests)
        time.sleep(8)
        return {"score": 1.0}
    elif 'SLEEP_SHORT' in code:
        # Sleep for a short time that should not timeout
        time.sleep(0.5)
        return {"score": 0.8}
    elif 'RAISE_ERROR' in code:
        # Raise an error to trigger retries
        raise RuntimeError("Evaluation failed")
    else:
        # Fast evaluation
        return {"score": 0.5}

def evaluate_stage1(program_path):
    with open(program_path, 'r') as f:
        code = f.read()
    
    if 'STAGE1_TIMEOUT' in code:
        time.sleep(8)
        return {"stage1_score": 1.0}
    else:
        return {"stage1_score": 0.7}

def evaluate_stage2(program_path):
    with open(program_path, 'r') as f:
        code = f.read()
    
    if 'STAGE2_TIMEOUT' in code:
        time.sleep(8)
        return {"stage2_score": 1.0}
    else:
        return {"stage2_score": 0.8}

def evaluate_stage3(program_path):
    with open(program_path, 'r') as f:
        code = f.read()
    
    if 'STAGE3_TIMEOUT' in code:
        time.sleep(8)
        return {"stage3_score": 1.0}
    else:
        return {"stage3_score": 0.9}
"""
        )
        self.test_eval_file.close()

    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.test_eval_file.name):
            os.unlink(self.test_eval_file.name)

    def _create_evaluator(self, timeout=3, cascade_evaluation=False):
        """Helper to create evaluator with given settings (shorter timeout for faster tests)"""
        config = EvaluatorConfig()
        config.timeout = timeout
        config.max_retries = 1  # Minimal retries for faster testing
        config.cascade_evaluation = cascade_evaluation
        config.cascade_thresholds = [0.5, 0.7, 0.9]

        return Evaluator(
            config=config,
            evaluation_file=self.test_eval_file.name,
            llm_ensemble=None,
            prompt_sampler=None,
        )

    def test_fast_evaluation_completes(self):
        """Test that fast evaluations complete successfully"""

        async def run_test():
            evaluator = self._create_evaluator(timeout=3)
            program_code = "def test(): return 'fast'"
            start_time = time.time()

            result = await evaluator.evaluate_program(program_code, "test_fast")

            elapsed_time = time.time() - start_time

            # Should complete quickly
            self.assertLess(elapsed_time, 2.0)
            # Should return successful result
            self.assertIn("score", result)
            self.assertEqual(result["score"], 0.5)
            # Should not have timeout or error flags
            self.assertNotIn("timeout", result)
            self.assertNotIn("error", result)

        asyncio.run(run_test())

    def test_short_evaluation_completes(self):
        """Test that evaluations shorter than timeout complete successfully"""

        async def run_test():
            evaluator = self._create_evaluator(timeout=3)
            program_code = "# SLEEP_SHORT\ndef test(): return 'short'"
            start_time = time.time()

            result = await evaluator.evaluate_program(program_code, "test_short")

            elapsed_time = time.time() - start_time

            # Should complete within timeout
            self.assertLess(elapsed_time, 3)
            # Should return successful result
            self.assertIn("score", result)
            self.assertEqual(result["score"], 0.8)
            # Should not have timeout or error flags
            self.assertNotIn("timeout", result)
            self.assertNotIn("error", result)

        asyncio.run(run_test())

    def test_long_evaluation_times_out(self):
        """Test that long evaluations time out properly"""

        async def run_test():
            evaluator = self._create_evaluator(timeout=3)
            program_code = "# SLEEP_LONG\ndef test(): return 'long'"
            start_time = time.time()

            result = await evaluator.evaluate_program(program_code, "test_long")

            elapsed_time = time.time() - start_time

            # Should complete around the timeout period (allowing some margin)
            self.assertGreater(elapsed_time, 2.5)
            self.assertLess(elapsed_time, 5)

            # Should return timeout result
            self.assertIn("combined_score", result)
            self.assertEqual(result["combined_score"], 0.0)
            self.assertIn("error", result)
            self.assertEqual(result["error"], "Evaluation timed out")
            self.assertIn("timeout", result)
            self.assertTrue(result["timeout"])

        asyncio.run(run_test())

    def test_cascade_evaluation_timeout_stage1(self):
        """Test timeout in cascade evaluation stage 1"""

        async def run_test():
            evaluator = self._create_evaluator(timeout=3, cascade_evaluation=True)
            program_code = "# STAGE1_TIMEOUT\ndef test(): return 'stage1_timeout'"
            start_time = time.time()

            result = await evaluator.evaluate_program(program_code, "test_cascade_stage1")

            elapsed_time = time.time() - start_time

            # Should timeout around the configured timeout
            self.assertGreater(elapsed_time, 2.5)
            self.assertLess(elapsed_time, 5)

            # Should return stage1 timeout result
            self.assertIn("stage1_passed", result)
            self.assertEqual(result["stage1_passed"], 0.0)
            self.assertIn("timeout", result)
            self.assertTrue(result["timeout"])

        asyncio.run(run_test())

    def test_cascade_evaluation_timeout_stage2(self):
        """Test timeout in cascade evaluation stage 2"""

        async def run_test():
            evaluator = self._create_evaluator(timeout=3, cascade_evaluation=True)
            program_code = "# STAGE2_TIMEOUT\ndef test(): return 'stage2_timeout'"
            start_time = time.time()

            result = await evaluator.evaluate_program(program_code, "test_cascade_stage2")

            elapsed_time = time.time() - start_time

            # Should timeout on stage 2, but stage 1 should complete first
            self.assertGreater(elapsed_time, 2.5)
            self.assertLess(elapsed_time, 5)

            # Should have stage1 result but stage2 timeout
            self.assertIn("stage1_score", result)
            self.assertEqual(result["stage1_score"], 0.7)
            self.assertIn("stage2_passed", result)
            self.assertEqual(result["stage2_passed"], 0.0)
            self.assertIn("timeout", result)
            self.assertTrue(result["timeout"])

        asyncio.run(run_test())

    def test_cascade_evaluation_timeout_stage3(self):
        """Test timeout in cascade evaluation stage 3"""

        async def run_test():
            evaluator = self._create_evaluator(timeout=3, cascade_evaluation=True)
            program_code = "# STAGE3_TIMEOUT\ndef test(): return 'stage3_timeout'"
            start_time = time.time()

            result = await evaluator.evaluate_program(program_code, "test_cascade_stage3")

            elapsed_time = time.time() - start_time

            # Should timeout on stage 3, but stages 1 and 2 should complete first
            self.assertGreater(elapsed_time, 2.5)
            self.assertLess(elapsed_time, 5)

            # Should have stage1 and stage2 results but stage3 timeout
            self.assertIn("stage1_score", result)
            self.assertEqual(result["stage1_score"], 0.7)
            self.assertIn("stage2_score", result)
            self.assertEqual(result["stage2_score"], 0.8)
            self.assertIn("stage3_passed", result)
            self.assertEqual(result["stage3_passed"], 0.0)
            self.assertIn("timeout", result)
            self.assertTrue(result["timeout"])

        asyncio.run(run_test())

    def test_timeout_config_respected(self):
        """Test that the timeout configuration value is actually used"""

        async def run_test():
            # Create evaluator with different timeout
            evaluator = self._create_evaluator(timeout=5)

            program_code = "# SLEEP_LONG\ndef test(): return 'long'"
            start_time = time.time()

            result = await evaluator.evaluate_program(program_code, "test_config")

            elapsed_time = time.time() - start_time

            # Should timeout around 5 seconds, not 3
            self.assertGreater(elapsed_time, 4.5)
            self.assertLess(elapsed_time, 7)

            # Should return timeout result
            self.assertIn("timeout", result)
            self.assertTrue(result["timeout"])

        asyncio.run(run_test())

    def test_multiple_retries_with_errors(self):
        """Test that retries work correctly with actual errors (not timeouts)"""

        async def run_test():
            # Create evaluator with more retries
            config = EvaluatorConfig()
            config.timeout = 8  # Long timeout to avoid timeout during this test
            config.max_retries = 2  # 3 total attempts
            config.cascade_evaluation = False

            evaluator = Evaluator(
                config=config,
                evaluation_file=self.test_eval_file.name,
                llm_ensemble=None,
                prompt_sampler=None,
            )

            # Use RAISE_ERROR to trigger actual exceptions that will be retried
            program_code = "# RAISE_ERROR\ndef test(): return 'error'"
            start_time = time.time()

            result = await evaluator.evaluate_program(program_code, "test_retries")

            elapsed_time = time.time() - start_time

            # Should have retried 3 times (max_retries=2 means 3 total attempts)
            # Each attempt should fail quickly, plus 1 second sleep between retries
            # So total time should be around 2-3 seconds (quick failures + 2 sleep periods)
            self.assertGreater(elapsed_time, 1.8)  # At least 2 sleep periods
            self.assertLess(elapsed_time, 4)  # But not too long

            # Should return error result after all retries fail
            self.assertIn("error", result)
            self.assertEqual(result["error"], 0.0)

        asyncio.run(run_test())

    def test_timeout_does_not_trigger_retries(self):
        """Test that timeouts do not trigger retries (correct behavior)"""

        async def run_test():
            # Create evaluator with retries enabled
            config = EvaluatorConfig()
            config.timeout = 2  # Short timeout
            config.max_retries = 2  # Would allow 3 attempts if retries were triggered
            config.cascade_evaluation = False

            evaluator = Evaluator(
                config=config,
                evaluation_file=self.test_eval_file.name,
                llm_ensemble=None,
                prompt_sampler=None,
            )

            # Use SLEEP_LONG to trigger timeout
            program_code = "# SLEEP_LONG\ndef test(): return 'long'"
            start_time = time.time()

            result = await evaluator.evaluate_program(program_code, "test_timeout_no_retry")

            elapsed_time = time.time() - start_time

            # Should timeout only once (~2 seconds), not retry multiple times
            # If retries were happening, this would take ~6 seconds
            self.assertGreater(elapsed_time, 1.8)  # At least the timeout period
            self.assertLess(elapsed_time, 3.5)  # But not multiple timeout periods

            # Should return timeout result
            self.assertIn("timeout", result)
            self.assertTrue(result["timeout"])

        asyncio.run(run_test())

    def test_artifacts_on_timeout(self):
        """Test that timeout artifacts are properly captured"""

        async def run_test():
            # Enable artifacts
            with patch.dict(os.environ, {"ENABLE_ARTIFACTS": "true"}):
                evaluator = self._create_evaluator(timeout=3)
                program_code = "# SLEEP_LONG\ndef test(): return 'long'"

                # Execute evaluation
                result = await evaluator.evaluate_program(program_code, "test_artifacts")

                # Verify timeout occurred
                self.assertIn("timeout", result, "Result should contain timeout flag")
                self.assertTrue(result["timeout"], "Timeout flag should be True")

                # Verify artifacts were captured
                artifacts = evaluator.get_pending_artifacts("test_artifacts")
                self.assertIsNotNone(artifacts, "Artifacts should not be None")

                # Verify required artifact fields
                self.assertIn("failure_stage", artifacts, "Artifacts should contain failure_stage")
                self.assertEqual(
                    artifacts["failure_stage"], "evaluation", "failure_stage should be 'evaluation'"
                )

                self.assertIn("timeout", artifacts, "Artifacts should contain timeout flag")
                self.assertTrue(artifacts["timeout"], "Artifact timeout flag should be True")

                self.assertIn("error_type", artifacts, "Artifacts should contain error_type")
                self.assertEqual(
                    artifacts["error_type"], "timeout", "error_type should be 'timeout'"
                )

                self.assertIn(
                    "timeout_duration", artifacts, "Artifacts should contain timeout_duration"
                )
                self.assertEqual(
                    artifacts["timeout_duration"], 3, "timeout_duration should match config"
                )

                print(f"✅ Artifacts captured correctly: {list(artifacts.keys())}")

        asyncio.run(run_test())


class TestTimeoutIntegration(unittest.TestCase):
    """Integration tests for timeout functionality"""

    def test_real_world_scenario(self):
        """Test a scenario similar to the reported bug"""

        async def run_test():
            # Create a test evaluation file that simulates a long-running evaluation
            test_eval_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)

            test_eval_file.write(
                """
import time

def evaluate(program_path):
    # Simulate a very long evaluation (like the 11-hour case) 
    time.sleep(6)  # 6 seconds to test timeout (reduced for faster tests)
    return {"accReturn": 0.1, "CalmarRatio": 0.9, "combined_score": 0.82}
"""
            )
            test_eval_file.close()

            try:
                # Configure like user's config but with shorter timeout for testing
                config = EvaluatorConfig()
                config.timeout = 3  # 3 seconds instead of 600
                config.max_retries = 1
                config.cascade_evaluation = False
                config.parallel_evaluations = 1

                evaluator = Evaluator(
                    config=config,
                    evaluation_file=test_eval_file.name,
                    llm_ensemble=None,
                    prompt_sampler=None,
                )

                program_code = """
# Financial optimization algorithm
def search_algorithm():
    # This would normally run for hours
    return {"report_type_factor_map": {}}
"""

                start_time = time.time()
                result = await evaluator.evaluate_program(program_code, "financial_test")
                elapsed_time = time.time() - start_time

                # Should timeout in ~3 seconds, not 6+ seconds
                self.assertLess(elapsed_time, 5)
                self.assertGreater(elapsed_time, 2.5)

                # Should return timeout error
                self.assertIn("error", result)
                self.assertIn("timeout", result)
                self.assertTrue(result["timeout"])

            finally:
                if os.path.exists(test_eval_file.name):
                    os.unlink(test_eval_file.name)

        asyncio.run(run_test())


if __name__ == "__main__":
    # Run with verbose output to see test progress
    unittest.main(verbosity=2)
