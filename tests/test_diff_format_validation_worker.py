"""
Tests that the iteration worker rejects malformed SEARCH/REPLACE responses
before building or evaluating a child program.
"""

import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test")

from openevolve import process_parallel as process_parallel_module
from openevolve.config import Config
from openevolve.database import Program

PARENT_CODE = "x = 1\ny = 1"
VALID_BLOCK = "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE\n"
EXTRA_SEPARATOR = "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n=======\n>>>>>>> REPLACE\n"
USAGE = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}


class TestWorkerDiffFormatValidation(unittest.TestCase):
    """Run _run_iteration_worker with mocked LLM, prompt sampler and evaluator"""

    def setUp(self):
        self.config = Config()
        self.config.diff_based_evolution = True

        parent = Program(
            id="parent",
            code=PARENT_CODE,
            metrics={"combined_score": 0.5},
            metadata={"island": 0},
        )
        self.snapshot = {
            "programs": {"parent": parent.to_dict()},
            "artifacts": {},
            "islands": [["parent"]],
            "current_island": 0,
            "sampling_island": 0,
            "feature_dimensions": [],
        }

        self.evaluator = MagicMock()
        self.evaluator.evaluate_program = AsyncMock(return_value={"combined_score": 1.0})
        self.evaluator.get_pending_artifacts.return_value = None

        self.sampler = MagicMock()
        self.sampler.build_prompt.return_value = {"system": "sys", "user": "usr"}

    def _run(self, llm_response):
        llm = MagicMock()
        llm.generate_with_context = AsyncMock(return_value=llm_response)
        llm.last_usage = USAGE
        # Worker globals are only created in worker processes, hence create=True.
        with patch.multiple(
            process_parallel_module,
            create=True,
            _worker_config=self.config,
            _worker_llm_ensemble=llm,
            _worker_prompt_sampler=self.sampler,
            _worker_evaluator=self.evaluator,
        ):
            return process_parallel_module._run_iteration_worker(3, self.snapshot, "parent", [])

    def assert_rejected(self, result, message):
        self.assertIsNone(result.child_program_dict)
        self.assertIn(message, result.error)
        self.assertEqual(result.iteration, 3)
        self.assertEqual(result.token_usage, USAGE)
        self.evaluator.evaluate_program.assert_not_called()

    def test_extra_separator_is_rejected_without_evaluation(self):
        result = self._run(EXTRA_SEPARATOR)
        self.assert_rejected(result, "Malformed SEARCH/REPLACE delimiter sequence")

    def test_malformed_later_block_rejects_whole_response(self):
        bad = EXTRA_SEPARATOR.replace("x", "y")
        result = self._run(VALID_BLOCK + bad)
        self.assert_rejected(result, "Malformed SEARCH/REPLACE delimiter sequence")

    def test_stray_marker_is_rejected_without_evaluation(self):
        result = self._run(VALID_BLOCK + "=======\n")
        self.assert_rejected(result, "Unmatched SEARCH/REPLACE delimiter")

    def test_rejected_with_changes_description_mode(self):
        # The changes-description path shares the same extraction step.
        self.config.prompt.programs_as_changes_description = True
        result = self._run(EXTRA_SEPARATOR)
        self.assert_rejected(result, "Malformed SEARCH/REPLACE delimiter sequence")

    def test_no_diffs_error_is_unchanged(self):
        result = self._run("I could not think of a change.")
        self.assert_rejected(result, "No valid diffs found in response")

    def test_valid_diff_is_applied_and_evaluated(self):
        result = self._run("Plan: bump x.\n" + VALID_BLOCK)
        self.assertIsNone(result.error)
        self.assertEqual(result.child_program_dict["code"], "x = 2\ny = 1")
        self.assertEqual(result.child_program_dict["parent_id"], "parent")
        self.assertEqual(result.token_usage, USAGE)
        self.evaluator.evaluate_program.assert_called_once()
        self.assertEqual(self.evaluator.evaluate_program.call_args.args[0], "x = 2\ny = 1")

    def test_custom_diff_pattern_keeps_permissive_parsing(self):
        # Stricter validation applies only to the standard pattern; custom
        # patterns keep their previous behaviour.
        self.config.diff_pattern = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE\n?"
        result = self._run(EXTRA_SEPARATOR)
        self.assertIsNone(result.error)
        self.assertEqual(result.child_program_dict["code"], "x = 2\n=======\ny = 1")
        self.evaluator.evaluate_program.assert_called_once()


if __name__ == "__main__":
    unittest.main()
