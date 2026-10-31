"""
Tests that the iteration worker rejects malformed, unapplied or no-op LLM
responses before evaluating a child program, and enforces EVOLVE-BLOCK regions
when configured.
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


class _WorkerTestBase(unittest.TestCase):
    """Run _run_iteration_worker with mocked LLM, prompt sampler and evaluator"""

    parent_code = PARENT_CODE

    def setUp(self):
        self.config = Config()
        self.config.diff_based_evolution = True
        self.config.language = "python"  # set by the controller in real runs

        parent = Program(
            id="parent",
            code=self.parent_code,
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

    def evaluated_code(self):
        self.evaluator.evaluate_program.assert_called_once()
        return self.evaluator.evaluate_program.call_args.args[0]


class TestWorkerDiffFormatValidation(_WorkerTestBase):
    """Malformed SEARCH/REPLACE delimiters are rejected before evaluation"""

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


class TestWorkerUnappliedDiffs(_WorkerTestBase):
    """Diffs that do not change the parent are reported, not evaluated (#346)"""

    def test_no_matching_search_block_is_rejected(self):
        response = "<<<<<<< SEARCH\nz = 1\n=======\nz = 2\n>>>>>>> REPLACE\n"
        result = self._run(response)
        self.assert_rejected(result, "None of the 1 SEARCH block(s) matched")

    def test_partial_match_applies_matching_blocks(self):
        response = VALID_BLOCK + "<<<<<<< SEARCH\nz = 1\n=======\nz = 2\n>>>>>>> REPLACE\n"
        with self.assertLogs(process_parallel_module.logger, level="WARNING") as logs:
            result = self._run(response)
        self.assertIsNone(result.error)
        self.assertEqual(self.evaluated_code(), "x = 2\ny = 1")
        self.assertIn("only 1 of 2 SEARCH block(s) matched", "\n".join(logs.output))

    def test_diff_that_changes_nothing_is_rejected(self):
        response = "<<<<<<< SEARCH\nx = 1\n=======\nx = 1\n>>>>>>> REPLACE\n"
        result = self._run(response)
        self.assert_rejected(result, "Diff did not change the parent program")

    def test_trailing_whitespace_mismatch_still_applies(self):
        response = "<<<<<<< SEARCH\nx = 1   \n=======\nx = 2\n>>>>>>> REPLACE\n"
        result = self._run(response)
        self.assertIsNone(result.error)
        self.assertEqual(self.evaluated_code(), "x = 2\ny = 1")

    def test_identical_full_rewrite_is_rejected(self):
        self.config.diff_based_evolution = False
        result = self._run(f"```python\n{PARENT_CODE}\n```")
        self.assert_rejected(result, "Rewrite is identical to the parent program")

    def test_changed_full_rewrite_is_evaluated(self):
        self.config.diff_based_evolution = False
        result = self._run("```python\nx = 5\ny = 1\n```")
        self.assertIsNone(result.error)
        self.assertEqual(self.evaluated_code(), "x = 5\ny = 1")


BLOCK_PARENT = (
    "import os\n"
    "# EVOLVE-BLOCK-START\n"
    "def solve():\n"
    "    return 1\n"
    "# EVOLVE-BLOCK-END\n"
    "def reward():\n"
    "    return 0"
)


def _diff(search, replace):
    return f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


class TestWorkerEnforceEvolveBlocks(_WorkerTestBase):
    """enforce_evolve_blocks reverts edits outside EVOLVE-BLOCK regions (#106, #422)"""

    parent_code = BLOCK_PARENT

    def setUp(self):
        super().setUp()
        self.config.enforce_evolve_blocks = True

    def test_disabled_by_default(self):
        self.assertFalse(Config().enforce_evolve_blocks)

    def test_outside_edits_kept_when_disabled(self):
        self.config.enforce_evolve_blocks = False
        result = self._run(_diff("    return 0", "    return 999"))
        self.assertIsNone(result.error)
        self.assertIn("return 999", self.evaluated_code())

    def test_inside_edit_is_evaluated_unchanged(self):
        result = self._run(_diff("    return 1", "    return 2"))
        self.assertIsNone(result.error)
        self.assertEqual(self.evaluated_code(), BLOCK_PARENT.replace("return 1", "return 2"))

    def test_outside_edit_is_reverted_before_evaluation(self):
        response = _diff("    return 1", "    return 2") + _diff("    return 0", "    return 999")
        with self.assertLogs(process_parallel_module.logger, level="INFO") as logs:
            result = self._run(response)
        self.assertIsNone(result.error)
        expected = BLOCK_PARENT.replace("return 1", "return 2")
        self.assertEqual(self.evaluated_code(), expected)
        self.assertEqual(result.child_program_dict["code"], expected)
        self.assertIn("reverted edits outside the EVOLVE-BLOCK regions", "\n".join(logs.output))

    def test_only_outside_edits_are_rejected(self):
        result = self._run(_diff("    return 0", "    return 999"))
        self.assert_rejected(result, "All edits were outside the EVOLVE-BLOCK regions")

    def test_removed_marker_is_rejected(self):
        result = self._run(_diff("# EVOLVE-BLOCK-END", "# the end"))
        self.assert_rejected(result, "EVOLVE-BLOCK markers were changed or removed")

    def test_full_rewrite_outside_edits_are_reverted(self):
        self.config.diff_based_evolution = False
        rewrite = "import os, sys\n" + BLOCK_PARENT.split("\n", 1)[1].replace(
            "return 1", "return 2"
        ).replace("return 0", "return 999")
        result = self._run(f"```python\n{rewrite}\n```")
        self.assertIsNone(result.error)
        self.assertEqual(self.evaluated_code(), BLOCK_PARENT.replace("return 1", "return 2"))

    def test_config_loads_from_dict(self):
        config = Config.from_dict({"enforce_evolve_blocks": True})
        self.assertTrue(config.enforce_evolve_blocks)
        self.assertTrue(config.to_dict()["enforce_evolve_blocks"])


if __name__ == "__main__":
    unittest.main()
