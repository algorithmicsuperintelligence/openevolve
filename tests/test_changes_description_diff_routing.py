"""Changes-description routing must use the patch application's matching rules."""

import unittest

from test_diff_format_validation_worker import _WorkerTestBase

from openevolve.utils.code_utils import apply_diff_blocks, split_diffs_by_target


class TestChangesDescriptionRouting(unittest.TestCase):
    def test_routes_search_with_trailing_whitespace_difference(self):
        for source, search in [("x = 1   ", "x = 1"), ("x = 1", "x = 1   ")]:
            with self.subTest(source=source, search=search):
                blocks = [(search, "x = 2"), ("Old description", "New description")]
                code, description, unmatched = split_diffs_by_target(
                    blocks, code_text=source, changes_description_text="Old description   "
                )
                self.assertEqual(unmatched, [])
                self.assertEqual(apply_diff_blocks(source, code), ("x = 2", 1))
                self.assertEqual(
                    apply_diff_blocks("Old description   ", description), ("New description", 1)
                )

    def test_whitespace_ambiguity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Ambiguous diff block"):
            split_diffs_by_target(
                [("shared", "replacement")],
                code_text="shared",
                changes_description_text="shared   ",
            )

    def test_leading_indentation_is_still_significant(self):
        block = ("    return 1", "    return 2")
        result = split_diffs_by_target(
            [block], code_text="return 1", changes_description_text="Unrelated description"
        )
        self.assertEqual(result, ([], [], [block]))


class TestChangesDescriptionWorkerRouting(_WorkerTestBase):
    def setUp(self):
        super().setUp()
        self.config.prompt.programs_as_changes_description = True
        self.snapshot["programs"]["parent"]["changes_description"] = "Old description   "

    def test_worker_applies_both_code_and_description_diffs(self):
        response = (
            "<<<<<<< SEARCH\nx = 1   \n=======\nx = 2\n>>>>>>> REPLACE\n"
            "<<<<<<< SEARCH\nOld description\n=======\nNew description\n>>>>>>> REPLACE\n"
        )
        result = self._run(response)
        self.assertIsNone(result.error)
        self.assertEqual(self.evaluated_code(), "x = 2\ny = 1")
        self.assertEqual(result.child_program_dict["changes_description"], "New description")

    def test_worker_rejects_ambiguous_whitespace_match(self):
        self.snapshot["programs"]["parent"]["changes_description"] = "x = 1   "
        response = "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE\n"
        self.assert_rejected(self._run(response), "Ambiguous diff block")


if __name__ == "__main__":
    unittest.main()
