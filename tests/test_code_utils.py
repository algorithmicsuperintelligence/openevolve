"""
Tests for code utilities in openevolve.utils.code_utils
"""

import unittest

from openevolve.utils.code_utils import (
    DiffApplicationError,
    _format_block_lines,
    apply_diff,
    apply_diff_strict,
    extract_diffs,
    format_diff_summary,
    validate_evolve_blocks,
)


class TestCodeUtils(unittest.TestCase):
    """Tests for code utilities"""

    def test_extract_diffs(self):
        """Test extracting diffs from a response"""
        diff_text = """
        Let's improve this code:

        <<<<<<< SEARCH
        def hello():
            print("Hello")
        =======
        def hello():
            print("Hello, World!")
        >>>>>>> REPLACE

        Another change:

        <<<<<<< SEARCH
        x = 1
        =======
        x = 2
        >>>>>>> REPLACE
        """

        diffs = extract_diffs(diff_text)
        self.assertEqual(len(diffs), 2)
        self.assertEqual(
            diffs[0][0],
            """        def hello():
            print(\"Hello\")""",
        )
        self.assertEqual(
            diffs[0][1],
            """        def hello():
            print(\"Hello, World!\")""",
        )
        self.assertEqual(diffs[1][0], "        x = 1")
        self.assertEqual(diffs[1][1], "        x = 2")

    def test_apply_diff(self):
        """Test applying diffs to code"""
        original_code = """
        def hello():
            print("Hello")

        x = 1
        y = 2
        """

        diff_text = """
        <<<<<<< SEARCH
        def hello():
            print("Hello")
        =======
        def hello():
            print("Hello, World!")
        >>>>>>> REPLACE

        <<<<<<< SEARCH
        x = 1
        =======
        x = 2
        >>>>>>> REPLACE
        """

        expected_code = """
        def hello():
            print("Hello, World!")

        x = 2
        y = 2
        """

        result = apply_diff(original_code, diff_text)

        # Normalize whitespace for comparison
        self.assertEqual(
            result,
            expected_code,
        )

    def test_strict_diff_requires_exactly_one_match(self):
        original = "x = 1\nx = 1\n"
        diff = "\n".join(
            ["<" * 7 + " SEARCH", "x = 1", "=" * 7, "x = 2", ">" * 7 + " REPLACE"]
        )
        with self.assertRaises(DiffApplicationError):
            apply_diff_strict(original, diff)

    def test_strict_diff_is_atomic(self):
        original = "x = 1\ny = 2\n"
        diff = "\n".join(
            [
                "<" * 7 + " SEARCH",
                "x = 1",
                "=" * 7,
                "x = 3",
                ">" * 7 + " REPLACE",
                "<" * 7 + " SEARCH",
                "missing = 0",
                "=" * 7,
                "missing = 1",
                ">" * 7 + " REPLACE",
            ]
        )
        with self.assertRaises(DiffApplicationError):
            apply_diff_strict(original, diff)
        self.assertEqual(original, "x = 1\ny = 2\n")

    def test_strict_diff_can_enforce_evolve_blocks(self):
        original = """header = 1
# EVOLVE-BLOCK-START
x = 1
# EVOLVE-BLOCK-END
footer = 2"""
        inside = "\n".join(
            ["<" * 7 + " SEARCH", "x = 1", "=" * 7, "x = 2", ">" * 7 + " REPLACE"]
        )
        self.assertIn(
            "x = 2",
            apply_diff_strict(original, inside, enforce_evolve_blocks=True),
        )
        outside = "\n".join(
            [
                "<" * 7 + " SEARCH",
                "header = 1",
                "=" * 7,
                "header = 2",
                ">" * 7 + " REPLACE",
            ]
        )
        with self.assertRaises(DiffApplicationError):
            apply_diff_strict(original, outside, enforce_evolve_blocks=True)

    def test_strict_diff_applies_sequential_dependent_blocks(self):
        original = """# EVOLVE-BLOCK-START
x = 1
# EVOLVE-BLOCK-END"""
        diff = "\n".join(
            [
                "<" * 7 + " SEARCH",
                "x = 1",
                "=" * 7,
                "x = 2",
                ">" * 7 + " REPLACE",
                "<" * 7 + " SEARCH",
                "x = 2",
                "=" * 7,
                "x = 3",
                ">" * 7 + " REPLACE",
            ]
        )
        result = apply_diff_strict(original, diff, enforce_evolve_blocks=True)
        self.assertIn("x = 3", result)

    def test_evolve_blocks_must_be_balanced_and_non_nested(self):
        with self.assertRaises(DiffApplicationError):
            validate_evolve_blocks("# EVOLVE-BLOCK-START\nx = 1")
        with self.assertRaises(DiffApplicationError):
            validate_evolve_blocks("# EVOLVE-BLOCK-END")
        nested = """# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-START
x = 1
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END"""
        with self.assertRaises(DiffApplicationError):
            validate_evolve_blocks(nested)


class TestFormatDiffSummary(unittest.TestCase):
    """Tests for format_diff_summary showing actual diff content"""

    def test_single_line_changes(self):
        """Single-line changes should show inline format"""
        diff_blocks = [("x = 1", "x = 2")]
        result = format_diff_summary(diff_blocks)
        self.assertEqual(result, "Change 1: 'x = 1' to 'x = 2'")

    def test_multi_line_changes_show_actual_content(self):
        """Multi-line changes should show actual SEARCH/REPLACE content"""
        diff_blocks = [
            (
                "def old():\n    return False",
                "def new():\n    return True",
            )
        ]
        result = format_diff_summary(diff_blocks)
        # Should contain actual code, not "2 lines"
        self.assertIn("def old():", result)
        self.assertIn("return False", result)
        self.assertIn("def new():", result)
        self.assertIn("return True", result)
        self.assertIn("Replace:", result)
        self.assertIn("with:", result)
        # Should NOT contain generic line count
        self.assertNotIn("2 lines", result)

    def test_multiple_diff_blocks(self):
        """Multiple diff blocks should be numbered"""
        diff_blocks = [
            ("a = 1", "a = 2"),
            ("def foo():\n    pass", "def bar():\n    return 1"),
        ]
        result = format_diff_summary(diff_blocks)
        self.assertIn("Change 1:", result)
        self.assertIn("Change 2:", result)
        self.assertIn("'a = 1' to 'a = 2'", result)
        self.assertIn("def foo():", result)
        self.assertIn("def bar():", result)

    def test_configurable_max_line_len(self):
        """max_line_len parameter should control line truncation"""
        long_line = "x" * 50
        # Must be multi-line to trigger block format (single-line uses inline format)
        diff_blocks = [(long_line + "\nline2", "short\nline2")]
        # With default (100), no truncation
        result_default = format_diff_summary(diff_blocks)
        self.assertNotIn("...", result_default)
        # With max_line_len=30, should truncate the long line
        result_short = format_diff_summary(diff_blocks, max_line_len=30)
        self.assertIn("...", result_short)

    def test_configurable_max_lines(self):
        """max_lines parameter should control block truncation"""
        many_lines = "\n".join([f"line{i}" for i in range(20)])
        diff_blocks = [(many_lines, "replacement")]
        # With max_lines=10, should truncate
        result = format_diff_summary(diff_blocks, max_lines=10)
        self.assertIn("... (10 more lines)", result)

    def test_block_lines_basic_formatting(self):
        """Lines should be indented with 2 spaces"""
        lines = ["line1", "line2"]
        result = _format_block_lines(lines)
        self.assertEqual(result, "  line1\n  line2")

    def test_block_lines_long_line_truncation(self):
        """Lines over 100 chars should be truncated by default"""
        long_line = "x" * 150
        result = _format_block_lines([long_line])
        self.assertIn("...", result)
        self.assertLess(len(result.split("\n")[0]), 110)

    def test_block_lines_many_lines_truncation(self):
        """More than 30 lines should show truncation message by default"""
        lines = [f"line{i}" for i in range(50)]
        result = _format_block_lines(lines)
        self.assertIn("... (20 more lines)", result)
        self.assertEqual(len(result.split("\n")), 31)

    def test_block_lines_empty_input(self):
        """Empty input should return '(empty)'"""
        result = _format_block_lines([])
        self.assertEqual(result, "  (empty)")


if __name__ == "__main__":
    unittest.main()
