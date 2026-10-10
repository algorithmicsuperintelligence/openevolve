"""
Tests for code utilities in openevolve.utils.code_utils
"""

import unittest

from openevolve.utils.code_utils import (
    _format_block_lines,
    apply_diff,
    apply_diff_blocks,
    enforce_evolve_blocks,
    extract_diffs,
    format_diff_summary,
    parse_evolve_blocks,
    parse_full_rewrite,
)


class TestCodeUtils(unittest.TestCase):
    """Tests for code utilities"""

    def test_parse_full_rewrite_line_endings(self):
        for newline in ("\n", "\r\n"):
            with self.subTest(newline=repr(newline)):
                code = f"def solve():{newline}    return 42"
                response = f"Updated program:{newline}```python{newline}{code}{newline}```"
                result = parse_full_rewrite(response)
                self.assertEqual(result, code)
                namespace = {}
                exec(result, namespace)
                self.assertEqual(namespace["solve"](), 42)

    def test_parse_full_rewrite_literal_language(self):
        response = "```c\nint wrong;\n```\n```c++\nint correct;\n```"
        self.assertEqual(parse_full_rewrite(response, "c++"), "int correct;")

    def test_parse_full_rewrite_fallbacks(self):
        for response in ("```\nx = 1\n```", "x = 1"):
            with self.subTest(response=response):
                self.assertEqual(parse_full_rewrite(response), "x = 1")

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


class TestDiffDelimiterValidation(unittest.TestCase):
    """Tests for rejecting malformed SEARCH/REPLACE delimiter sequences"""

    VALID_BLOCK = "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE\n"

    def test_valid_blocks_with_prose(self):
        text = "Plan:\n" + self.VALID_BLOCK + "and\n" + self.VALID_BLOCK.replace("x", "y")
        self.assertEqual(extract_diffs(text), [("x = 1", "x = 2"), ("y = 1", "y = 2")])

    def test_extra_separator_in_replace_raises(self):
        text = "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n=======\nx = 3\n>>>>>>> REPLACE\n"
        with self.assertRaises(ValueError):
            extract_diffs(text)

    def test_extra_separator_does_not_reach_code(self):
        text = "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n=======\n>>>>>>> REPLACE\n"
        with self.assertRaises(ValueError):
            apply_diff("x = 1", text)

    def test_nested_search_raises(self):
        text = "<<<<<<< SEARCH\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE\n"
        with self.assertRaises(ValueError):
            extract_diffs(text)

    def test_stray_marker_outside_block_raises(self):
        for stray in ("=======\n", ">>>>>>> REPLACE\n", "<<<<<<< SEARCH\n"):
            with self.subTest(stray=stray):
                with self.assertRaises(ValueError):
                    extract_diffs(self.VALID_BLOCK + stray)
                with self.assertRaises(ValueError):
                    extract_diffs(stray + self.VALID_BLOCK)

    def test_no_blocks_returns_empty(self):
        self.assertEqual(extract_diffs("no diff here"), [])

    def test_custom_pattern_keeps_its_own_grammar(self):
        pattern = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE\n?"
        text = "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n=======\n>>>>>>> REPLACE\n"
        self.assertEqual(extract_diffs(text, pattern), [("x = 1", "x = 2\n=======")])

    def test_config_default_uses_standard_pattern(self):
        from openevolve.config import Config
        from openevolve.utils.code_utils import _STANDARD_DIFF_PATTERN

        # Validation is keyed on equality with the standard pattern, so the
        # config default must stay identical to it.
        self.assertEqual(Config().diff_pattern, _STANDARD_DIFF_PATTERN)

    def test_config_default_pattern_is_validated(self):
        from openevolve.config import Config

        text = "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n=======\n>>>>>>> REPLACE\n"
        with self.assertRaises(ValueError):
            extract_diffs(text, Config().diff_pattern)

    # --- malformed shapes -------------------------------------------------

    def test_extra_separator_in_second_block_raises(self):
        bad = "<<<<<<< SEARCH\ny = 1\n=======\ny = 2\n=======\n>>>>>>> REPLACE\n"
        with self.assertRaises(ValueError):
            extract_diffs(self.VALID_BLOCK + bad)

    def test_extra_separator_in_search_raises(self):
        text = "<<<<<<< SEARCH\nx = 1\n=======\n=======\nx = 2\n>>>>>>> REPLACE\n"
        with self.assertRaises(ValueError):
            extract_diffs(text)

    def test_missing_separator_raises(self):
        text = "<<<<<<< SEARCH\nx = 1\n>>>>>>> REPLACE\n"
        with self.assertRaisesRegex(ValueError, "Unmatched"):
            extract_diffs(text)

    def test_missing_replace_marker_raises(self):
        text = "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n"
        with self.assertRaisesRegex(ValueError, "Unmatched"):
            extract_diffs(text)

    def test_missing_search_marker_raises(self):
        text = "x = 1\n=======\nx = 2\n>>>>>>> REPLACE\n"
        with self.assertRaisesRegex(ValueError, "Unmatched"):
            extract_diffs(text)

    def test_malformed_block_error_message(self):
        text = "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n=======\n>>>>>>> REPLACE\n"
        with self.assertRaisesRegex(ValueError, "Malformed SEARCH/REPLACE delimiter sequence"):
            extract_diffs(text)

    def test_indented_extra_separator_raises(self):
        text = "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n    =======\n>>>>>>> REPLACE\n"
        with self.assertRaises(ValueError):
            extract_diffs(text)

    def test_separator_with_trailing_whitespace_raises(self):
        text = "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n=======  \t\n>>>>>>> REPLACE\n"
        with self.assertRaises(ValueError):
            extract_diffs(text)

    def test_crlf_marker_is_recognised(self):
        # Standard pattern needs LF after markers, so a CRLF response yields no
        # match; its marker lines are then reported rather than silently ignored.
        text = "<<<<<<< SEARCH\r\nx = 1\r\n=======\r\nx = 2\r\n>>>>>>> REPLACE\r\n"
        with self.assertRaises(ValueError):
            extract_diffs(text)

    def test_apply_diff_leaves_no_partial_changes(self):
        # A valid first block must not be applied when a later block is malformed.
        bad = "<<<<<<< SEARCH\ny = 1\n=======\ny = 2\n=======\n>>>>>>> REPLACE\n"
        original = "x = 1\ny = 1"
        with self.assertRaises(ValueError):
            apply_diff(original, self.VALID_BLOCK + bad)

    # --- look-alikes that are not delimiter lines ---------------------------

    def test_longer_equals_run_is_not_a_marker(self):
        text = "<<<<<<< SEARCH\nx = 1\n=======\n========\nx = 2\n>>>>>>> REPLACE\n"
        self.assertEqual(extract_diffs(text), [("x = 1", "========\nx = 2")])

    def test_shorter_equals_run_is_not_a_marker(self):
        text = "<<<<<<< SEARCH\nx = 1\n=======\n======\nx = 2\n>>>>>>> REPLACE\n"
        self.assertEqual(extract_diffs(text), [("x = 1", "======\nx = 2")])

    def test_separator_inside_code_line_is_allowed(self):
        replace = "print('=======')\n# ======= section\nsep = '=' * 7"
        text = f"<<<<<<< SEARCH\nx = 1\n=======\n{replace}\n>>>>>>> REPLACE\n"
        self.assertEqual(extract_diffs(text), [("x = 1", replace)])

    def test_marker_words_in_prose_are_allowed(self):
        text = (
            "I use the ======= separator and >>>>>>> REPLACE marker below.\n"
            + self.VALID_BLOCK
            + "Done; see <<<<<<< SEARCH above.\n"
        )
        self.assertEqual(extract_diffs(text), [("x = 1", "x = 2")])

    def test_marker_prefix_with_extra_text_is_not_a_marker(self):
        text = self.VALID_BLOCK + "<<<<<<< SEARCHING\n>>>>>>> REPLACEMENT\n"
        self.assertEqual(extract_diffs(text), [("x = 1", "x = 2")])

    # --- valid forms that must keep working -------------------------------

    def test_empty_replace_is_valid(self):
        text = "<<<<<<< SEARCH\nx = 1\n=======\n>>>>>>> REPLACE\n"
        self.assertEqual(extract_diffs(text), [("x = 1", "")])
        self.assertEqual(apply_diff("x = 1\ny = 2", text), "\ny = 2")

    def test_empty_search_is_valid(self):
        text = "<<<<<<< SEARCH\n=======\nx = 2\n>>>>>>> REPLACE\n"
        self.assertEqual(extract_diffs(text), [("", "x = 2")])

    def test_blocks_inside_code_fences_are_valid(self):
        text = "```python\n" + self.VALID_BLOCK + "```\n"
        self.assertEqual(extract_diffs(text), [("x = 1", "x = 2")])

    def test_valid_multi_block_apply(self):
        second = self.VALID_BLOCK.replace("x", "y")
        self.assertEqual(apply_diff("x = 1\ny = 1", self.VALID_BLOCK + second), "x = 2\ny = 2")

    def test_response_ending_without_newline_is_valid(self):
        text = self.VALID_BLOCK.rstrip("\n")
        self.assertEqual(extract_diffs(text), [("x = 1", "x = 2")])


class TestSearchWhitespaceTolerance(unittest.TestCase):
    """SEARCH text that differs only in trailing whitespace still matches (#346)"""

    def test_search_without_trailing_spaces_matches(self):
        original = "def f():   \n    return 1  \n"
        diff = "<<<<<<< SEARCH\ndef f():\n    return 1\n=======\ndef f():\n    return 2\n>>>>>>> REPLACE"
        self.assertEqual(apply_diff(original, diff), "def f():\n    return 2\n")

    def test_search_with_extra_trailing_spaces_matches(self):
        original = "x = 1\ny = 1"
        code, applied = apply_diff_blocks(original, [("x = 1   ", "x = 2")])
        self.assertEqual((code, applied), ("x = 2\ny = 1", 1))

    def test_exact_match_is_preferred(self):
        # The exact occurrence wins over an earlier whitespace-only match.
        original = "x = 1  \nx = 1"
        code, applied = apply_diff_blocks(original, [("x = 1", "x = 2")])
        self.assertEqual((code, applied), ("x = 1  \nx = 2", 1))

    def test_leading_whitespace_is_not_ignored(self):
        original = "    x = 1"
        code, applied = apply_diff_blocks(original, [("x = 1", "x = 2")])
        self.assertEqual((code, applied), ("    x = 1", 0))

    def test_unmatched_block_counts_zero(self):
        code, applied = apply_diff_blocks("x = 1", [("y = 1", "y = 2"), ("x = 1", "x = 3")])
        self.assertEqual((code, applied), ("x = 3", 1))


class TestEvolveBlockParsing(unittest.TestCase):
    """Evolve block markers work with any comment syntax (#422)"""

    def test_python_markers(self):
        code = "a\n# EVOLVE-BLOCK-START\nb\n# EVOLVE-BLOCK-END\nc"
        self.assertEqual(parse_evolve_blocks(code), [(1, 3, "b")])

    def test_other_comment_syntaxes(self):
        for start, end in [
            ("// EVOLVE-BLOCK-START", "// EVOLVE-BLOCK-END"),
            ("//EVOLVE-BLOCK-START", "//EVOLVE-BLOCK-END"),
            ("// #EVOLVE-BLOCK-START", "// #EVOLVE-BLOCK-END"),
            ("/* EVOLVE-BLOCK-START */", "/* EVOLVE-BLOCK-END */"),
            ("-- EVOLVE-BLOCK-START", "-- EVOLVE-BLOCK-END"),
            ("% EVOLVE-BLOCK-START", "% EVOLVE-BLOCK-END"),
            ("    # EVOLVE-BLOCK-START", "    # EVOLVE-BLOCK-END"),
        ]:
            with self.subTest(start=start):
                code = f"int a;\n{start}\nint b;\n{end}\nint c;"
                self.assertEqual(parse_evolve_blocks(code), [(1, 3, "int b;")])

    def test_trailing_python_marker_still_supported(self):
        code = "a  # EVOLVE-BLOCK-START\nb\nc  # EVOLVE-BLOCK-END"
        self.assertEqual(parse_evolve_blocks(code), [(0, 2, "b")])

    def test_marker_word_inside_code_is_not_a_marker(self):
        code = 'msg = "EVOLVE-BLOCK-START"\nx = 1'
        self.assertEqual(parse_evolve_blocks(code), [])

    def test_multiple_blocks(self):
        code = "#EVOLVE-BLOCK-START\na\n#EVOLVE-BLOCK-END\nb\n#EVOLVE-BLOCK-START\nc\n#EVOLVE-BLOCK-END"
        self.assertEqual(parse_evolve_blocks(code), [(0, 2, "a"), (4, 6, "c")])


class TestEnforceEvolveBlocks(unittest.TestCase):
    """Edits outside evolve blocks are reverted when enforcement is on (#106, #422)"""

    PARENT = (
        "import os\n"
        "# EVOLVE-BLOCK-START\n"
        "def solve():\n"
        "    return 1\n"
        "# EVOLVE-BLOCK-END\n"
        "def reward():\n"
        "    return 0\n"
    )

    def test_inside_edits_are_kept(self):
        child = self.PARENT.replace("return 1", "return 2")
        self.assertEqual(enforce_evolve_blocks(self.PARENT, child), child)

    def test_outside_edits_are_reverted(self):
        child = (
            self.PARENT.replace("return 1", "return 2")
            .replace("return 0", "return 999")
            .replace("import os", "import os, sys")
        )
        expected = self.PARENT.replace("return 1", "return 2")
        self.assertEqual(enforce_evolve_blocks(self.PARENT, child), expected)

    def test_new_code_outside_blocks_is_dropped(self):
        child = "HACK = True\n" + self.PARENT.replace("return 1", "return 2") + "extra()\n"
        expected = self.PARENT.replace("return 1", "return 2")
        self.assertEqual(enforce_evolve_blocks(self.PARENT, child), expected)

    def test_block_can_grow_and_shrink(self):
        child = self.PARENT.replace("    return 1\n", "    x = 1\n    y = 2\n    return x + y\n")
        self.assertEqual(enforce_evolve_blocks(self.PARENT, child), child)
        child = self.PARENT.replace("def solve():\n    return 1\n", "")
        self.assertEqual(enforce_evolve_blocks(self.PARENT, child), child)

    def test_multiple_blocks_are_mapped_in_order(self):
        parent = "h\n// EVOLVE-BLOCK-START\na\n// EVOLVE-BLOCK-END\nm\n// EVOLVE-BLOCK-START\nb\n// EVOLVE-BLOCK-END\nt"
        child = "H\n// EVOLVE-BLOCK-START\nA\n// EVOLVE-BLOCK-END\nM\n// EVOLVE-BLOCK-START\nB\n// EVOLVE-BLOCK-END\nT"
        expected = "h\n// EVOLVE-BLOCK-START\nA\n// EVOLVE-BLOCK-END\nm\n// EVOLVE-BLOCK-START\nB\n// EVOLVE-BLOCK-END\nt"
        self.assertEqual(enforce_evolve_blocks(parent, child), expected)

    def test_marker_style_changes_keep_parent_markers(self):
        child = self.PARENT.replace("# EVOLVE-BLOCK-START", "#EVOLVE-BLOCK-START  ").replace(
            "return 1", "return 2"
        )
        expected = self.PARENT.replace("return 1", "return 2")
        self.assertEqual(enforce_evolve_blocks(self.PARENT, child), expected)

    def test_removed_markers_raise(self):
        child = self.PARENT.replace("# EVOLVE-BLOCK-END\n", "")
        with self.assertRaisesRegex(ValueError, "EVOLVE-BLOCK markers"):
            enforce_evolve_blocks(self.PARENT, child)

    def test_all_markers_removed_raise(self):
        child = "def solve():\n    return 2\n"
        with self.assertRaises(ValueError):
            enforce_evolve_blocks(self.PARENT, child)

    def test_extra_block_raises(self):
        child = self.PARENT + "# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END\n"
        with self.assertRaisesRegex(ValueError, "expected 1 evolve block"):
            enforce_evolve_blocks(self.PARENT, child)

    def test_nested_markers_raise(self):
        child = self.PARENT.replace("    return 1", "    # EVOLVE-BLOCK-START\n    return 1")
        with self.assertRaises(ValueError):
            enforce_evolve_blocks(self.PARENT, child)

    def test_parent_without_blocks_is_not_enforced(self):
        parent = "x = 1\n"
        self.assertEqual(enforce_evolve_blocks(parent, "y = 2\n"), "y = 2\n")

    def test_parent_with_malformed_blocks_is_not_enforced(self):
        parent = "# EVOLVE-BLOCK-START\nx = 1\n"
        self.assertEqual(enforce_evolve_blocks(parent, "y = 2\n"), "y = 2\n")

    def test_idempotent(self):
        child = self.PARENT.replace("return 1", "return 2").replace("return 0", "return 9")
        once = enforce_evolve_blocks(self.PARENT, child)
        self.assertEqual(enforce_evolve_blocks(self.PARENT, once), once)


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
