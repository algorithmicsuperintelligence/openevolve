"""
Utilities for code parsing, diffing, and manipulation
"""

import re
from typing import Dict, List, Optional, Tuple, Union


_STANDARD_DIFF_PATTERN = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
_STANDARD_DIFF_MARKER = re.compile(
    r"^[ \t]*(<<<<<<< SEARCH|=======|>>>>>>> REPLACE)[ \t]*\r?$", re.MULTILINE
)

# A marker line starts with any comment syntax (#, //, /*, --, %, <!--, ...)
# followed by the marker token, so evolve blocks work in any language.
_EVOLVE_MARKER = re.compile(r"^\W*EVOLVE-BLOCK-(START|END)\b")


def _evolve_marker(line: str) -> Optional[str]:
    """Return "START" or "END" if the line is an evolve block marker"""
    match = _EVOLVE_MARKER.match(line)
    if match:
        return match.group(1)
    # Also accept a trailing Python-style marker comment, as before
    if "# EVOLVE-BLOCK-START" in line:
        return "START"
    if "# EVOLVE-BLOCK-END" in line:
        return "END"
    return None


def parse_evolve_blocks(code: str) -> List[Tuple[int, int, str]]:
    """
    Parse evolve blocks from code

    Args:
        code: Source code with evolve blocks

    Returns:
        List of tuples (start_line, end_line, block_content)
    """
    lines = code.split("\n")
    blocks = []

    in_block = False
    start_line = -1
    block_content = []

    for i, line in enumerate(lines):
        marker = _evolve_marker(line)
        if marker == "START":
            in_block = True
            start_line = i
            block_content = []
        elif marker == "END" and in_block:
            in_block = False
            blocks.append((start_line, i, "\n".join(block_content)))
        elif in_block:
            block_content.append(line)

    return blocks


def _split_evolve_regions(code: str) -> Optional[Tuple[List[List[str]], List[List[str]]]]:
    """
    Split code into the lines outside and inside evolve blocks

    Returns (outside, blocks) where outside has one more segment than blocks and
    includes the marker lines, or None if the markers are nested or unmatched.
    """
    lines = code.split("\n")
    outside: List[List[str]] = []
    blocks: List[List[str]] = []
    segment_start = 0
    open_line: Optional[int] = None

    for i, line in enumerate(lines):
        marker = _evolve_marker(line)
        if marker == "START":
            if open_line is not None:
                return None
            outside.append(lines[segment_start : i + 1])
            open_line = i
        elif marker == "END":
            if open_line is None:
                return None
            blocks.append(lines[open_line + 1 : i])
            segment_start = i
            open_line = None

    if open_line is not None:
        return None
    outside.append(lines[segment_start:])
    return outside, blocks


def enforce_evolve_blocks(original_code: str, new_code: str) -> str:
    """
    Keep everything outside the evolve blocks identical to the original code

    The evolve block contents of new_code are placed back into the original
    code, so edits made outside the blocks are reverted. If the original code
    has no well-formed evolve blocks, new_code is returned unchanged.

    Args:
        original_code: Code whose evolve block markers define the editable regions
        new_code: Code proposed by the LLM

    Returns:
        new_code with every region outside the evolve blocks restored

    Raises:
        ValueError: new_code does not keep the same number of well-formed
            evolve blocks, so its edits cannot be mapped back.
    """
    original = _split_evolve_regions(original_code)
    if original is None or not original[1]:
        return new_code

    proposed = _split_evolve_regions(new_code)
    if proposed is None or len(proposed[1]) != len(original[1]):
        raise ValueError(
            "EVOLVE-BLOCK markers were changed or removed; "
            f"expected {len(original[1])} evolve block(s)"
        )

    outside, _ = original
    _, blocks = proposed
    result: List[str] = []
    for segment, block in zip(outside, blocks):
        result.extend(segment)
        result.extend(block)
    result.extend(outside[-1])
    return "\n".join(result)


def _find_search_lines(lines: List[str], search_lines: List[str]) -> int:
    """
    Find the first occurrence of search_lines in lines

    Matches exactly first, then ignoring trailing whitespace on each line, since
    LLMs often drop or add trailing spaces in SEARCH text. Returns -1 if absent.
    """
    size = len(search_lines)
    for i in range(len(lines) - size + 1):
        if lines[i : i + size] == search_lines:
            return i

    stripped_search = [line.rstrip() for line in search_lines]
    for i in range(len(lines) - size + 1):
        if [line.rstrip() for line in lines[i : i + size]] == stripped_search:
            return i
    return -1


def apply_diff(
    original_code: str,
    diff_text: str,
    diff_pattern: str = _STANDARD_DIFF_PATTERN,
) -> str:
    """
    Apply a diff to the original code

    Args:
        original_code: Original source code
        diff_text: Diff in the SEARCH/REPLACE format
        diff_pattern: Regex pattern for the SEARCH/REPLACE format

    Returns:
        Modified code
    """
    # Split into lines for easier processing
    original_lines = original_code.split("\n")
    result_lines = original_lines.copy()

    # Extract diff blocks
    diff_blocks = extract_diffs(diff_text, diff_pattern)

    # Apply each diff block
    for search_text, replace_text in diff_blocks:
        search_lines = search_text.split("\n")
        replace_lines = replace_text.split("\n")

        # Find where the search pattern starts in the original code
        i = _find_search_lines(result_lines, search_lines)
        if i >= 0:
            # Replace the matched section
            result_lines[i : i + len(search_lines)] = replace_lines

    return "\n".join(result_lines)


def extract_diffs(
    diff_text: str, diff_pattern: str = _STANDARD_DIFF_PATTERN
) -> List[Tuple[str, str]]:
    """
    Extract diff blocks from the diff text

    Args:
        diff_text: Diff in the SEARCH/REPLACE format
        diff_pattern: Regex pattern for the SEARCH/REPLACE format

    Returns:
        List of tuples (search_text, replace_text)

    Raises:
        ValueError: The standard SEARCH/REPLACE response has an extra or
            unmatched delimiter line. Custom diff patterns retain their own grammar.
    """
    matches = list(re.finditer(diff_pattern, diff_text, re.DOTALL))
    if diff_pattern == _STANDARD_DIFF_PATTERN:
        # A permissive regex can absorb a second separator into replacement
        # text and then insert it as code. Validate the complete response's
        # delimiter lines before applying any matched block.
        cursor = 0
        for match in matches:
            if _STANDARD_DIFF_MARKER.search(diff_text[cursor : match.start()]):
                raise ValueError("Unmatched SEARCH/REPLACE delimiter outside a diff block")
            block_markers = [
                marker.group(1) for marker in _STANDARD_DIFF_MARKER.finditer(match.group(0))
            ]
            if block_markers != ["<<<<<<< SEARCH", "=======", ">>>>>>> REPLACE"]:
                raise ValueError("Malformed SEARCH/REPLACE delimiter sequence")
            cursor = match.end()
        if _STANDARD_DIFF_MARKER.search(diff_text[cursor:]):
            raise ValueError("Unmatched SEARCH/REPLACE delimiter outside a diff block")
    return [(match.group(1).rstrip(), match.group(2).rstrip()) for match in matches]


def parse_full_rewrite(llm_response: str, language: str = "python") -> Optional[str]:
    """
    Extract a full rewrite from an LLM response

    Args:
        llm_response: Response from the LLM
        language: Programming language

    Returns:
        Extracted code or None if not found
    """
    code_block_pattern = r"```" + re.escape(language) + r"\r?\n(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Fallback to any code block
    code_block_pattern = r"```(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Fallback to plain text
    return llm_response


def _format_block_lines(lines: List[str], max_line_len: int = 100, max_lines: int = 30) -> str:
    """Format a block of lines for diff summary: show all lines (truncated per line, optional cap)."""
    truncated = []
    for line in lines[:max_lines]:
        s = line.rstrip()
        if len(s) > max_line_len:
            s = s[: max_line_len - 3] + "..."
        truncated.append("  " + s)
    if len(lines) > max_lines:
        truncated.append(f"  ... ({len(lines) - max_lines} more lines)")
    return "\n".join(truncated) if truncated else "  (empty)"


def format_diff_summary(
    diff_blocks: List[Tuple[str, str]],
    max_line_len: int = 100,
    max_lines: int = 30,
) -> str:
    """
    Create a human-readable summary of the diff.
    For multi-line blocks, shows the full search and replace content (all lines).

    Args:
        diff_blocks: List of (search_text, replace_text) tuples
        max_line_len: Maximum characters per line before truncation (default: 100)
        max_lines: Maximum lines per SEARCH/REPLACE block (default: 30)

    Returns:
        Summary string
    """
    summary = []

    for i, (search_text, replace_text) in enumerate(diff_blocks):
        search_lines = search_text.strip().split("\n")
        replace_lines = replace_text.strip().split("\n")

        if len(search_lines) == 1 and len(replace_lines) == 1:
            summary.append(f"Change {i+1}: '{search_lines[0]}' to '{replace_lines[0]}'")
        else:
            search_block = _format_block_lines(search_lines, max_line_len, max_lines)
            replace_block = _format_block_lines(replace_lines, max_line_len, max_lines)
            summary.append(f"Change {i+1}: Replace:\n{search_block}\nwith:\n{replace_block}")

    return "\n".join(summary)


def calculate_edit_distance(code1: str, code2: str) -> int:
    """
    Calculate the Levenshtein edit distance between two code snippets

    Args:
        code1: First code snippet
        code2: Second code snippet

    Returns:
        Edit distance (number of operations needed to transform code1 into code2)
    """
    if code1 == code2:
        return 0

    # Simple implementation of Levenshtein distance
    m, n = len(code1), len(code2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if code1[i - 1] == code2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    return dp[m][n]


def extract_code_language(code: str) -> str:
    """
    Try to determine the language of a code snippet

    Args:
        code: Code snippet

    Returns:
        Detected language or "unknown"
    """
    # Look for common language signatures
    if re.search(r"^(import|from|def|class)\s", code, re.MULTILINE):
        return "python"
    elif re.search(r"^(package|import java|public class)", code, re.MULTILINE):
        return "java"
    elif re.search(r"^(#include|int main|void main)", code, re.MULTILINE):
        return "cpp"
    elif re.search(r"^(function|var|let|const|console\.log)", code, re.MULTILINE):
        return "javascript"
    elif re.search(r"^(module|fn|let mut|impl)", code, re.MULTILINE):
        return "rust"
    elif re.search(r"^(SELECT|CREATE TABLE|INSERT INTO)", code, re.MULTILINE):
        return "sql"

    return "unknown"


def _can_apply_linewise(haystack_lines: List[str], needle_lines: List[str]) -> bool:
    if not needle_lines:
        return False

    for i in range(len(haystack_lines) - len(needle_lines) + 1):
        if haystack_lines[i : i + len(needle_lines)] == needle_lines:
            return True

    return False


def apply_diff_blocks(original_text: str, diff_blocks: List[Tuple[str, str]]) -> Tuple[str, int]:
    """
    Apply diff blocks line-wise and return (new_text, applied_count)
    """
    lines = original_text.split("\n")
    applied = 0

    for search_text, replace_text in diff_blocks:
        search_lines = search_text.split("\n")
        replace_lines = replace_text.split("\n")

        i = _find_search_lines(lines, search_lines)
        if i >= 0:
            lines[i : i + len(search_lines)] = replace_lines
            applied += 1

    return "\n".join(lines), applied


def split_diffs_by_target(
    diff_blocks: List[Tuple[str, str]],
    *,
    code_text: str,
    changes_description_text: str,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Route diff blocks to either code or changes_description based on exact line-wise match
    of SEARCH text. Returns (code_blocks, changes_desc_blocks, unmatched_blocks)

    If a SEARCH matches both targets, it's ambiguous and we raise error
    """
    code_lines = code_text.split("\n")
    desc_lines = changes_description_text.split("\n")

    code_blocks: List[Tuple[str, str]] = []
    desc_blocks: List[Tuple[str, str]] = []
    unmatched: List[Tuple[str, str]] = []

    for search_text, replace_text in diff_blocks:
        search_lines = search_text.split("\n")

        matches_code = _can_apply_linewise(code_lines, search_lines)
        matches_desc = _can_apply_linewise(desc_lines, search_lines)

        if matches_code and matches_desc:
            raise ValueError(
                "Ambiguous diff block: SEARCH matches both code and changes_description"
            )
        if matches_code:
            code_blocks.append((search_text, replace_text))
        elif matches_desc:
            desc_blocks.append((search_text, replace_text))
        else:
            unmatched.append((search_text, replace_text))

    return code_blocks, desc_blocks, unmatched
