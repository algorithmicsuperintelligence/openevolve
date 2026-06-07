"""Summarize agent verification failures into a reusable taxonomy.

Example:
    python -m pipeline.summarize_failure_taxonomy \
        /tmp/layernorm_pipeline_comparison_gpt4o_attempt10/A_fusion_agent
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _walk_failures(value: Any):
    if isinstance(value, dict):
        error_message = value.get("error_message") or value.get("error")
        error_type = value.get("error_type")
        if error_type or error_message:
            yield {
                "kind": "error",
                "error_type": error_type or "Error",
                "message": str(error_message or ""),
            }
        if value.get("passed") is False and "max_abs" in value:
            yield {
                "kind": "numerical",
                "error_type": "NumericalMismatch",
                "message": (
                    f"output_index={value.get('index')} "
                    f"max_abs={value.get('max_abs')} "
                    f"max_rel={value.get('max_rel')} "
                    f"shape={value.get('shape')}"
                ),
            }
        for child in value.values():
            yield from _walk_failures(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_failures(child)


def classify_failure(message: str, error_type: str) -> str:
    text = message.lower()
    if "arange" in text and "constexpr" in text:
        return "triton_constexpr_arange"
    if "tl.zeros" in text or "tl.full" in text or "shape element" in text:
        if "constexpr" in text:
            return "triton_constexpr_block_shape"
    if "global variable eps" in text or "cannot access global variable" in text:
        return "triton_global_scalar_in_jit"
    if "did not match" in text and ("half" in text or "float" in text):
        return "dtype_mismatch"
    if "invalid for input of size" in text or "shape" in text and "invalid" in text:
        return "trace_shape_constant_leak"
    if "shape mismatch" in text:
        return "shape_mismatch"
    if error_type == "NumericalMismatch":
        return "numerical_mismatch"
    if error_type and "compilation" in error_type.lower():
        return "triton_compilation_error"
    return "other_failure"


def _attempt_number(path: Path) -> int:
    match = re.search(r"attempt_(\d+)", str(path))
    return int(match.group(1)) if match else -1


def summarize(root: Path) -> dict[str, Any]:
    reports = sorted(root.glob("attempt_*/verification_report.json"), key=_attempt_number)
    attempts = []
    taxonomy_counts: Counter[str] = Counter()
    taxonomy_examples: dict[str, str] = {}

    for report_path in reports:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        per_attempt: Counter[str] = Counter()
        examples: dict[str, str] = {}
        for failure in _walk_failures(data):
            category = classify_failure(failure["message"], failure["error_type"])
            per_attempt[category] += 1
            taxonomy_counts[category] += 1
            example = failure["message"].replace("\n", " ")[:500]
            examples.setdefault(category, example)
            taxonomy_examples.setdefault(category, example)
        attempts.append(
            {
                "attempt": _attempt_number(report_path),
                "passed": bool(data.get("passed")),
                "categories": dict(per_attempt),
                "examples": examples,
                "report": str(report_path),
            }
        )

    repeated_categories = defaultdict(list)
    for attempt in attempts:
        for category in attempt["categories"]:
            repeated_categories[category].append(attempt["attempt"])

    return {
        "root": str(root),
        "attempts": attempts,
        "taxonomy_counts": dict(taxonomy_counts),
        "taxonomy_examples": taxonomy_examples,
        "repeated_categories": {
            key: value for key, value in repeated_categories.items() if len(value) > 1
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Failure Taxonomy Summary",
        "",
        f"Root: `{summary['root']}`",
        "",
        "## Taxonomy Counts",
        "",
    ]
    for category, count in sorted(summary["taxonomy_counts"].items()):
        lines.append(f"- `{category}`: {count}")
    if not summary["taxonomy_counts"]:
        lines.append("- No failures found.")

    lines.extend(["", "## Attempts", ""])
    for attempt in summary["attempts"]:
        lines.append(f"### Attempt {attempt['attempt']:03d}")
        lines.append(f"- passed: `{attempt['passed']}`")
        if attempt["categories"]:
            for category, count in sorted(attempt["categories"].items()):
                example = attempt["examples"].get(category, "")
                lines.append(f"- `{category}`: {count}")
                lines.append(f"  example: {example}")
        else:
            lines.append("- No classified failure details.")
        lines.append("")

    if summary["repeated_categories"]:
        lines.extend(["## Repeated Failure Categories", ""])
        for category, attempts in sorted(summary["repeated_categories"].items()):
            joined = ", ".join(f"{attempt:03d}" for attempt in attempts)
            lines.append(f"- `{category}` repeated in attempts: {joined}")

    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize verification failures")
    parser.add_argument("root", help="Agent output directory containing attempt_*/verification_report.json")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--md-out", default=None)
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    summary = summarize(root)
    json_out = Path(args.json_out).expanduser().resolve() if args.json_out else root / "failure_taxonomy.json"
    md_out = Path(args.md_out).expanduser().resolve() if args.md_out else root / "failure_taxonomy.md"
    json_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    md_out.write_text(render_markdown(summary), encoding="utf-8")
    print(f"Wrote {json_out}")
    print(f"Wrote {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
