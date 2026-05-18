#!/usr/bin/env python3
"""Collect a z3-bench raw-data dump into a usable table + manifest.

Reads:
  <raw-dir>/<sha256>.smt2                                      -> problem instance
  <raw-dir>/<sha256>__<applied_hash>__seed<N>.meta.jsonl       -> one JSON per solve

Writes:
  <out-dir>/problems.csv     -> flattened table, one row per meta entry
  <out-dir>/problems.jsonl   -> combined jsonl with absolute smt2_path appended

Self-contained, stdlib-only, Python 3.7+.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# CSV schema (flattened from meta.jsonl)
# ---------------------------------------------------------------------------

FIELDS = [
    "problem_sha256",
    "applied_params_hash",
    "seed",
    "solver",
    "path",
    "z3_version",
    # status
    "result",
    "elapsed_ms",
    # features
    "num_variables",
    "num_bool",
    "num_int",
    "num_real",
    "num_hard_constraints",
    "num_soft_constraints",
    "num_minimize_objectives",
    "num_maximize_objectives",
    # cli params
    "cli_effort",
    "cli_tech",
    "cli_process",
    "cli_solver_iter_timeout",
    "cli_use_reboot",
    "cli_optimize_m1",
    "cli_optimize_m2",
    "cli_num_of_heights",
    # selected z3 statistics
    "z3_conflicts",
    "z3_decisions",
    "z3_propagations",
    "z3_final_checks",
    "z3_num_checks",
    "z3_max_memory_mb",
    "z3_time_s",
    "z3_rlimit_count",
    # file references
    "smt2_filename",
    "smt2_path",
    "meta_path",
    # diagnostics
    "error",
]


# ---------------------------------------------------------------------------
# Meta parsing
# ---------------------------------------------------------------------------


def _iter_meta_records(meta_path: Path) -> Iterator[Tuple[int, Dict[str, Any], Optional[str]]]:
    """Yield (line_index, record, error) for each non-empty line in a .meta.jsonl."""
    with meta_path.open("r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                yield idx, {}, "json parse: {}".format(exc)
                continue
            yield idx, rec, None


def _seed_from_filename(meta_name: str) -> Optional[int]:
    """Extract <seed> from '<sha>__<hash>__seed<N>.meta.jsonl'."""
    stem = meta_name
    for suffix in (".meta.jsonl", ".jsonl"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    parts = stem.split("__")
    for tok in reversed(parts):
        if tok.startswith("seed"):
            try:
                return int(tok[len("seed") :])
            except ValueError:
                return None
    return None


def _flatten(rec: Dict[str, Any], meta_path: Path, raw_dir: Path) -> Dict[str, object]:
    """Map a meta JSON record into a flat row keyed by FIELDS."""
    row: Dict[str, object] = {k: "" for k in FIELDS}
    row["meta_path"] = str(meta_path)

    row["problem_sha256"] = rec.get("problem_sha256", "")
    row["applied_params_hash"] = rec.get("applied_params_hash", "")
    row["solver"] = rec.get("solver", "")
    row["path"] = rec.get("path", "")
    row["z3_version"] = rec.get("z3_version", "")

    status = rec.get("z3_status") or {}
    row["result"] = status.get("result", "")
    row["elapsed_ms"] = status.get("elapsed_ms", "")

    feats = rec.get("features") or {}
    for k in (
        "num_variables",
        "num_bool",
        "num_int",
        "num_real",
        "num_hard_constraints",
        "num_soft_constraints",
        "num_minimize_objectives",
        "num_maximize_objectives",
    ):
        row[k] = feats.get(k, "")

    cli = rec.get("cli_params") or {}
    row["cli_effort"] = cli.get("effort", "")
    row["cli_tech"] = cli.get("tech", "")
    row["cli_process"] = cli.get("process", "")
    row["cli_solver_iter_timeout"] = cli.get("solver_iter_timeout", "")
    row["cli_use_reboot"] = cli.get("use_reboot", "")
    row["cli_optimize_m1"] = cli.get("optimize_m1", "")
    row["cli_optimize_m2"] = cli.get("optimize_m2", "")
    row["cli_num_of_heights"] = cli.get("num_of_heights", "")

    stats = rec.get("z3_statistics") or {}
    row["z3_conflicts"] = stats.get("conflicts", "")
    row["z3_decisions"] = stats.get("decisions", "")
    row["z3_propagations"] = stats.get("propagations", "")
    row["z3_final_checks"] = stats.get("final checks", "")
    row["z3_num_checks"] = stats.get("num checks", "")
    row["z3_max_memory_mb"] = stats.get("max memory", "")
    row["z3_time_s"] = stats.get("time", "")
    row["z3_rlimit_count"] = stats.get("rlimit count", "")

    seed = rec.get("seed")
    if seed is None:
        seed = _seed_from_filename(meta_path.name)
    row["seed"] = seed if seed is not None else ""

    smt2_name = rec.get("smt2_filename") or ""
    if not smt2_name and row["problem_sha256"]:
        smt2_name = "{}.smt2".format(row["problem_sha256"])
    row["smt2_filename"] = smt2_name

    if smt2_name:
        smt2_path = (raw_dir / smt2_name).resolve()
        row["smt2_path"] = str(smt2_path)
        if not smt2_path.exists():
            row["error"] = "smt2 missing: {}".format(smt2_name)
    else:
        row["error"] = "no smt2_filename in record"

    return row


# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------


def collect(raw_dir: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, Any]]]:
    """Return (csv_rows, jsonl_records_with_smt2_path) sorted by (problem, params, seed)."""
    metas = sorted(raw_dir.glob("*.meta.jsonl"))
    if not metas:
        sys.exit("error: no *.meta.jsonl found in {}".format(raw_dir))

    rows: List[Dict[str, object]] = []
    augmented: List[Dict[str, Any]] = []

    for meta_path in metas:
        for line_idx, rec, parse_err in _iter_meta_records(meta_path):
            if parse_err:
                rows.append(
                    {
                        **{k: "" for k in FIELDS},
                        "meta_path": str(meta_path),
                        "error": "line {}: {}".format(line_idx, parse_err),
                    }
                )
                continue
            row = _flatten(rec, meta_path, raw_dir)
            rows.append(row)
            aug = dict(rec)
            aug["smt2_path"] = row["smt2_path"]
            aug["meta_path"] = str(meta_path)
            if not aug.get("seed") and row["seed"] != "":
                aug["seed"] = row["seed"]
            augmented.append(aug)

    rows.sort(
        key=lambda r: (
            str(r.get("problem_sha256") or ""),
            str(r.get("applied_params_hash") or ""),
            str(r.get("seed") or ""),
        )
    )
    return rows, augmented


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True))
            f.write("\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("/Users/heedeok.son/workspace/openevolve/input/z3-bench/raw-data"),
        help="Directory of paired <sha>.smt2 + <sha>__<hash>__seed<N>.meta.jsonl files.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: parent of --raw-dir).",
    )
    args = p.parse_args()

    raw_dir = args.raw_dir.resolve()
    if not raw_dir.is_dir():
        sys.exit("error: raw dir not found: {}".format(raw_dir))
    out_dir = (args.out_dir or raw_dir.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, augmented = collect(raw_dir)

    csv_path = out_dir / "problems.csv"
    jsonl_path = out_dir / "problems.jsonl"
    write_csv(rows, csv_path)
    write_jsonl(augmented, jsonl_path)

    ok = sum(1 for r in rows if not r["error"])
    failed = len(rows) - ok
    print(
        "[z3-bench collect] {} rows ({} ok, {} with errors) -> {}".format(
            len(rows), ok, failed, csv_path
        )
    )
    print("[z3-bench collect] manifest -> {}".format(jsonl_path))


if __name__ == "__main__":
    main()
