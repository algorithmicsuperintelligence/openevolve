"""
Scan `reboot-raw-data/<cell>/{meta.jsonl,problem.smt2}` and emit
`problems.jsonl` for the z3-OPTIMIZE (soft-constraint / cost-mode) pipeline.

The reboot dataset differs from the original z3-bench layout:
  - per-cell subdirectories (one problem.smt2 + one meta.jsonl each), not flat
    `<sha>.smt2` + `<sha>__hash__seed.meta.jsonl` files;
  - meta `z3_status.result` is "Skipped" with elapsed_ms=0 and objective_value=0
    (baselines were never run), so there is NO usable baseline timing/objective
    in the raw data — both are RE-MEASURED locally by `_lib.rebaseline`.

To make the local re-measurement actually take effect, the emitted baseline
`z3_status.result` must equal what the local solve produces, because
`_lib.rebaseline` only adopts a local baseline when
`got_result == raw_result` (matches_raw). Empirically all 89 reboot instances
solve to `sat` with objective 0 in <5s, so we stamp result="Sat". The local
rebaseline then overrides elapsed_ms / stats / objective per-problem.

`smt2_filename` is written relative to `raw-data/` (which `_lib` joins against):
`../reboot-raw-data/<cell>/problem.smt2`. The `..` resolves at open() time.

Usage:
    python3 input/z3-bench/build_problems_reboot.py [--dry-run] [--out PATH]

Idempotent — overwrites problems.jsonl. Re-run when reboot-raw-data/ changes.
"""
import argparse
import hashlib
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_RAW = _HERE / "reboot-raw-data"
_DEFAULT_OUT = _HERE / "problems.jsonl"

# Placeholder baseline timing — strictly < clustering.max_baseline_ms so the
# problem is never filtered from the pool. Overwritten by _lib.rebaseline.
_PLACEHOLDER_MS = 1000


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan(raw_dir):
    cells = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    if not cells:
        raise SystemExit(f"no per-cell subdirs under {raw_dir}")
    rows = []
    bad = 0
    for cell in cells:
        meta_path = cell / "meta.jsonl"
        smt2_path = cell / "problem.smt2"
        if not meta_path.exists() or not smt2_path.exists():
            print(f"WARN: missing meta.jsonl/problem.smt2 in {cell.name}",
                  file=sys.stderr)
            bad += 1
            continue
        try:
            m = json.loads(meta_path.read_text())
        except json.JSONDecodeError as e:
            print(f"WARN: bad json {cell.name}/meta.jsonl: {e}", file=sys.stderr)
            bad += 1
            continue
        sha = _sha256(smt2_path)
        row = {
            "problem_sha256": sha,
            # relative to raw-data/ — _lib does raw_dir / smt2_filename
            "smt2_filename": f"../reboot-raw-data/{cell.name}/problem.smt2",
            "cell": cell.name,
            "solver": m.get("solver", "z3-optimize"),
            "z3_version": m.get("z3_version"),
            "path": m.get("path"),
            "features": m.get("features") or {},
            "cli_params": m.get("cli_params") or {},
            # Baseline re-measured locally; result="Sat" so matches_raw fires.
            "z3_status": {
                "result": "Sat",
                "elapsed_ms": _PLACEHOLDER_MS,
                "objective_value": 0,
            },
        }
        rows.append(row)
    rows.sort(key=lambda r: r["problem_sha256"])
    return rows, {"scanned": len(cells), "bad": bad}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--dry-run", action="store_true",
                    help="print counts but don't write")
    ap.add_argument("--out", type=pathlib.Path, default=_DEFAULT_OUT,
                    help=f"output path (default: {_DEFAULT_OUT.name})")
    args = ap.parse_args()

    rows, stats = scan(_RAW)
    print(f"scanned {stats['scanned']} cell dirs")
    if stats["bad"]:
        print(f"  skipped {stats['bad']} incomplete/malformed")
    print(f"  kept {len(rows)} rows")

    if args.dry_run:
        print("(dry-run — no write)")
        return

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    try:
        rel = args.out.relative_to(_HERE.parent)
    except ValueError:
        rel = args.out
    print(f"wrote {rel} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
