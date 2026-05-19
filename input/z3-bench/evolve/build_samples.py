"""
Build problems.jsonl + stage1_sample.json + stage2_sample.json from raw-data.

Source of truth = raw-data/*.meta.jsonl. raw-data accumulates over time;
this script (re)scans the directory each run, rewrites problems.jsonl as the
full aggregate, and re-selects stage1 / stage2 samples.

Size metric: (num_hard_constraints, num_variables) tuple ascending — primary
then tiebreak. Runtime (elapsed_ms) is NOT used for selection.

Stage1 (5):  SAT-only, quintile-spread by size (5 buckets, 1 per bucket).
Stage2 (50): SAT + UNSAT, quintile-spread by size (5 buckets, 10 per bucket
             rank-linspace within bucket).

Quintile-spread = stable, deterministic ordering by rank then equal-density
sampling per bucket. Guarantees representation across the size distribution
rather than clustering on fastest / smallest problems.
"""
import json
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
_BENCH = _HERE.parent
_RAW = _BENCH / "raw-data"
_PROBLEMS = _BENCH / "problems.jsonl"
_STAGE1 = _HERE / "shared" / "stage1_sample.json"
_STAGE2 = _HERE / "shared" / "stage2_sample.json"

STAGE1_N = 5
STAGE2_N = 50
N_BUCKETS = 5


def _scan_raw():
    rows = []
    for path in sorted(_RAW.glob("*.meta.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
                break
    return rows


def _size_key(d):
    feats = d.get("features") or {}
    return (feats.get("num_hard_constraints", 0), feats.get("num_variables", 0))


def _quintile_spread(sorted_rows, n_pick, n_buckets=N_BUCKETS):
    total = len(sorted_rows)
    if total == 0 or n_pick <= 0:
        return []
    if total <= n_pick:
        return list(sorted_rows)

    per_bucket = n_pick // n_buckets
    remainder = n_pick % n_buckets
    picked = []
    for b in range(n_buckets):
        lo = (b * total) // n_buckets
        hi = ((b + 1) * total) // n_buckets
        bucket = sorted_rows[lo:hi]
        if not bucket:
            continue
        k = per_bucket + (1 if b < remainder else 0)
        if k <= 0:
            continue
        if k == 1:
            picked.append(bucket[len(bucket) // 2])
        else:
            for j in range(k):
                idx = round(j * (len(bucket) - 1) / (k - 1))
                picked.append(bucket[idx])
    return picked


def _summary(d):
    f = d.get("features") or {}
    z = d.get("z3_status") or {}
    return {
        "sha": d["problem_sha256"][:12],
        "num_hard_constraints": f.get("num_hard_constraints", 0),
        "num_variables": f.get("num_variables", 0),
        "baseline_result": z.get("result"),
        "baseline_ms": z.get("elapsed_ms"),
    }


def _write_sample(path, picks, label, criteria):
    path.write_text(
        json.dumps(
            {
                "selection": f"{len(picks)} {criteria}, quintile-spread by "
                             "(num_hard_constraints, num_variables)",
                "source": str(_PROBLEMS.relative_to(_BENCH.parent)),
                "sha256": [d["problem_sha256"] for d in picks],
                "summary": [_summary(d) for d in picks],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {path.relative_to(_BENCH.parent)} ({len(picks)} {label})")


def main():
    rows = _scan_raw()
    if not rows:
        raise SystemExit(f"no *.meta.jsonl files found under {_RAW}")
    print(f"scanned {len(rows)} raw meta files")

    with open(_PROBLEMS, "w") as f:
        for d in rows:
            f.write(json.dumps(d) + "\n")
    print(f"wrote {_PROBLEMS.relative_to(_BENCH.parent)} ({len(rows)} entries)")

    sat = sorted(
        (d for d in rows if (d.get("z3_status") or {}).get("result") == "Sat"),
        key=_size_key,
    )
    all_sorted = sorted(rows, key=_size_key)

    s1 = _quintile_spread(sat, STAGE1_N, N_BUCKETS)
    s2 = _quintile_spread(all_sorted, STAGE2_N, N_BUCKETS)

    _write_sample(_STAGE1, s1, "stage1", "SAT-only")
    _write_sample(_STAGE2, s2, "stage2", "SAT+UNSAT")

    for label, picks in (("stage1", s1), ("stage2", s2)):
        print(f"\n{label}:")
        for d in picks:
            f_ = d.get("features") or {}
            z = d.get("z3_status") or {}
            print(
                f"  {d['problem_sha256'][:12]}  "
                f"hc={f_.get('num_hard_constraints', 0):>7}  "
                f"vars={f_.get('num_variables', 0):>7}  "
                f"{z.get('result', '?'):<7}  "
                f"{z.get('elapsed_ms', 0):>6}ms"
            )


if __name__ == "__main__":
    main()
