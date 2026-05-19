"""
Build problems.jsonl + stage1/2/3/4 sample files from raw-data.

Source of truth = raw-data/*.meta.jsonl. raw-data accumulates over time;
this script rescans each run, rewrites problems.jsonl as the full aggregate,
and re-selects all stage samples.

Sample pool cap: baseline_ms <= MAX_BASELINE_MS (5 min). problems.jsonl
still contains the full set; only sample selection applies the cap.

Outlier filter: after the cap, problems whose baseline_ms lies beyond
[Q1 - k*IQR, Q3 + k*IQR] (Tukey rule, k=OUTLIER_IQR_K=3.0 "far outlier") are
dropped. k=3 instead of the textbook 1.5 because runtime is heavily right-
skewed — k=1.5 would trim away the entire upper quartile that stage3 is
supposed to test. k=3 only removes the genuine long-tail monsters that
distort quintile boundaries and inflate stage3/4 wall-clock.

Runtime quintiles (SAT-only, sorted by elapsed_ms ascending):
  Q1 = bottom 20%, Q2 = 20-40%, ..., Q5 = top 20%.

Stage1 (5):  SAT, quintiles 1+2 (fastest 40%). Quintile-spread within pool.
             Cascade gate: geomean_speedup >= 1.03 → stage2.
Stage2 (5):  SAT, quintiles 3+4 (middle 40-80%). Quintile-spread within pool.
             Cascade gate: geomean_speedup >= 1.03 → stage3.
Stage3 (5):  SAT, quintile 5 (slowest 20%). Quintile-spread within pool.
             Cascade gate: geomean_speedup >= 1.03 → stage4.
Stage4 (20): SAT+UNSAT, broad runtime-spread, deduplicated against
             stage1+2+3. Final cascade gate / scoring sample.

Quintile-spread = sort by key, split into N_BUCKETS equal-rank buckets,
pick N/N_BUCKETS from each bucket via rank-linspace within bucket.
Deterministic, no randomness.
"""
import json
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
_BENCH = _HERE.parent
_RAW = _BENCH / "raw-data"
_PROBLEMS = _BENCH / "problems.jsonl"
_STAGE1 = _HERE / "shared" / "stage1_sample.json"
_STAGE2 = _HERE / "shared" / "stage2_sample.json"
_STAGE3 = _HERE / "shared" / "stage3_sample.json"
_STAGE4 = _HERE / "shared" / "stage4_sample.json"

STAGE1_N = 5
STAGE2_N = 5
STAGE3_N = 5
STAGE4_N = 20
N_BUCKETS = 5
MAX_BASELINE_MS = 300_000  # 5 min cap — exclude monster problems from sample pool
OUTLIER_IQR_K = 3.0         # linear Tukey k (k=1.5=outlier, k=3=far outlier).
                            # k=3 drops only extreme tails (e.g. 181s, 132s vs ~13s median).


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


def _runtime_key(d):
    return (d.get("z3_status") or {}).get("elapsed_ms", 0)


def _drop_runtime_outliers(rows, k=OUTLIER_IQR_K):
    """
    Remove problems whose baseline_ms lies beyond [Q1 - k*IQR, Q3 + k*IQR].
    Tukey-style linear IQR rule. k=1.5 = standard "outlier", k=3.0 = "far
    outlier" — we use 3.0 because runtime is heavily right-skewed and 1.5
    would trim too aggressively (the upper quartile already lives in the
    long tail).
    Returns (kept_rows, dropped_rows).
    """
    ms_sorted = sorted(_runtime_key(d) for d in rows if _runtime_key(d) > 0)
    n = len(ms_sorted)
    if n < 4:
        return list(rows), []
    q1 = ms_sorted[n // 4]
    q3 = ms_sorted[(3 * n) // 4]
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    kept, dropped = [], []
    for d in rows:
        ms = _runtime_key(d)
        if ms <= 0 or lo <= ms <= hi:
            kept.append(d)
        else:
            dropped.append(d)
    return kept, dropped


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
                "selection": f"{len(picks)} {criteria}",
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

    candidates = [
        d for d in rows
        if (d.get("z3_status") or {}).get("elapsed_ms", 0) <= MAX_BASELINE_MS
    ]
    skipped = len(rows) - len(candidates)
    print(f"sample pool: {len(candidates)} (skipped {skipped} with "
          f"baseline_ms > {MAX_BASELINE_MS}ms)")

    candidates, outliers = _drop_runtime_outliers(candidates)
    if outliers:
        print(f"dropped {len(outliers)} runtime outliers "
              f"(Tukey IQR k={OUTLIER_IQR_K}):")
        for d in sorted(outliers, key=_runtime_key):
            print(f"  {d['problem_sha256'][:12]}  "
                  f"{_runtime_key(d):>7}ms  "
                  f"{(d.get('z3_status') or {}).get('result', '?')}")

    # SAT pool sorted by runtime — basis for stage1/2/3 quintile split.
    sat_by_rt = sorted(
        (d for d in candidates if (d.get("z3_status") or {}).get("result") == "Sat"),
        key=_runtime_key,
    )
    n_sat = len(sat_by_rt)

    def q_idx(i):  # rank boundary for the i-th quintile cut (i in 0..5)
        return (i * n_sat) // 5

    pool_q12 = sat_by_rt[q_idx(0):q_idx(2)]   # quintiles 1+2 (fastest 40%)
    pool_q34 = sat_by_rt[q_idx(2):q_idx(4)]   # quintiles 3+4 (middle 40%)
    pool_q5  = sat_by_rt[q_idx(4):q_idx(5)]   # quintile 5 (slowest 20%)
    print(f"SAT runtime pool: {n_sat} | Q1+2={len(pool_q12)} | "
          f"Q3+4={len(pool_q34)} | Q5={len(pool_q5)}")

    s1 = _quintile_spread(pool_q12, STAGE1_N, N_BUCKETS)
    s2 = _quintile_spread(pool_q34, STAGE2_N, N_BUCKETS)
    s3 = _quintile_spread(pool_q5,  STAGE3_N, N_BUCKETS)

    # Stage4: SAT+UNSAT, broad runtime-spread, exclude SHAs already in stage1+2+3.
    used = {d["problem_sha256"] for d in (s1 + s2 + s3)}
    broad = sorted(
        (d for d in candidates if d["problem_sha256"] not in used),
        key=_runtime_key,
    )
    s4 = _quintile_spread(broad, STAGE4_N, N_BUCKETS)

    _write_sample(_STAGE1, s1, "stage1", "SAT runtime Q1+2 (fastest 40%)")
    _write_sample(_STAGE2, s2, "stage2", "SAT runtime Q3+4 (middle 40%)")
    _write_sample(_STAGE3, s3, "stage3", "SAT runtime Q5 (slowest 20%)")
    _write_sample(_STAGE4, s4, "stage4", "SAT+UNSAT broad, dedup vs stage1-3")

    for label, picks in (("stage1", s1), ("stage2", s2),
                         ("stage3", s3), ("stage4", s4)):
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
