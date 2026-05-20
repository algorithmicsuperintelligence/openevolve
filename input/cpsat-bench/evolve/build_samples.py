"""
Build problems.jsonl + stage1/2/3/4 sample files from raw-data/.

Layout (flat, mirroring z3-bench):
    raw-data/<sha>.cpsat.pb                          binary CpModelProto
    raw-data/<sha>__<applied_hash>__seed0.meta.jsonl one-line JSON with
        problem_sha256, problem_filename, cpsat_applied_params,
        cpsat_status (result, elapsed_ms), cpsat_response_stats, ...

problems.jsonl is the full aggregate; sample selection applies a runtime cap
(MAX_BASELINE_MS) and a Tukey IQR outlier filter (k=3.0) so stage quintile
boundaries don't get distorted by long-tail monsters.

Stages (decisive = OPTIMAL or FEASIBLE; this dataset is all OPTIMAL):
  stage1 (5)  center pick from runtime Q1+Q2 (fastest 40%)
  stage2 (5)  center pick from runtime Q3+Q4 (middle 40%)
  stage3 (5)  center pick from runtime Q5    (slowest 20%)
  stage4 (20) quintile-spread broad sample, dedup vs stage1-3
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
MAX_BASELINE_MS = 120_000   # cap — exclude > 2 min monsters from sample pool
OUTLIER_IQR_K = 3.0

STAGE1_STRATEGY = "center"
STAGE2_STRATEGY = "center"
STAGE3_STRATEGY = "center"
STAGE4_STRATEGY = "spread"

_DECISIVE_RESULTS = {"OPTIMAL", "FEASIBLE"}


def _scan_raw():
    """Glob raw-data/*.meta.jsonl (one-line JSON per problem). The meta
    already contains problem_sha256 + problem_filename; no derivation needed."""
    rows = []
    for path in sorted(_RAW.glob("*.meta.jsonl")):
        with open(path) as f:
            line = f.readline().strip()
        if not line:
            continue
        d = json.loads(line)
        rows.append(d)
    return rows


def _runtime_key(d):
    return (d.get("cpsat_status") or {}).get("elapsed_ms", 0)


def _result_key(d):
    return (d.get("cpsat_status") or {}).get("result")


def _id_key(d):
    return d["problem_sha256"]


def _drop_runtime_outliers(rows, k=OUTLIER_IQR_K):
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


def _pick(strategy, sorted_rows, n_pick):
    if strategy == "center":
        return _center_pick(sorted_rows, n_pick)
    if strategy == "spread":
        return _quintile_spread(sorted_rows, n_pick, N_BUCKETS)
    raise ValueError(f"unknown sample strategy: {strategy!r}")


def _center_pick(sorted_rows, n_pick):
    total = len(sorted_rows)
    if total == 0 or n_pick <= 0:
        return []
    if total <= n_pick:
        return list(sorted_rows)
    start = (total - n_pick) // 2
    return sorted_rows[start:start + n_pick]


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
    return {
        "sha": _id_key(d)[:12],
        "baseline_result": _result_key(d),
        "baseline_ms": _runtime_key(d),
    }


def _write_sample(path, picks, label, criteria):
    path.write_text(
        json.dumps(
            {
                "selection": f"{len(picks)} {criteria}",
                "source": str(_PROBLEMS.relative_to(_BENCH.parent)),
                "sha256": [_id_key(d) for d in picks],
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
        raise SystemExit(f"no *.meta.jsonl found under {_RAW}")
    print(f"scanned {len(rows)} problems")

    with open(_PROBLEMS, "w") as f:
        for d in rows:
            f.write(json.dumps(d) + "\n")
    print(f"wrote {_PROBLEMS.relative_to(_BENCH.parent)} ({len(rows)} entries)")

    candidates = [d for d in rows if _runtime_key(d) <= MAX_BASELINE_MS]
    print(f"sample pool: {len(candidates)} (skipped {len(rows) - len(candidates)} "
          f"with baseline_ms > {MAX_BASELINE_MS}ms)")

    candidates, outliers = _drop_runtime_outliers(candidates)
    if outliers:
        print(f"dropped {len(outliers)} runtime outliers (Tukey IQR k={OUTLIER_IQR_K}):")
        for d in sorted(outliers, key=_runtime_key):
            print(f"  {_id_key(d)[:12]}  {int(_runtime_key(d)):>7}ms  {_result_key(d)}")

    decided_by_rt = sorted(
        (d for d in candidates if _result_key(d) in _DECISIVE_RESULTS),
        key=_runtime_key,
    )
    n_decided = len(decided_by_rt)

    def q_idx(i):
        return (i * n_decided) // 5

    pool_q12 = decided_by_rt[q_idx(0):q_idx(2)]
    pool_q34 = decided_by_rt[q_idx(2):q_idx(4)]
    pool_q5 = decided_by_rt[q_idx(4):q_idx(5)]
    print(f"decisive-result runtime pool: {n_decided} | Q1+2={len(pool_q12)} | "
          f"Q3+4={len(pool_q34)} | Q5={len(pool_q5)}")

    s1 = _pick(STAGE1_STRATEGY, pool_q12, STAGE1_N)
    s2 = _pick(STAGE2_STRATEGY, pool_q34, STAGE2_N)
    s3 = _pick(STAGE3_STRATEGY, pool_q5, STAGE3_N)

    # Stage4: broad spread across full decisive pool, dedup vs stage1-3.
    used = {_id_key(d) for d in (s1 + s2 + s3)}
    broad = sorted(
        (d for d in candidates if _id_key(d) not in used),
        key=_runtime_key,
    )
    s4 = _pick(STAGE4_STRATEGY, broad, STAGE4_N)

    _write_sample(_STAGE1, s1, "stage1", "decisive Q1+2 (fastest 40%)")
    _write_sample(_STAGE2, s2, "stage2", "decisive Q3+4 (middle 40%)")
    _write_sample(_STAGE3, s3, "stage3", "decisive Q5 (slowest 20%)")
    _write_sample(_STAGE4, s4, "stage4", "broad runtime spread, dedup vs stage1-3")

    for label, picks in (("stage1", s1), ("stage2", s2),
                         ("stage3", s3), ("stage4", s4)):
        print(f"\n{label}:")
        for d in picks:
            print(f"  {_id_key(d)[:12]}  "
                  f"{str(_result_key(d)):<10}  "
                  f"{int(_runtime_key(d)):>7}ms")


if __name__ == "__main__":
    main()
