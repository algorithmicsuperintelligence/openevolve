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
  Runtime is clustered into N_BUCKETS via 1D k-means (Lloyd's). Clusters
  are ordered by ascending centroid (c1=fastest, c5=slowest), then merged.

  stage1 (5)  center pick from clusters c1+c2 (fast group)   — default-param sanity
  stage2 (5)  center pick from clusters c3+c4 (mid group)    — default-param sanity
  stage3 (5)  outliers from Statistics/outliers_top.csv      — per-problem tune target
              (top residual, capped at MAX_BASELINE_MS so a single tune
              iteration stays in time budget)
  stage4 (20) quintile-spread broad sample, dedup vs stage1-3

Stage3 sample also writes shared/outliers.json (sha -> {residual, baseline_ms,
n_cons, n_vars}) so the evaluator can mark `is_outlier` on problem records and
phase initial_program.py can branch STAGE3_OVERRIDES on it.
"""
import csv
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_BENCH = _HERE.parent
_RAW = _BENCH / "raw-data"
_PROBLEMS = _BENCH / "problems.jsonl"
_STAGE1 = _HERE / "shared" / "stage1_sample.json"
_STAGE2 = _HERE / "shared" / "stage2_sample.json"
_STAGE3 = _HERE / "shared" / "stage3_sample.json"
_STAGE4 = _HERE / "shared" / "stage4_sample.json"
_OUTLIERS_JSON = _HERE / "shared" / "outliers.json"

# outliers_top.csv lives under cpsat-bench/Statistics/ (or its rename "1/").
_OUTLIERS_CSV_CANDIDATES = [
    _BENCH / "Statistics" / "outliers_top.csv",
    _BENCH / "1" / "outliers_top.csv",
]

STAGE1_N = 5
STAGE2_N = 5
STAGE3_N = 5
STAGE4_N = 20
N_BUCKETS = 5
MAX_BASELINE_MS = 120_000   # cap — exclude > 2 min monsters from sample pool
OUTLIER_IQR_K = 3.0

STAGE1_STRATEGY = "center"
STAGE2_STRATEGY = "center"
STAGE3_STRATEGY = "outliers"   # was "center" — now picks outliers_top.csv
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
    if strategy == "outliers":
        # stage3 uses _pick_outliers() directly; this branch should not fire.
        raise ValueError("'outliers' strategy requires _pick_outliers(), not _pick()")
    raise ValueError(f"unknown sample strategy: {strategy!r}")


def _find_outliers_csv():
    for p in _OUTLIERS_CSV_CANDIDATES:
        if p.exists():
            return p
    return None


def _load_outliers_top(csv_path):
    """Read outliers_top.csv -> ordered list of {sha, residual, elapsed_ms,
    n_vars, n_cons}, sorted by descending residual (already the file's order)."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append({
                    "sha": r["_sha"],
                    "residual": float(r.get("_residual") or 0.0),
                    "elapsed_ms": float(r.get("_elapsed_ms") or 0.0),
                    "n_vars": int(float(r.get("_num_variables") or 0)),
                    "n_cons": int(float(r.get("_num_constraints") or 0)),
                })
            except (KeyError, ValueError) as e:
                print(f"warning: skipping malformed outlier row: {e}",
                      file=sys.stderr)
                continue
    rows.sort(key=lambda d: -d["residual"])
    return rows


def _pick_outliers(rows_by_sha, csv_path, n_pick, max_baseline_ms):
    """Pick top-N residual outliers from outliers_top.csv whose baseline_ms is
    ≤ max_baseline_ms (so a single tune iteration fits time budget).

    Falls back to slowest-cluster center pick (legacy stage3 logic) if csv is
    missing or yields < n_pick eligible entries."""
    if csv_path is None:
        print("warning: outliers_top.csv not found — falling back to "
              "slow-cluster center pick for stage3", file=sys.stderr)
        return None, []

    csv_rows = _load_outliers_top(csv_path)
    picks = []
    used = set()
    diag = []
    for c in csv_rows:
        if len(picks) >= n_pick:
            break
        if c["elapsed_ms"] > max_baseline_ms:
            diag.append((c["sha"][:12], c["elapsed_ms"],
                         "skip: > MAX_BASELINE_MS"))
            continue
        d = rows_by_sha.get(c["sha"])
        if d is None:
            diag.append((c["sha"][:12], c["elapsed_ms"],
                         "skip: not in problems.jsonl"))
            continue
        if c["sha"] in used:
            continue
        picks.append(d)
        used.add(c["sha"])
        diag.append((c["sha"][:12], c["elapsed_ms"],
                     f"pick (residual={c['residual']:.3f}, "
                     f"n_cons={c['n_cons']})"))

    print(f"outliers stage3: from {csv_path.relative_to(_BENCH.parent)}, "
          f"picked {len(picks)}/{n_pick} under MAX_BASELINE_MS={max_baseline_ms}ms")
    for sha12, ms, note in diag:
        print(f"  {sha12}  {int(ms):>10}ms  {note}")

    return picks, csv_rows


def _write_outliers_json(picks, csv_all):
    """Write shared/outliers.json: {sha: {residual, elapsed_ms, n_vars, n_cons}}
    for every entry in outliers_top.csv (not just the stage3 picks). The
    evaluator uses this map to set `is_outlier` on problem records."""
    by_sha = {c["sha"]: {
        "residual": c["residual"],
        "elapsed_ms": c["elapsed_ms"],
        "n_vars": c["n_vars"],
        "n_cons": c["n_cons"],
    } for c in (csv_all or [])}
    stage3_shas = [_id_key(d) for d in picks]
    payload = {
        "stage3_sample": stage3_shas,
        "outliers": by_sha,
    }
    _OUTLIERS_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {_OUTLIERS_JSON.relative_to(_BENCH.parent)} "
          f"({len(by_sha)} outliers, stage3_sample={len(stage3_shas)})")


def _center_pick(sorted_rows, n_pick):
    total = len(sorted_rows)
    if total == 0 or n_pick <= 0:
        return []
    if total <= n_pick:
        return list(sorted_rows)
    start = (total - n_pick) // 2
    return sorted_rows[start:start + n_pick]


def _cluster_runtime(sorted_rows, k, max_iter=100):
    """1D Lloyd's k-means on runtime. Input must be runtime-sorted ascending.
    Returns list of k buckets (lists of rows), ordered by ascending centroid."""
    n = len(sorted_rows)
    if k <= 0:
        return []
    if n == 0:
        return [[] for _ in range(k)]
    if n <= k:
        buckets = [[r] for r in sorted_rows]
        buckets += [[] for _ in range(k - n)]
        return buckets

    values = [_runtime_key(d) for d in sorted_rows]
    # init centroids at k-quantile midpoints over the sorted values.
    centroids = [values[((2 * i + 1) * n) // (2 * k)] for i in range(k)]
    labels = [0] * n

    for _ in range(max_iter):
        changed = False
        for i, v in enumerate(values):
            best, best_d = 0, abs(v - centroids[0])
            for c in range(1, k):
                d = abs(v - centroids[c])
                if d < best_d:
                    best_d, best = d, c
            if labels[i] != best:
                labels[i] = best
                changed = True
        if not changed:
            break
        sums = [0.0] * k
        counts = [0] * k
        for i, v in enumerate(values):
            sums[labels[i]] += v
            counts[labels[i]] += 1
        for c in range(k):
            if counts[c] > 0:
                centroids[c] = sums[c] / counts[c]

    order = sorted(range(k), key=lambda c: centroids[c])
    rank = {old: new for new, old in enumerate(order)}
    buckets = [[] for _ in range(k)]
    for i, lbl in enumerate(labels):
        buckets[rank[lbl]].append(sorted_rows[i])
    return buckets


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

    clusters = _cluster_runtime(decided_by_rt, N_BUCKETS)
    pool_c12 = clusters[0] + clusters[1]
    pool_c34 = clusters[2] + clusters[3]
    pool_c5 = clusters[4]

    def _bucket_range(b):
        if not b:
            return "empty"
        return f"{int(_runtime_key(b[0]))}-{int(_runtime_key(b[-1]))}ms"

    print(f"decisive-result runtime pool: {n_decided} | clusters: " +
          " | ".join(f"c{i+1}({len(b)},{_bucket_range(b)})"
                     for i, b in enumerate(clusters)))
    print(f"stage pools: c1+2={len(pool_c12)} | c3+4={len(pool_c34)} | "
          f"c5={len(pool_c5)}")

    s1 = _pick(STAGE1_STRATEGY, pool_c12, STAGE1_N)
    s2 = _pick(STAGE2_STRATEGY, pool_c34, STAGE2_N)

    # Stage3: outliers from Statistics/outliers_top.csv (top residual, capped
    # at MAX_BASELINE_MS). Fallback to slow-cluster center pick if csv missing
    # or yields nothing usable.
    rows_by_sha = {_id_key(d): d for d in rows}
    outliers_csv = _find_outliers_csv()
    s3_outliers, csv_all = _pick_outliers(rows_by_sha, outliers_csv,
                                          STAGE3_N, MAX_BASELINE_MS) \
        if outliers_csv else (None, [])
    if s3_outliers:
        s3 = s3_outliers
        stage3_criteria = (f"top-{STAGE3_N} residual outliers from "
                           f"{outliers_csv.name}, capped @ "
                           f"{MAX_BASELINE_MS}ms baseline")
    else:
        s3 = _center_pick(pool_c5, STAGE3_N)
        stage3_criteria = ("FALLBACK: decisive runtime cluster c5 "
                           "(outliers_top.csv unavailable or empty)")

    _write_outliers_json(s3 if s3_outliers else [], csv_all)

    # Stage4: broad spread across full decisive pool, dedup vs stage1-3.
    used = {_id_key(d) for d in (s1 + s2 + s3)}
    broad = sorted(
        (d for d in candidates if _id_key(d) not in used),
        key=_runtime_key,
    )
    s4 = _pick(STAGE4_STRATEGY, broad, STAGE4_N)

    _write_sample(_STAGE1, s1, "stage1", "decisive runtime clusters c1+c2 (fast group)")
    _write_sample(_STAGE2, s2, "stage2", "decisive runtime clusters c3+c4 (mid group)")
    _write_sample(_STAGE3, s3, "stage3", stage3_criteria)
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
