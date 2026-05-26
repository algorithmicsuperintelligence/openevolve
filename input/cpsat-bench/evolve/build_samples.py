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
              (top residual, capped at STAGE3_MAX_BASELINE_MS — higher than
              MAX_BASELINE_MS so genuinely slow outliers enter the tune set;
              still bounded so a single evolve iteration finishes)
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

STAGE1_N = 10
STAGE2_N = 10
STAGE3_N = 5
STAGE4_N = 20
N_BUCKETS = 5
# Global cap for stage1/2/4 sample pool. Anything slower than 2 min skews
# the quintile clustering.
MAX_BASELINE_MS = 120_000
# Stage3 (outlier-only) gets a higher cap so genuinely slow outliers can
# enter the tune set. 25 min ≈ what one stage3 problem can chew through
# under W=8 (raw-data times are W=8-equivalent). Variant timeout =
# baseline_ms * 1.3 in evaluator.
STAGE3_MAX_BASELINE_MS = 1_500_000
# Stratified stage3 pick — outliers cluster in two size regimes (small
# SAT-like and large LP-heavy). Picking only by residual order biases to
# small ones (they pass the cap easier). Bands:
#   small  : elapsed_ms <  10_000          (≤10s, vars≈1700-2000)
#   mid    : 10_000  ≤ ms <   500_000      (10s-500s, vars≈17000)
#   large  : 500_000 ≤ ms ≤ STAGE3_MAX     (500s-1500s, vars≈20000)
# Within each band pick top-N by residual.
STAGE3_SMALL_MS = 10_000
STAGE3_MID_MS = 500_000
STAGE3_PICK_SMALL = 2
STAGE3_PICK_MID = 2
STAGE3_PICK_LARGE = 1
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


def _pick_outliers(rows_by_sha, csv_path):
    """Stratified outlier pick: bucket csv rows by elapsed_ms into
    small/mid/large, then within each bucket pick top-N by residual.

    Returns (picks, csv_rows). picks is in [small..., mid..., large...] order.
    csv_rows is the full csv (for outliers.json metadata).

    Falls back to (None, []) if csv missing — caller uses legacy slow-cluster
    center pick then."""
    if csv_path is None:
        print("warning: outliers_top.csv not found — falling back to "
              "slow-cluster center pick for stage3", file=sys.stderr)
        return None, []

    csv_rows = _load_outliers_top(csv_path)

    def _band(ms):
        if ms <= 0 or ms > STAGE3_MAX_BASELINE_MS:
            return None
        if ms < STAGE3_SMALL_MS:
            return "small"
        if ms < STAGE3_MID_MS:
            return "mid"
        return "large"

    by_band = {"small": [], "mid": [], "large": []}
    for c in csv_rows:
        if c["sha"] not in rows_by_sha:
            continue
        band = _band(c["elapsed_ms"])
        if band is None:
            continue
        by_band[band].append(c)
    # csv_rows already sorted by descending residual → by_band lists inherit.

    band_targets = {
        "small": STAGE3_PICK_SMALL,
        "mid": STAGE3_PICK_MID,
        "large": STAGE3_PICK_LARGE,
    }

    picks = []
    used = set()
    diag = []
    for band in ("small", "mid", "large"):
        target = band_targets[band]
        taken = 0
        for c in by_band[band]:
            if taken >= target:
                break
            if c["sha"] in used:
                continue
            d = rows_by_sha[c["sha"]]
            picks.append(d)
            used.add(c["sha"])
            taken += 1
            diag.append((band, c["sha"][:12], c["elapsed_ms"],
                         f"pick (residual={c['residual']:.3f}, "
                         f"n_cons={c['n_cons']})"))
        # Backfill: if a band can't fill its quota, try next band's pool.
        for c in by_band[band][taken:]:
            diag.append((band, c["sha"][:12], c["elapsed_ms"],
                         f"available (residual={c['residual']:.3f})"))

    # If any band underfilled, top off from remaining bands by residual.
    n_target_total = sum(band_targets.values())
    if len(picks) < n_target_total:
        leftover = [c for band in ("large", "mid", "small")
                    for c in by_band[band] if c["sha"] not in used]
        leftover.sort(key=lambda c: -c["residual"])
        for c in leftover:
            if len(picks) >= n_target_total:
                break
            d = rows_by_sha[c["sha"]]
            picks.append(d)
            used.add(c["sha"])
            diag.append(("backfill", c["sha"][:12], c["elapsed_ms"],
                         f"backfill (residual={c['residual']:.3f})"))

    n_small = sum(1 for b, *_ in diag if b == "small" and "pick" in _[-1])
    n_mid = sum(1 for b, *_ in diag if b == "mid" and "pick" in _[-1])
    n_large = sum(1 for b, *_ in diag if b == "large" and "pick" in _[-1])
    print(f"outliers stage3: from {csv_path.relative_to(_BENCH.parent)}, "
          f"stratified pick (cap={STAGE3_MAX_BASELINE_MS}ms) "
          f"→ {len(picks)} total "
          f"[small={n_small}/{STAGE3_PICK_SMALL} "
          f"mid={n_mid}/{STAGE3_PICK_MID} "
          f"large={n_large}/{STAGE3_PICK_LARGE}]")
    for band, sha12, ms, note in diag:
        print(f"  [{band:<8}] {sha12}  {int(ms):>10}ms  {note}")

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

    # Stage3: stratified outlier pick from Statistics/outliers_top.csv
    # (small/mid/large by elapsed_ms, top-residual within each band). cap =
    # STAGE3_MAX_BASELINE_MS (higher than stage1/2/4's MAX_BASELINE_MS so
    # genuinely slow outliers enter the tune set). Fallback to slow-cluster
    # center pick if csv missing.
    rows_by_sha = {_id_key(d): d for d in rows}
    outliers_csv = _find_outliers_csv()
    s3_outliers, csv_all = _pick_outliers(rows_by_sha, outliers_csv) \
        if outliers_csv else (None, [])
    if s3_outliers:
        s3 = s3_outliers
        stage3_criteria = (f"stratified outliers from {outliers_csv.name} "
                           f"(small≤{STAGE3_SMALL_MS}ms / "
                           f"mid<{STAGE3_MID_MS}ms / "
                           f"large≤{STAGE3_MAX_BASELINE_MS}ms; "
                           f"top-residual within band; "
                           f"target {STAGE3_PICK_SMALL}+{STAGE3_PICK_MID}+"
                           f"{STAGE3_PICK_LARGE})")
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
