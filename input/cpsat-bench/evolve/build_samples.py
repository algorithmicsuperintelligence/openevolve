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

# Large-profile sample (selected via OPENEVOLVE_PROFILE=large). Single file
# only — large profile bypasses cascade staging entirely; evaluator dispatches
# every stage entry point to a single outlier-set evaluation. File name keeps
# the stage1_large prefix so legacy callers (and evaluator's path resolver)
# continue to find it.
_STAGE1_LARGE = _HERE / "shared" / "stage1_large_sample.json"

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
# Large profile: single hardest outlier (top residual from outliers_top.csv).
# Tune higher if want to evaluate against more outliers at the cost of
# proportionally longer per-iteration eval time.
STAGE1_LARGE_N = 1
# Global cap for stage1/2/4 sample pool. Anything slower than 2 min skews
# the quintile clustering.
MAX_BASELINE_MS = 120_000
# Stage3 (outlier-only) gets a higher cap so genuinely slow outliers can
# enter the tune set. 25 min ≈ what one stage3 problem can chew through
# under W=8 (raw-data times are W=8-equivalent). Variant timeout =
# baseline_ms * 1.3 in evaluator.
STAGE3_MAX_BASELINE_MS = 1_500_000
# Stage3 target-based pick — outliers in outliers_top.csv span 4 orders of
# magnitude (180ms .. 5.7M ms). Picking by residual alone produces a sample
# with wild runtime variance, which makes per-iteration scoring noisy. Instead
# pick STAGE3_N outliers whose baseline elapsed_ms is closest to
# STAGE3_TARGET_MS, dropping any whose ratio to target exceeds
# STAGE3_TARGET_TOL_RATIO (e.g. target=20000 tol=5.0 → accepts 4000..100000ms).
STAGE3_TARGET_MS = 20_000
STAGE3_TARGET_TOL_RATIO = 5.0
OUTLIER_IQR_K = 3.0

STAGE1_STRATEGY = "center"
STAGE2_STRATEGY = "center"
STAGE3_STRATEGY = "target_ms"   # pick outliers nearest STAGE3_TARGET_MS
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


def _pick_outliers(rows_by_sha, csv_path, all_rows=None, n_pick=None,
                    target_ms=None, tol_ratio=None):
    """Target-ms pick for stage3: prefer outliers_top.csv entries whose
    baseline elapsed_ms is closest to target_ms; if fewer than n_pick survive
    the tol_ratio filter, backfill from the full problems pool (non-outlier
    problems also near target_ms).

    Returns (picks, csv_rows). picks sorted by ascending |elapsed_ms-target_ms|.
    csv_rows is the full csv (for outliers.json metadata).

    Returns (None, []) when csv missing — caller falls back to slow-cluster
    center pick."""
    if csv_path is None:
        print("warning: outliers_top.csv not found — falling back to "
              "slow-cluster center pick for stage3", file=sys.stderr)
        return None, []

    if n_pick is None:
        n_pick = STAGE3_N
    if target_ms is None:
        target_ms = STAGE3_TARGET_MS
    if tol_ratio is None:
        tol_ratio = STAGE3_TARGET_TOL_RATIO

    csv_rows = _load_outliers_top(csv_path)

    def _within_tol(ms):
        if ms <= 0 or ms > STAGE3_MAX_BASELINE_MS:
            return False
        return max(ms / target_ms, target_ms / ms) <= tol_ratio

    # Pass 1: outliers within tol.
    outlier_eligible = [c for c in csv_rows
                        if c["sha"] in rows_by_sha and _within_tol(c["elapsed_ms"])]
    outlier_eligible.sort(key=lambda c: abs(c["elapsed_ms"] - target_ms))
    outlier_pick_csv = outlier_eligible[:n_pick]
    picks = [rows_by_sha[c["sha"]] for c in outlier_pick_csv]
    used_shas = {c["sha"] for c in outlier_pick_csv}

    # Pass 2: backfill from full problems pool near target_ms (non-outliers).
    backfill_picks = []
    if len(picks) < n_pick and all_rows is not None:
        decisive = (d for d in all_rows
                    if _id_key(d) not in used_shas
                    and _within_tol(_runtime_key(d))
                    and _result_key(d) in _DECISIVE_RESULTS)
        backfill_pool = sorted(decisive,
                               key=lambda d: abs(_runtime_key(d) - target_ms))
        need = n_pick - len(picks)
        backfill_picks = backfill_pool[:need]
        picks.extend(backfill_picks)

    print(f"outliers stage3: from {csv_path.relative_to(_BENCH.parent)}, "
          f"target={target_ms}ms tol={tol_ratio}x cap={STAGE3_MAX_BASELINE_MS}ms "
          f"→ {len(picks)}/{n_pick} "
          f"(outliers={len(outlier_pick_csv)}, backfill={len(backfill_picks)})")
    for c in outlier_pick_csv:
        ratio = max(c["elapsed_ms"] / target_ms, target_ms / c["elapsed_ms"])
        print(f"  outlier   {c['sha'][:12]}  {int(c['elapsed_ms']):>10}ms  "
              f"Δ={int(c['elapsed_ms'] - target_ms):+d}ms "
              f"ratio={ratio:.2f}x residual={c['residual']:.3f}")
    for d in backfill_picks:
        ms = _runtime_key(d)
        ratio = max(ms / target_ms, target_ms / ms)
        print(f"  backfill  {_id_key(d)[:12]}  {int(ms):>10}ms  "
              f"Δ={int(ms - target_ms):+d}ms ratio={ratio:.2f}x")
    if len(picks) < n_pick:
        print(f"  warning: only {len(picks)} candidates within tol={tol_ratio}x "
              f"of {target_ms}ms — widen STAGE3_TARGET_TOL_RATIO or change "
              f"STAGE3_TARGET_MS to fill stage3")

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

    # Stage3: target-ms outlier pick from Statistics/outliers_top.csv —
    # outliers whose baseline_ms is closest to STAGE3_TARGET_MS (within
    # STAGE3_TARGET_TOL_RATIO). Keeps stage3 runtime variance low so cascade
    # score is dominated by signal, not noise from a 1500ms vs 1500000ms split.
    # Fallback to slow-cluster center pick if csv missing.
    rows_by_sha = {_id_key(d): d for d in rows}
    outliers_csv = _find_outliers_csv()
    s3_outliers, csv_all = _pick_outliers(rows_by_sha, outliers_csv,
                                          all_rows=rows) \
        if outliers_csv else (None, [])
    if s3_outliers:
        s3 = s3_outliers
        stage3_criteria = (
            f"outliers from {outliers_csv.name} nearest "
            f"{STAGE3_TARGET_MS}ms (tol={STAGE3_TARGET_TOL_RATIO}x, "
            f"cap={STAGE3_MAX_BASELINE_MS}ms; target {STAGE3_N})"
        )
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

    # ---- large profile sample (OPENEVOLVE_PROFILE=large) ----
    # Single sample file with STAGE1_LARGE_N top-residual outliers (default 1).
    # evaluator dispatches every cascade stage entry point to one outlier-set
    # evaluation, so no stage2/3/4 split is needed.
    if csv_all:
        eligible = [c for c in csv_all if c["sha"] in rows_by_sha]
        # csv_all already sorted by descending residual in _load_outliers_top.
        large_picks = eligible[:STAGE1_LARGE_N]
        large_rows = [rows_by_sha[c["sha"]] for c in large_picks]
    else:
        large_picks = []
        large_rows = []
    _write_sample(
        _STAGE1_LARGE, large_rows, "stage1_large",
        f"top-{STAGE1_LARGE_N} residual outlier(s) from outliers_top.csv "
        f"(W=8 tuning set)",
    )

    # Clean up stale empty stage{2,3,4}_large files from older builds — they
    # would be picked up by evaluator's path resolver and produce stale empty
    # pass-throughs, masking real outlier scores.
    for stale in (
        _HERE / "shared" / "stage2_large_sample.json",
        _HERE / "shared" / "stage3_large_sample.json",
        _HERE / "shared" / "stage4_large_sample.json",
    ):
        if stale.exists():
            stale.unlink()
            print(f"removed stale {stale.relative_to(_BENCH.parent)}")

    print(f"\nstage1_large ({len(large_rows)} outliers):")
    for d in large_rows:
        print(f"  {_id_key(d)[:12]}  "
              f"{str(_result_key(d)):<10}  "
              f"{int(_runtime_key(d)):>7}ms")


if __name__ == "__main__":
    main()
