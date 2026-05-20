"""
Scoring: weighted_geomean(speedup) * solved_rate^2 * efficiency^STATS_WEIGHT.

- match baseline result: speedup = baseline_ms / elapsed_ms
- mismatch (regression / unknown / timeout): contributes 1e-6 to geomean
- per-problem weight = baseline_ms so absolute time savings on long-running
  problems dominate; small-runtime wins barely move the needle, and a
  regression on a slow problem is penalized far more than on a fast one
- solved_rate squared to strongly gate on correctness
- efficiency = cross-problem geomean of per-problem weighted geomean over
  SMT solver stats {conflicts, decisions, propagations}, with each ratio
  (baseline_stat+1)/(variant_stat+1) clipped to [0.1, 10] to bound outliers.
  Only solved problems with baseline stats present contribute. Lower solver
  work vs baseline -> efficiency > 1. Folded multiplicatively via STATS_WEIGHT
  exponent (default 0.333 -> stats / (speedup + stats) ~= 25% in log space;
  recommended band 0.2-0.5). Override via env OPENEVOLVE_STATS_WEIGHT
  (0 disables).

Per-key weights reflect SMT solver signal quality:
- conflicts (2.0): CDCL backtracks/learned clauses. Strongest predictor of
  search difficulty; lower vs baseline = smarter navigation; robust to
  hardware noise.
- decisions (1.5): branching choices. Tracks search-tree size and branching
  heuristic quality independent of conflicts.
- propagations (0.5): BCP + theory propagation. High variance; can mean
  early pruning (good) or theory thrashing (bad). Tiebreaker only.
- 'mk clause' / 'restarts' intentionally excluded: learning is a feature
  (fewer learned clauses != better), restart count alone no work signal.

Two-level aggregate (per-problem then cross-problem) gives each benchmark
an equal vote regardless of how many stat keys it reports.

Why stats matter: identical elapsed_ms with far fewer conflicts/decisions is
a sturdier improvement (less variance across machines / problems) than a raw
wall-clock win, and runtime alone can hide regressions where Z3 happens to
hit a fast path on the stage1 sample.
"""
import math
import os

_STATS_WEIGHTS = {"conflicts": 2.0, "decisions": 1.5, "propagations": 0.5}
_RATIO_CLIP_LO = 0.1
_RATIO_CLIP_HI = 10.0


def _efficiency(per_problem):
    """Cross-problem geomean of per-problem weighted geomean of clipped ratios.

    Returns (efficiency, num_problems). efficiency=1.0 if no usable problems
    (no baseline stats yet, or no solved problems) so the multiplier is a no-op.
    """
    per_prob_effs = []
    for p in per_problem:
        if p["result"] != p["baseline_result"]:
            continue
        bs = p.get("baseline_stats") or {}
        vs = p.get("stats") or {}
        log_sum = 0.0
        w_sum = 0.0
        for k, w in _STATS_WEIGHTS.items():
            b = bs.get(k)
            v = vs.get(k)
            if b is None or v is None:
                continue
            # +1 smoothing avoids div-by-zero and absurd ratios for tiny counts
            r = (float(b) + 1.0) / (float(v) + 1.0)
            # Clip so one runaway problem/key can't dominate the geomean
            r = max(_RATIO_CLIP_LO, min(_RATIO_CLIP_HI, r))
            log_sum += w * math.log(r)
            w_sum += w
        if w_sum > 0:
            per_prob_effs.append(math.exp(log_sum / w_sum))
    if not per_prob_effs:
        return 1.0, 0
    log_sum = sum(math.log(e) for e in per_prob_effs)
    return math.exp(log_sum / len(per_prob_effs)), len(per_prob_effs)


def score(per_problem):
    n = len(per_problem)
    if n == 0:
        return {
            "combined_score": 0.0,
            "geomean_speedup": 0.0,
            "solved_rate": 0.0,
            "regressions": 0,
            "solved": 0,
            "total": 0,
            "efficiency": 1.0,
            "efficiency_pairs": 0,
            "stats_weight": 0.0,
        }

    speedups = []
    weights = []
    solved = 0
    regressions = 0
    for p in per_problem:
        baseline_decided = p["baseline_result"] in ("Sat", "Unsat")
        match = p["result"] == p["baseline_result"]
        w = max(float(p["baseline_ms"]), 1.0)
        weights.append(w)
        if match:
            solved += 1
            sp = p["baseline_ms"] / max(p["elapsed_ms"], 1)
            speedups.append(sp)
        else:
            speedups.append(1e-6)
            if baseline_decided and p["result"] in ("Sat", "Unsat"):
                regressions += 1

    w_total = sum(weights)
    log_sum = sum(w * math.log(s) for s, w in zip(speedups, weights))
    geomean = math.exp(log_sum / w_total)
    solved_rate = solved / n

    efficiency, eff_problems = _efficiency(per_problem)
    try:
        stats_weight = float(os.environ.get("OPENEVOLVE_STATS_WEIGHT", "0.333"))
    except ValueError:
        stats_weight = 0.0
    # Clamp to a sensible band so a runaway env var can't dominate score.
    stats_weight = max(0.0, min(stats_weight, 2.0))

    combined = geomean * (solved_rate**2) * (efficiency**stats_weight)

    return {
        "combined_score": float(combined),
        "geomean_speedup": float(geomean),
        "solved_rate": float(solved_rate),
        "regressions": int(regressions),
        "solved": int(solved),
        "total": int(n),
        "efficiency": float(efficiency),
        "efficiency_pairs": int(eff_problems),
        "stats_weight": float(stats_weight),
    }
