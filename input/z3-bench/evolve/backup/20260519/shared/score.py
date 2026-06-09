"""
Scoring: geomean(speedup) * solved_rate^2 * efficiency^STATS_WEIGHT.

- match baseline result: speedup = baseline_ms / elapsed_ms
- mismatch (regression / unknown / timeout): contributes 1e-6 to geomean
- solved_rate squared to strongly gate on correctness
- efficiency = geomean over {decisions, propagations, conflicts, restarts}
  of (baseline_stat / variant_stat), only on solved problems with baseline
  stats present. Lower solver work vs baseline -> efficiency > 1.
  Folded multiplicatively via STATS_WEIGHT exponent (default 0 = disabled,
  preserves prior scoring). Set env OPENEVOLVE_STATS_WEIGHT=0.25 to enable.

Why stats matter: identical elapsed_ms with far fewer conflicts/decisions is
a sturdier improvement (less variance across machines / problems) than a raw
wall-clock win, and runtime alone can hide regressions where Z3 happens to
hit a fast path on the stage1 sample.
"""
import math
import os

_STATS_KEYS = ("decisions", "propagations", "conflicts", "mk clause")


def _efficiency(per_problem):
    """Geomean of baseline/variant ratio across stat keys, solved problems only.

    Returns (efficiency, num_pairs). efficiency=1.0 if no usable pairs (no
    baseline stats yet, or no solved problems) so the multiplier is a no-op.
    """
    ratios = []
    for p in per_problem:
        if p["result"] != p["baseline_result"]:
            continue
        bs = p.get("baseline_stats") or {}
        vs = p.get("stats") or {}
        for k in _STATS_KEYS:
            b = bs.get(k)
            v = vs.get(k)
            if b is None or v is None:
                continue
            # +1 smoothing avoids div-by-zero and absurd ratios for tiny counts
            ratios.append((float(b) + 1.0) / (float(v) + 1.0))
    if not ratios:
        return 1.0, 0
    log_sum = sum(math.log(r) for r in ratios)
    return math.exp(log_sum / len(ratios)), len(ratios)


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
    solved = 0
    regressions = 0
    for p in per_problem:
        baseline_decided = p["baseline_result"] in ("Sat", "Unsat")
        match = p["result"] == p["baseline_result"]
        if match:
            solved += 1
            sp = p["baseline_ms"] / max(p["elapsed_ms"], 1)
            speedups.append(sp)
        else:
            speedups.append(1e-6)
            if baseline_decided and p["result"] in ("Sat", "Unsat"):
                regressions += 1

    log_sum = sum(math.log(s) for s in speedups)
    geomean = math.exp(log_sum / len(speedups))
    solved_rate = solved / n

    efficiency, eff_pairs = _efficiency(per_problem)
    try:
        stats_weight = float(os.environ.get("OPENEVOLVE_STATS_WEIGHT", "0"))
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
        "efficiency_pairs": int(eff_pairs),
        "stats_weight": float(stats_weight),
    }
