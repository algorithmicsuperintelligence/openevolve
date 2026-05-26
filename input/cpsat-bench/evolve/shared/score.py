"""
Scoring for cpsat-bench.

Score mode: cost (minimize CP-SAT objective).

    combined = geomean(cost_ratio^COST_W * time_ratio) * solved_rate^2 * efficiency^STATS_WEIGHT

    time_ratio (preferred): baseline_dtime / variant_dtime
                            CP-SAT's deterministic_time is hardware-independent,
                            so per-iteration noise across machines / system load
                            stays out of the score. ~order-of-magnitude same as
                            wall_time but stable across runs.
    time_ratio (fallback) : baseline_ms / variant_ms (wall)
                            used when either dtime is missing (legacy local
                            baselines without dtime in stats, or worker not
                            emitting it). Diagnostic geomean_wall_speedup is
                            always reported alongside.

    - both baseline+variant decisive (OPTIMAL or FEASIBLE) with objective values:
        cost_ratio = (baseline_obj + EPS) / (variant_obj + EPS)   [minimize]
    - status mismatch (e.g. variant UNKNOWN where baseline OPTIMAL):  1e-6
    - missing baseline_objective (rebaseline not yet run) falls back to time_ratio only

Note: all 85 baseline runs in this dataset reach OPTIMAL, so when variant
ALSO reaches OPTIMAL, cost_ratio collapses to 1.0 and score reduces to
geomean(time_ratio). Cost mode mainly catches variants that bail to FEASIBLE
with a worse objective.

Efficiency factor: cross-problem geomean of per-problem weighted geomean over
CP-SAT counters (num_conflicts, num_branches). Each ratio
(baseline+1)/(variant+1) clipped to [0.1, 10]. Lower variant work => efficiency > 1.

Env overrides:
  OPENEVOLVE_STATS_WEIGHT   exponent on efficiency, default 0.333, 0 disables
  OPENEVOLVE_COST_WEIGHT    exponent on cost_ratio, default 1.0, 0 disables cost factor
  OPENEVOLVE_TIME_METRIC    "dtime" (default) | "wall" — force wall ratio
"""
import math
import os

_SCORE_MODE = "cost"

_STATS_WEIGHTS = {
    "num_conflicts": 2.0,
    "num_branches": 1.5,
}
_RATIO_CLIP_LO = 0.1
_RATIO_CLIP_HI = 10.0

_COST_RATIO_CLIP_LO = 0.01
_COST_RATIO_CLIP_HI = 100.0
_COST_EPS = 1e-9

_DECISIVE = ("OPTIMAL", "FEASIBLE")


def _efficiency(per_problem):
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
            r = (float(b) + 1.0) / (float(v) + 1.0)
            r = max(_RATIO_CLIP_LO, min(_RATIO_CLIP_HI, r))
            log_sum += w * math.log(r)
            w_sum += w
        if w_sum > 0:
            per_prob_effs.append(math.exp(log_sum / w_sum))
    if not per_prob_effs:
        return 1.0, 0
    log_sum = sum(math.log(e) for e in per_prob_effs)
    return math.exp(log_sum / len(per_prob_effs)), len(per_prob_effs)


def _time_ratio(p, metric):
    """Compute time_ratio. metric ∈ {"dtime", "wall"}.
    Returns (ratio, source) where source describes which clock was used.
    Falls back to wall when dtime missing on either side."""
    bs = p.get("baseline_stats") or {}
    vs = p.get("stats") or {}
    if metric == "dtime":
        b_dt = bs.get("deterministic_time")
        v_dt = vs.get("deterministic_time")
        if b_dt and v_dt and b_dt > 0 and v_dt > 0:
            return float(b_dt) / float(v_dt), "dtime"
    # fallback / forced wall
    return p["baseline_ms"] / max(p["elapsed_ms"], 1), "wall"


def _wall_ratio(p):
    return p["baseline_ms"] / max(p["elapsed_ms"], 1)


def _score_cost(per_problem):
    """
    Cost mode scoring rules:
      - baseline NOT decisive → uncomparable; skip from geomean entirely.
        (Baseline never reached OPTIMAL/FEASIBLE within timeout → variant has
         no target to beat. Counting it as 1e-6 unfairly tanks the geomean.)
      - baseline decisive + variant decisive → ratio = cost_ratio^W * time_ratio
        (time_ratio is deterministic_time-based when both sides have it, else wall)
      - baseline decisive + variant NOT decisive → 1e-6 + regression++ (real loss).

    Returns: (geomean, geomean_wall, solved_rate, solved, regressions,
              comparable, dtime_used, dtime_fallback)
      comparable    = problems where baseline was decisive (counted in geomean).
      dtime_used    = comparable problems where dtime ratio applied.
      dtime_fallback= comparable problems forced to wall ratio (dtime missing).
    """
    cost_weight = float(os.environ.get("OPENEVOLVE_COST_WEIGHT", "1.0"))
    cost_weight = max(0.0, min(cost_weight, 2.0))
    time_metric = os.environ.get("OPENEVOLVE_TIME_METRIC", "dtime").lower()
    if time_metric not in ("dtime", "wall"):
        time_metric = "dtime"

    ratios = []
    wall_ratios = []
    solved = 0
    regressions = 0
    comparable = 0
    dtime_used = 0
    dtime_fallback = 0
    for p in per_problem:
        b_ok = p["baseline_result"] in _DECISIVE
        if not b_ok:
            continue        # uncomparable
        comparable += 1
        v_ok = p["result"] in _DECISIVE
        b_cost = p.get("baseline_objective")
        v_cost = p.get("objective")
        time_r, src = _time_ratio(p, time_metric)
        wall_r = _wall_ratio(p)
        if time_metric == "dtime":
            if src == "dtime":
                dtime_used += 1
            else:
                dtime_fallback += 1
        if v_ok:
            solved += 1
            if b_cost is not None and v_cost is not None:
                cost_r = (float(b_cost) + _COST_EPS) / (float(v_cost) + _COST_EPS)
                cost_r = max(_COST_RATIO_CLIP_LO, min(_COST_RATIO_CLIP_HI, cost_r))
                ratios.append((cost_r ** cost_weight) * time_r)
                wall_ratios.append((cost_r ** cost_weight) * wall_r)
            else:
                ratios.append(time_r)
                wall_ratios.append(wall_r)
        else:
            ratios.append(1e-6)
            wall_ratios.append(1e-6)
            regressions += 1
    if not ratios:
        return 1.0, 1.0, 0.0, 0, 0, 0, 0, 0
    geomean = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    geomean_wall = math.exp(sum(math.log(r) for r in wall_ratios)
                            / len(wall_ratios))
    solved_rate = solved / comparable
    return (geomean, geomean_wall, solved_rate, solved, regressions,
            comparable, dtime_used, dtime_fallback)


def score(per_problem):
    n = len(per_problem)
    if n == 0:
        return {
            "combined_score": 0.0,
            "geomean_speedup": 0.0,
            "geomean_wall_speedup": 0.0,
            "solved_rate": 0.0,
            "regressions": 0,
            "solved": 0,
            "comparable": 0,
            "total": 0,
            "uncomparable": 0,
            "efficiency": 1.0,
            "efficiency_pairs": 0,
            "stats_weight": 0.0,
            "dtime_used": 0,
            "dtime_fallback": 0,
        }

    (geomean, geomean_wall, solved_rate, solved, regressions,
     comparable, dtime_used, dtime_fallback) = _score_cost(per_problem)

    efficiency, eff_pairs = _efficiency(per_problem)
    try:
        stats_weight = float(os.environ.get("OPENEVOLVE_STATS_WEIGHT", "0.333"))
    except ValueError:
        stats_weight = 0.0
    stats_weight = max(0.0, min(stats_weight, 2.0))

    combined = geomean * (solved_rate ** 2.0) * (efficiency ** stats_weight)

    return {
        "combined_score": float(combined),
        "geomean_speedup": float(geomean),  # primary (dtime when available)
        "geomean_wall_speedup": float(geomean_wall),  # diagnostic (wall)
        "solved_rate": float(solved_rate),
        "regressions": int(regressions),
        "solved": int(solved),
        "comparable": int(comparable),
        "total": int(n),
        "uncomparable": int(n - comparable),
        "efficiency": float(efficiency),
        "efficiency_pairs": int(eff_pairs),
        "stats_weight": float(stats_weight),
        "dtime_used": int(dtime_used),
        "dtime_fallback": int(dtime_fallback),
    }
