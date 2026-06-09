"""
Unified scoring. Modes:

  speedup (z3 default):
    combined = weighted_geomean(speedup) * solved_rate^2 * efficiency^stats_weight
    speedup  = baseline_ms / variant_ms
    per-problem weight = baseline_ms (long problems dominate)
    regression (baseline decisive, variant mismatch) contributes 1e-6.

  cost (cpsat default):
    combined = unweighted_geomean(cost_ratio^cw * time_ratio) * solved_rate^2 * efficiency^sw
    time_ratio  = baseline_dtime / variant_dtime   (when both sides report
                  deterministic_time; falls back to wall ratio)
    cost_ratio  = (baseline_obj + eps) / (variant_obj + eps)   [minimize]
    non-decisive variant: contributes its real (slow) time_ratio
                          (NOT a 1e-6 sentinel); solved_rate^2 penalizes.
    uncomparable (baseline non-decisive): excluded from geomean entirely.

Efficiency factor: cross-problem geomean of per-problem weighted geomean
over a configurable stat-key set, each ratio (baseline+1)/(variant+1)
clipped to [0.1, 10]. Mirror of the old per-bench score.py logic.

Env overrides honored:
  OPENEVOLVE_STATS_WEIGHT  exponent on efficiency (default 0.333)
  OPENEVOLVE_COST_WEIGHT   exponent on cost_ratio (cost mode only)
  OPENEVOLVE_TIME_METRIC   "dtime" (default) | "wall" — cost mode only
"""
import math
import os


_RATIO_CLIP_LO = 0.1
_RATIO_CLIP_HI = 10.0
_COST_RATIO_CLIP_LO = 0.01
_COST_RATIO_CLIP_HI = 100.0
_COST_EPS = 1e-9


def _efficiency(per_problem, stats_weights):
    per_prob_effs = []
    for p in per_problem:
        if p["result"] != p["baseline_result"]:
            continue
        bs = p.get("baseline_stats") or {}
        vs = p.get("stats") or {}
        log_sum = 0.0
        w_sum = 0.0
        for k, w in stats_weights.items():
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


def _stats_weight_env(default):
    try:
        v = float(os.environ.get("OPENEVOLVE_STATS_WEIGHT", str(default)))
    except ValueError:
        return 0.0
    return max(0.0, min(v, 2.0))


def _score_speedup(per_problem, decided_set):
    """
    z3-style: weighted geomean(speedup) * solved_rate^2.
    Mismatch (decided baseline → wrong variant) = 1e-6 contribution + regression++.
    """
    speedups = []
    weights = []
    solved = 0
    regressions = 0
    for p in per_problem:
        baseline_decided = p["baseline_result"] in decided_set
        match = p["result"] == p["baseline_result"]
        w = max(float(p["baseline_ms"]), 1.0)
        weights.append(w)
        if match:
            solved += 1
            speedups.append(max(p["baseline_ms"], 1) / max(p["elapsed_ms"], 1))
        else:
            speedups.append(1e-6)
            if baseline_decided and p["result"] in decided_set:
                regressions += 1
    if not speedups:
        return 1.0, 1.0, 0.0, 0, 0, 0, 0, 0
    w_total = sum(weights)
    log_sum = sum(w * math.log(s) for s, w in zip(speedups, weights))
    geomean = math.exp(log_sum / w_total)
    solved_rate = solved / len(per_problem)
    return (geomean, geomean, solved_rate, solved, regressions,
            len(per_problem), 0, 0)


def _time_ratio(p, metric):
    bs = p.get("baseline_stats") or {}
    vs = p.get("stats") or {}
    if metric == "dtime":
        b_dt = bs.get("deterministic_time")
        v_dt = vs.get("deterministic_time")
        if b_dt and v_dt and b_dt > 0 and v_dt > 0:
            return float(b_dt) / float(v_dt), "dtime"
    return max(p["baseline_ms"], 1) / max(p["elapsed_ms"], 1), "wall"


def _score_cost(per_problem, decisive_set, time_metric, cost_weight):
    """
    cpsat-style: cost-aware geomean using deterministic_time when available.
    Uncomparable (baseline non-decisive) is excluded from geomean.
    """
    ratios = []
    wall_ratios = []
    solved = 0
    regressions = 0
    comparable = 0
    dtime_used = 0
    dtime_fallback = 0
    for p in per_problem:
        if p["baseline_result"] not in decisive_set:
            continue
        comparable += 1
        v_ok = p["result"] in decisive_set
        b_cost = p.get("baseline_objective")
        v_cost = p.get("objective")
        time_r, src = _time_ratio(p, time_metric)
        wall_r = max(p["baseline_ms"], 1) / max(p["elapsed_ms"], 1)
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
            ratios.append(time_r)
            wall_ratios.append(wall_r)
            regressions += 1
    if not ratios:
        return 1.0, 1.0, 0.0, 0, 0, 0, 0, 0
    geomean = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    geomean_wall = math.exp(sum(math.log(r) for r in wall_ratios) / len(wall_ratios))
    solved_rate = solved / comparable
    return (geomean, geomean_wall, solved_rate, solved, regressions,
            comparable, dtime_used, dtime_fallback)


def score(per_problem, *,
          mode="speedup",
          decisive_results=("Sat", "Unsat"),
          decided_results=None,
          stats_weights=None,
          stats_weight_default=0.333,
          time_metric=None,
          cost_weight=None):
    """
    Unified scoring entry point.

    Args:
        per_problem: list of dicts each with keys
            sha, baseline_result, baseline_ms, baseline_stats, [baseline_objective],
            result, elapsed_ms, stats, [objective]
        mode: "speedup" or "cost"
        decisive_results: tuple of result strings the SOLVER counts as decisive.
            cpsat → ("OPTIMAL","FEASIBLE"); z3 → ("Sat","Unsat").
        decided_results: tuple of results that count as a "definitive answer"
            for regression detection in speedup mode. Defaults to decisive_results.
        stats_weights: {stat_key: weight} for efficiency factor. None → no
            efficiency contribution.
        stats_weight_default: default exponent on efficiency (env override wins).
        time_metric: "dtime"|"wall" — cost mode only. Default → env or "dtime".
        cost_weight: float — cost mode only. Default → env or 1.0.

    Returns metrics dict (always populated):
        combined_score, geomean_speedup, geomean_wall_speedup, solved_rate,
        regressions, solved, comparable, total, uncomparable,
        efficiency, efficiency_pairs, stats_weight,
        dtime_used, dtime_fallback
    """
    n = len(per_problem)
    if stats_weights is None:
        stats_weights = {}
    if decided_results is None:
        decided_results = decisive_results
    decisive_set = set(decisive_results)
    decided_set = set(decided_results)

    if n == 0:
        return {
            "combined_score": 0.0,
            "geomean_speedup": 0.0,
            "geomean_wall_speedup": 0.0,
            "solved_rate": 0.0,
            "regressions": 0, "solved": 0, "comparable": 0, "total": 0,
            "uncomparable": 0,
            "efficiency": 1.0, "efficiency_pairs": 0,
            "stats_weight": 0.0,
            "dtime_used": 0, "dtime_fallback": 0,
        }

    if mode == "cost":
        if time_metric is None:
            time_metric = (os.environ.get("OPENEVOLVE_TIME_METRIC", "dtime").lower())
            if time_metric not in ("dtime", "wall"):
                time_metric = "dtime"
        if cost_weight is None:
            try:
                cost_weight = float(os.environ.get("OPENEVOLVE_COST_WEIGHT", "1.0"))
            except ValueError:
                cost_weight = 1.0
            cost_weight = max(0.0, min(cost_weight, 2.0))
        (geomean, geomean_wall, solved_rate, solved, regressions,
         comparable, dtime_used, dtime_fallback) = _score_cost(
            per_problem, decisive_set, time_metric, cost_weight)
    else:
        (geomean, geomean_wall, solved_rate, solved, regressions,
         comparable, dtime_used, dtime_fallback) = _score_speedup(
            per_problem, decided_set)

    efficiency, eff_pairs = _efficiency(per_problem, stats_weights)
    stats_weight = _stats_weight_env(stats_weight_default)

    combined = geomean * (solved_rate ** 2.0) * (efficiency ** stats_weight)
    return {
        "combined_score": float(combined),
        "geomean_speedup": float(geomean),
        "geomean_wall_speedup": float(geomean_wall),
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
