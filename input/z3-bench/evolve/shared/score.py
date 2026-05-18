"""
Scoring: geomean(speedup) * solved_rate^2.
- match baseline result: speedup = baseline_ms / elapsed_ms
- mismatch (regression / unknown / timeout): contributes 1e-6 to geomean
- solved_rate squared to strongly gate on correctness
"""
import math


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
    combined = geomean * (solved_rate**2)

    return {
        "combined_score": float(combined),
        "geomean_speedup": float(geomean),
        "solved_rate": float(solved_rate),
        "regressions": int(regressions),
        "solved": int(solved),
        "total": int(n),
    }
