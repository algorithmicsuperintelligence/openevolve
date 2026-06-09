"""
z3-bench solver hooks. Consumed by every _lib module via
bench_paths.load_adapter().
"""

SOLVER_NAME = "z3"

PROBLEM_FILE_FIELD = "smt2_filename"
STATUS_FIELD = "z3_status"                  # {"result", "elapsed_ms", "objective_value"}
STATS_FIELD = None                          # baseline carries no separate stats block
FEATURES_FIELD = "features"
# Optimize mode: z3.Optimize derives one objective from the assert-soft
# constraints. The reboot baselines were "Skipped" (objective_value=0 placeholder),
# so the real baseline objective is re-measured by _lib.rebaseline and overrides
# this field per-problem (evaluator_core: lo["objective"] wins when matches_raw).
OBJECTIVE_FIELD = "objective_value"

DECISIVE_RESULTS = ("Sat", "Unsat")
DECIDED_RESULTS  = ("Sat", "Unsat")

KEY_STATS = ("decisions", "propagations", "conflicts", "rlimit count")

STATS_WEIGHTS = {
    "conflicts": 2.0,
    "decisions": 1.5,
    "propagations": 0.5,
}

# Cost mode: combined = geomean(cost_ratio^cw * time_ratio) * solved_rate^2 * eff^sw.
# On this reboot workload every problem's optimum is 0, so cost_ratio is ~1 and acts
# as a correctness guard (a variant that returns a worse, non-zero objective is
# penalized); the optimization signal comes from time_ratio on deterministic_time
# (rlimit count), emitted by _solve_worker.py. See config.yaml evaluation.score_mode.
SCORE_MODE = "cost"

WORKERS_KEY = None  # z3 single-threaded in this bench


def get_problem_size(features):
    """Reboot z3-optimize problems carry num_hard_constraints as the dominant
    size signal (2.5k–41k hard / 42–392 soft across the 89-instance workload)."""
    return int((features or {}).get("num_hard_constraints") or 0)
