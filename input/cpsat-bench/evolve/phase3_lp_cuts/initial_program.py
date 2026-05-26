"""
Phase 3: tune CP-SAT LP / cuts / MIP-bridge AND subsolver-mix knobs.

Reference: shared/cpsat_params_reference.md  (sections "LP/cuts" lines
~164-198, "MIP bridge" ~200-214, "parallel/workers" ~10-32.)

Worker count is RAISED to PHASE3_WORKERS (default 8) for this phase: many
subsolvers (max_lp, no_lp, core, quick_restart, reduced_costs, lb_tree_search,
probing_search, …) only activate when num_search_workers is large enough.
Phase 3's job is to find the best subsolver combination + LP/cuts tuning that
works at parallel-search scale.

Three evolution surfaces (see EVOLVE-BLOCK):
  GLOBAL_OVERRIDES  — applied to every problem
  SIZE_BUCKETS      — applied conditionally on `num_constraints`.
                      LARGE problems (≥150k constraints) often suffer from
                      LP iteration explosion; smaller max_num_cuts, lower
                      cut_level, dropping `max_lp` from subsolvers can pay
                      off here. SMALL problems can afford richer LP work.
  STAGE3_OVERRIDES  — applied ONLY when stage == "stage3" AND outlier.
                      Outliers spend ~30x more LP iterations than baseline
                      median (see Statistics/outliers_report.txt). Use this
                      to ship outlier-only LP/cut tuning.

Targeted namespace:
  max_num_cuts, cut_level, linearization_level,
  mip_max_bound, mip_var_scaling, mip_check_precision, mip_drop_tolerance,
  extra_subsolvers, ignore_subsolvers, diversify_lns_params, repair_hint,
  add_*_cuts toggles, root_lp_iterations, exploit_*_lp_solution.

Inherits phase1+phase2 winners (+ their size_buckets / stage3_overrides if
present). num_search_workers is re-pinned to PHASE3_WORKERS at the end of
get_params() — phases 1/2 ran at workers=1 so their wins still need to be
re-validated at workers=PHASE3_WORKERS here.

Do NOT modify locked keys (random_seed, num_search_workers).
"""
import json
import pathlib
import sys

_SHARED = pathlib.Path(__file__).resolve().parent.parent / "shared"
sys.path.insert(0, str(_SHARED))

from baseline_params import BASELINE  # noqa: E402


PHASE3_WORKERS = 8

PHASE_LOCKED = {
    "random_seed": 0,
    "num_search_workers": PHASE3_WORKERS,
}


def _load_prev_dict(name):
    p = _SHARED / name
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _load_prev_buckets(name):
    p = _SHARED / name
    if not p.exists():
        return None
    raw = json.loads(p.read_text())
    return [(float("inf") if u is None else u, override) for u, override in raw]


_PHASE1 = _load_prev_dict("phase1_best.json")
_PHASE2 = _load_prev_dict("phase2_best.json")
_PHASE1_BUCKETS = _load_prev_buckets("phase1_buckets.json")
_PHASE2_BUCKETS = _load_prev_buckets("phase2_buckets.json")
_PHASE1_STAGE3 = _load_prev_dict("phase1_stage3.json")
_PHASE2_STAGE3 = _load_prev_dict("phase2_stage3.json")


# EVOLVE-BLOCK-START
GLOBAL_OVERRIDES = {
    "max_num_cuts": 3000,
    "cut_level": 1,
    "mip_max_bound": 1e+07,
    "mip_var_scaling": 1,
    "mip_check_precision": 1e-06,
    "mip_drop_tolerance": 1e-07,
}

SIZE_BUCKETS = [
    (50_000,         {}),
    (150_000,        {}),
    (float("inf"),   {}),
]

STAGE3_OVERRIDES = {}
# EVOLVE-BLOCK-END


def _merge_bucket(num_constraints):
    out = {}
    for buckets in (_PHASE1_BUCKETS or [], _PHASE2_BUCKETS or [], SIZE_BUCKETS):
        for upper, override in buckets:
            if num_constraints < upper:
                out.update(override)
                break
    return out


def get_params(problem=None, stage=None):
    p = dict(BASELINE)
    p.update(_PHASE1)
    p.update(_PHASE2)
    p.update(GLOBAL_OVERRIDES)
    if problem is not None:
        p.update(_merge_bucket(int(problem.get("num_constraints") or 0)))
        if stage == "stage3" and problem.get("is_outlier"):
            p.update(_PHASE1_STAGE3)
            p.update(_PHASE2_STAGE3)
            p.update(STAGE3_OVERRIDES)
    p.update(PHASE_LOCKED)  # pin workers=PHASE3_WORKERS for this phase
    return p


def get_phase_overrides():
    return dict(GLOBAL_OVERRIDES)


def get_phase_size_buckets():
    return [(u, dict(d)) for u, d in SIZE_BUCKETS]


def get_phase_stage3_overrides():
    return dict(STAGE3_OVERRIDES)
