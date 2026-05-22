"""
Phase 3: tune CP-SAT LP / cuts / MIP-bridge AND subsolver-mix knobs.

Worker count is RAISED to PHASE3_WORKERS (default 8) for this phase: many
subsolvers (max_lp, no_lp, core, quick_restart, reduced_costs, lb_tree_search,
probing_search, …) only activate when num_search_workers is large enough.
Phase 3's job is to find the best subsolver combination + LP/cuts tuning that
works at parallel-search scale.

Targeted namespace:
  max_num_cuts, cut_level, linearization_level,
  mip_max_bound, mip_var_scaling, mip_check_precision, mip_drop_tolerance,
  extra_subsolvers, ignore_subsolvers, diversify_lns_params, repair_hint
(LLM may explore other *_cuts toggles and subsolver list contents.)

Inherits phase1+phase2 winners, but num_search_workers is re-pinned to
PHASE3_WORKERS at the end of get_params() — phases 1/2 ran at workers=1 so
their wins still need to be re-validated at workers=PHASE3_WORKERS here.

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


def _load_prev(name):
    p = _SHARED / name
    if p.exists():
        return json.loads(p.read_text())
    return {}


_PHASE1 = _load_prev("phase1_best.json")
_PHASE2 = _load_prev("phase2_best.json")


# EVOLVE-BLOCK-START
LP_CUTS_OVERRIDES = {
    "max_num_cuts": 3000,
    "cut_level": 1,
    "mip_max_bound": 1e+07,
    "mip_var_scaling": 1,
    "mip_check_precision": 1e-06,
    "mip_drop_tolerance": 1e-07,
}
# EVOLVE-BLOCK-END


def get_params():
    p = dict(BASELINE)
    p.update(_PHASE1)
    p.update(_PHASE2)
    p.update(LP_CUTS_OVERRIDES)
    p.update(PHASE_LOCKED)  # pin workers=PHASE3_WORKERS for this phase
    return p


def get_phase_overrides():
    return dict(LP_CUTS_OVERRIDES)
