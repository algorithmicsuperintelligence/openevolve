"""
Phase 4: unified refinement.

EVOLVE-BLOCK below is auto-materialized by prepare_phase_unified.py from
the union of phase{1,2,3}_best.json winners. LLM may then tune all keys
jointly.

num_search_workers stays at PHASE4_WORKERS (= phase3 setting) so unified
tuning happens at the same parallel-search scale as phase3.

Do NOT modify locked keys (random_seed, num_search_workers).
"""
import pathlib
import sys

_SHARED = pathlib.Path(__file__).resolve().parent.parent / "shared"
sys.path.insert(0, str(_SHARED))

from baseline_params import BASELINE  # noqa: E402


PHASE4_WORKERS = 8

PHASE_LOCKED = {
    "random_seed": 0,
    "num_search_workers": PHASE4_WORKERS,
}


# EVOLVE-BLOCK-START
UNIFIED_OVERRIDES = {}
# EVOLVE-BLOCK-END


def get_params():
    p = dict(BASELINE)
    p.update(UNIFIED_OVERRIDES)
    p.update(PHASE_LOCKED)
    return p


def get_phase_overrides():
    return dict(UNIFIED_OVERRIDES)
