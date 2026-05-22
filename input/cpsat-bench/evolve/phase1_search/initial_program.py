"""
Phase 1: tune CP-SAT search / subsolver knobs.

Targeted namespace (LLM may add/remove/modify keys in the EVOLVE-BLOCK):
  extra_subsolvers, ignore_subsolvers, interleave_search,
  use_feasibility_jump, use_feasibility_pump

Other params stay at BASELINE.
This phase pins num_search_workers=1 (single-worker search) so other knobs
are evaluated without multi-thread / multi-subsolver noise. Phase 3 raises
the worker count to explore subsolver-mix effects.

Do NOT modify locked keys (see PHASE_LOCKED below + baseline_params.LOCKED):
  random_seed, num_search_workers
Invalid solver keys cause evaluator to return 0 and surface the offending key.
"""
import pathlib
import sys

_SHARED = pathlib.Path(__file__).resolve().parent.parent / "shared"
sys.path.insert(0, str(_SHARED))

from baseline_params import BASELINE  # noqa: E402


PHASE_LOCKED = {
    "random_seed": 0,
    "num_search_workers": 1,
}


# EVOLVE-BLOCK-START
SEARCH_OVERRIDES = {
    "extra_subsolvers": ["default_lp", "no_lp"],
    "ignore_subsolvers": ["max_lp"],
    "interleave_search": True,
    "use_feasibility_jump": False,
    "use_feasibility_pump": False,
}
# EVOLVE-BLOCK-END


def get_params():
    p = dict(BASELINE)
    p.update(SEARCH_OVERRIDES)
    p.update(PHASE_LOCKED)  # re-enforce phase lock last
    return p


def get_phase_overrides():
    """Used by extract_best.py — returns ONLY this phase's evolved dict."""
    return dict(SEARCH_OVERRIDES)
