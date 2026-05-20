"""
Phase 1: tune CP-SAT search / subsolver knobs.

Targeted namespace (LLM may add/remove/modify keys in the EVOLVE-BLOCK):
  extra_subsolvers, ignore_subsolvers, interleave_search,
  use_feasibility_jump, use_feasibility_pump

Other params stay at BASELINE.
Do NOT modify locked keys (see baseline_params.LOCKED):
  random_seed, num_search_workers, timeout_sec
Invalid solver keys cause evaluator to return 0 and surface the offending key.
"""
import pathlib
import sys

_SHARED = pathlib.Path(__file__).resolve().parent.parent / "shared"
sys.path.insert(0, str(_SHARED))

from baseline_params import BASELINE  # noqa: E402


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
    return p


def get_phase_overrides():
    """Used by extract_best.py — returns ONLY this phase's evolved dict."""
    return dict(SEARCH_OVERRIDES)
