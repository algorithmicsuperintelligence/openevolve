"""
Phase 4: unified refinement.

EVOLVE-BLOCK below is auto-materialized by prepare_phase_unified.py from
the union of phase{1,2,3}_best.json winners. LLM may then tune all keys
jointly.

Do NOT modify locked keys (random_seed, num_search_workers, timeout_sec).
"""
import pathlib
import sys

_SHARED = pathlib.Path(__file__).resolve().parent.parent / "shared"
sys.path.insert(0, str(_SHARED))

from baseline_params import BASELINE  # noqa: E402


# EVOLVE-BLOCK-START
UNIFIED_OVERRIDES = {}
# EVOLVE-BLOCK-END


def get_params():
    p = dict(BASELINE)
    p.update(UNIFIED_OVERRIDES)
    return p


def get_phase_overrides():
    return dict(UNIFIED_OVERRIDES)
