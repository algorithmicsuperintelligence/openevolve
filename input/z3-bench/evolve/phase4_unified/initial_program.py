"""
Phase 4: unified fine-tuning — interaction effects.

The EVOLVE-BLOCK below is a literal dict materialized from phase1/2/3 best
overrides by prepare_phase4.py. Run that script BEFORE invoking run_phase.sh 4:

    python ../prepare_phase4.py

The LLM evolves UNIFIED_OVERRIDES directly. It may modify, remove, or add
keys from the Z3 4.13.x parameter space. Smaller iteration budget — local
refinement, not exploration.
"""
import pathlib
import sys

_SHARED = pathlib.Path(__file__).resolve().parent.parent / "shared"
sys.path.insert(0, str(_SHARED))

from baseline_params import BASELINE  # noqa: E402


# EVOLVE-BLOCK-START
# Seeded with baseline only. Run prepare_phase4.py after phases 1-3 finish
# to replace this dict with the union of their winners.
UNIFIED_OVERRIDES = {}
# EVOLVE-BLOCK-END


def get_params():
    p = dict(BASELINE)
    p.update(UNIFIED_OVERRIDES)
    return p


def get_phase_overrides():
    return dict(UNIFIED_OVERRIDES)
