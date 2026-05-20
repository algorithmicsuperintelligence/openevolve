"""
Phase 3: tune CP-SAT LP / cuts / MIP-bridge knobs.

Targeted namespace:
  max_num_cuts, cut_level,
  mip_max_bound, mip_var_scaling, mip_check_precision, mip_drop_tolerance
(LLM may explore other linearization_level, *_cuts toggles.)

Inherits phase1+phase2 winners.
Do NOT modify locked keys.
"""
import json
import pathlib
import sys

_SHARED = pathlib.Path(__file__).resolve().parent.parent / "shared"
sys.path.insert(0, str(_SHARED))

from baseline_params import BASELINE  # noqa: E402


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
    return p


def get_phase_overrides():
    return dict(LP_CUTS_OVERRIDES)
