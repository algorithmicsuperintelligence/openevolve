"""
Phase 2: tune CP-SAT presolve / probing knobs.

Targeted namespace:
  cp_model_probing_level, cp_model_presolve, symmetry_level,
  presolve_use_bva, presolve_bve_threshold
(LLM may explore other presolve_* / probing_* keys.)

Inherits phase1 winners (loaded from shared/phase1_best.json if present).
Like phase1, num_search_workers stays at 1 — presolve effects must be measured
without multi-worker search masking them.

Do NOT modify locked keys (random_seed, num_search_workers).
"""
import json
import pathlib
import sys

_SHARED = pathlib.Path(__file__).resolve().parent.parent / "shared"
sys.path.insert(0, str(_SHARED))

from baseline_params import BASELINE  # noqa: E402


PHASE_LOCKED = {
    "random_seed": 0,
    "num_search_workers": 1,
}


def _load_prev(name):
    p = _SHARED / name
    if p.exists():
        return json.loads(p.read_text())
    return {}


_PHASE1 = _load_prev("phase1_best.json")


# EVOLVE-BLOCK-START
PRESOLVE_OVERRIDES = {
    "cp_model_probing_level": 1,
}
# EVOLVE-BLOCK-END


def get_params():
    p = dict(BASELINE)
    p.update(_PHASE1)
    p.update(PRESOLVE_OVERRIDES)
    p.update(PHASE_LOCKED)  # phase1 may have stored workers; re-pin to 1
    return p


def get_phase_overrides():
    return dict(PRESOLVE_OVERRIDES)
