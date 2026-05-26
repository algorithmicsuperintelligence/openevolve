"""
Phase 2: tune CP-SAT presolve / probing knobs.

Reference: shared/cpsat_params_reference.md  (presolve section, lines ~132-162;
LP/cuts section may also be relevant for probing-LP interactions.)

Three evolution surfaces in this file's EVOLVE-BLOCK:
  GLOBAL_OVERRIDES   — applied to every problem
  SIZE_BUCKETS       — applied conditionally on `num_constraints`
                       (large problems often want LIGHTER presolve to avoid
                       spending budget on substitution work that doesn't pay
                       off; small problems can afford full probing.)
  STAGE3_OVERRIDES   — applied ONLY when stage == "stage3" AND outlier.
                       Outliers have heavy presolve-time / search-time
                       imbalance — extra knobs here let you trade presolve
                       depth vs search depth without hurting normal cases.

Targeted namespace:
  cp_model_probing_level, cp_model_presolve, symmetry_level,
  presolve_use_bva, presolve_bve_threshold, presolve_substitution_level,
  max_presolve_iterations, presolve_probing_deterministic_time_limit,
  find_big_linear_overlap, infer_all_diffs, mip_presolve_level.

Inherits phase1 winners from shared/phase1_*.json. Like phase1, workers stays
at 1 — presolve effects must be measured without multi-worker search masking
them. Phase 3 raises workers.

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


def _load_prev_dict(name):
    p = _SHARED / name
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _load_prev_buckets(name):
    """Returns list[(upper, override_dict)] or None if file missing.

    File schema: [[upper_or_null, override_dict], ...] (JSON cannot encode inf,
    so the writer stores null for the float('inf') sentinel)."""
    p = _SHARED / name
    if not p.exists():
        return None
    raw = json.loads(p.read_text())
    out = []
    for upper, override in raw:
        out.append((float("inf") if upper is None else upper, override))
    return out


_PHASE1 = _load_prev_dict("phase1_best.json")
_PHASE1_BUCKETS = _load_prev_buckets("phase1_buckets.json")
_PHASE1_STAGE3 = _load_prev_dict("phase1_stage3.json")


# EVOLVE-BLOCK-START
GLOBAL_OVERRIDES = {
    "cp_model_probing_level": 1,
}

SIZE_BUCKETS = [
    (50_000,         {}),
    (150_000,        {}),
    (float("inf"),   {}),
]

STAGE3_OVERRIDES = {}
# EVOLVE-BLOCK-END


def _merge_bucket(num_constraints):
    """Merge phase1 bucket + phase2 bucket. Phase2 wins on conflicts."""
    out = {}
    for buckets in (_PHASE1_BUCKETS or [], SIZE_BUCKETS):
        for upper, override in buckets:
            if num_constraints < upper:
                out.update(override)
                break
    return out


def get_params(problem=None, stage=None):
    p = dict(BASELINE)
    p.update(_PHASE1)
    p.update(GLOBAL_OVERRIDES)
    if problem is not None:
        p.update(_merge_bucket(int(problem.get("num_constraints") or 0)))
        if stage == "stage3" and problem.get("is_outlier"):
            p.update(_PHASE1_STAGE3)
            p.update(STAGE3_OVERRIDES)
    p.update(PHASE_LOCKED)  # phase1 may have stored workers; re-pin to 1
    return p


def get_phase_overrides():
    return dict(GLOBAL_OVERRIDES)


def get_phase_size_buckets():
    return [(u, dict(d)) for u, d in SIZE_BUCKETS]


def get_phase_stage3_overrides():
    return dict(STAGE3_OVERRIDES)
