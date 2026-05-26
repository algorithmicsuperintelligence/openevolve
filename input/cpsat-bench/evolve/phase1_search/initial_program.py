"""
Phase 1: tune CP-SAT search / subsolver knobs.

Reference: shared/cpsat_params_reference.md
  (full proto field listing — consult before adding keys; invalid names
  surface as `invalid_param` and zero the score.)

Three evolution surfaces live in this file's EVOLVE-BLOCK:
  GLOBAL_OVERRIDES  — applied to every problem
  SIZE_BUCKETS      — applied conditionally on `num_constraints`
                      (small / medium / large; first match wins).
                      The dataset spans ~7k–246k constraints; thresholds
                      below split it ~evenly. Adjust freely.
  STAGE3_OVERRIDES  — applied ONLY when stage == "stage3" AND the problem
                      is in the outlier set (Statistics/outliers_top.csv).
                      Tune knobs that help long-tail / hard outliers
                      without regressing fast/mid problems.

Targeted namespace for phase 1 (search/subsolvers):
  extra_subsolvers, ignore_subsolvers, interleave_search,
  use_feasibility_jump, use_feasibility_pump, search_branching,
  preferred_variable_order, repair_hint, diversify_lns_params.

Other params stay at BASELINE. This phase pins num_search_workers=1 so other
knobs are evaluated without multi-thread / multi-subsolver noise. Phase 3
raises the worker count to explore subsolver-mix effects.

Do NOT modify locked keys (random_seed, num_search_workers).
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
GLOBAL_OVERRIDES = {
    "extra_subsolvers": ["default_lp", "no_lp"],
    "ignore_subsolvers": ["max_lp"],
    "interleave_search": True,
    "use_feasibility_jump": False,
    "use_feasibility_pump": False,
}

# (max_num_constraints_exclusive, override_dict). First match wins; the final
# entry must use float("inf") as the sentinel for "no upper bound".
SIZE_BUCKETS = [
    (50_000,         {}),  # small problems — keep GLOBAL_OVERRIDES intact
    (150_000,        {}),  # medium
    (float("inf"),   {}),  # large
]

# Applied ONLY when stage == "stage3" AND problem["is_outlier"] is True.
# Outliers are pre-identified in Statistics/outliers_top.csv (residual log10
# slowdown vs the runtime ~ vars^a * cons^b regression). Their search budget
# is dominated by conflict learning + LP iter; use this dict to add aggressive
# settings that would hurt simple problems but pay off on hard ones.
STAGE3_OVERRIDES = {}
# EVOLVE-BLOCK-END


def _bucket_override(num_constraints):
    for upper, override in SIZE_BUCKETS:
        if num_constraints < upper:
            return override
    return {}


def get_params(problem=None, stage=None):
    p = dict(BASELINE)
    p.update(GLOBAL_OVERRIDES)
    if problem is not None:
        p.update(_bucket_override(int(problem.get("num_constraints") or 0)))
        if stage == "stage3" and problem.get("is_outlier"):
            p.update(STAGE3_OVERRIDES)
    p.update(PHASE_LOCKED)  # re-enforce phase lock last
    return p


def get_phase_overrides():
    """Used by extract_best.py — returns ONLY this phase's evolved GLOBAL dict.
    SIZE_BUCKETS / STAGE3_OVERRIDES are extracted via the helpers below."""
    return dict(GLOBAL_OVERRIDES)


def get_phase_size_buckets():
    """Returns list[(upper_exclusive, override_dict)] for chaining to next phase."""
    return [(u, dict(d)) for u, d in SIZE_BUCKETS]


def get_phase_stage3_overrides():
    return dict(STAGE3_OVERRIDES)
