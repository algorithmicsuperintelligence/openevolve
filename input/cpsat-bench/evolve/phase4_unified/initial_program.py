"""
Phase 4: unified refinement.

Reference: shared/cpsat_params_reference.md  (any section may be tuned here.)

EVOLVE-BLOCK below is auto-materialized by prepare_phase_unified.py from the
union of phase{1,2,3}_best.json winners (GLOBAL_OVERRIDES), plus the merged
SIZE_BUCKETS and STAGE3_OVERRIDES from those phases. LLM may then tune all
three surfaces jointly:

  GLOBAL_OVERRIDES  — every-problem params
  SIZE_BUCKETS      — per-constraint-count overrides
  STAGE3_OVERRIDES  — outlier-only overrides (stage3 cascade)

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
GLOBAL_OVERRIDES = {}
SIZE_BUCKETS = [
    (50_000,       {}),
    (150_000,      {}),
    (float("inf"), {}),
]
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
    p.update(PHASE_LOCKED)
    return p


def get_phase_overrides():
    return dict(GLOBAL_OVERRIDES)


def get_phase_size_buckets():
    return [(u, dict(d)) for u, d in SIZE_BUCKETS]


def get_phase_stage3_overrides():
    return dict(STAGE3_OVERRIDES)
