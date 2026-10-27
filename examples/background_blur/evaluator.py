"""
Evaluator for the portrait background-blur task.

The scoring rule, in one line:

    Quality is a GATE, not a trade-off. The only way to score is to look like the
    reference and be faster.

    speedup = baseline_time / candidate_time
    reject if mean_ssim < 0.98          (overall fidelity)
           or worst_frame_ssim < 0.95   (no bad frames)
           or worst_region_ssim < 0.90  (no bad REGIONS - see below)
    otherwise score = speedup

That gate is what stops the search from "winning" by simply doing less work.

The worst-REGION term is the one people forget. Mean SSIM is blind to localised
damage: a candidate that blurs frame 0's background once and reuses it forever
leaves a person-shaped ghost, yet still scores mean 0.987 / worst-frame 0.981 and
sails through the first two gates. It scored 47x before the region gate existed.
Attack your own scorer (see test_gaming.py) before you spend a single LLM token.

Structured as an OpenEvolve CASCADE so we never spend a timing benchmark on a
candidate that is already wrong:

    stage 1  cheap correctness smoke  (shape/dtype/finite, and NOT a no-op)
    stage 2  the hard quality gate    (per-frame SSIM vs the reference)
    stage 3  the timing benchmark     (only for candidates that earned it)

Every rejection returns ARTIFACTS explaining exactly what went wrong (which frame,
what SSIM, whether the output was simply the untouched input). The LLM sees the
reason, not just a number.
"""

import importlib.util
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixtures import (  # noqa: E402
    N_FRAMES,
    SIGMA,
    make_sequence,
    naive_process,
    reference_process,
    ssim,
    worst_region_ssim,
)
from openevolve.evaluation_result import EvaluationResult  # noqa: E402

# ---------------------------------------------------------------------------
# The hard quality gate
# ---------------------------------------------------------------------------
MEAN_SSIM_GATE = 0.98
WORST_SSIM_GATE = 0.95

# Whole-image SSIM is blind to LOCALISED damage. A candidate that blurs frame 0's
# background once and reuses it for the whole sequence leaves a person-shaped ghost
# where the person used to be - and still scores mean 0.987 / worst-frame 0.981,
# sailing through the two gates above. (It scored 47x before this gate existed.)
# Grading the worst 16x16 block catches it: the ghost region scores 0.74, while
# legitimate approximations (half-res blur, 3x box blur) stay above 0.97.
WORST_REGION_GATE = 0.90

# A candidate that just returns the input unchanged still scores fairly high SSIM
# (most of the frame is background that is only mildly blurred). This floor is a
# cheap stage-1 tripwire for "did you actually do anything at all".
NOOP_SSIM_CEILING = 0.995

TIMING_REPEATS = 3

# Computed once per evaluator process and reused (see README: the baseline must be
# measured under the same conditions as the candidate, serially).
_CACHE = {}


def _fixtures():
    if "frames" not in _CACHE:
        frames, masks = make_sequence()
        _CACHE["frames"] = frames
        _CACHE["masks"] = masks
        _CACHE["golden"] = reference_process(frames, masks, SIGMA)
        # SSIM of the *untouched input* vs the reference: anything at or above this
        # did not really blur anything.
        _CACHE["identity_ssim"] = float(
            np.mean([ssim(f, g) for f, g in zip(frames, _CACHE["golden"])])
        )
    return _CACHE["frames"], _CACHE["masks"], _CACHE["golden"]


def _benchmark(candidate_fn, frames, masks, sigma):
    """Time the baseline and the candidate FAIRLY, and return (baseline, candidate, out).

    Two things matter here, and getting either wrong silently corrupts every score:

    1. INTERLEAVE. Do not measure the baseline once, cache it, and compare every
       future candidate against it. If that one measurement happened under transient
       load it is inflated forever, and so is every speedup you report. (This is not
       hypothetical: an earlier version of this evaluator did exactly that and
       reported 82x for a program that is really 61x.) Measuring both back-to-back
       in the same loop cancels slow drift.

    2. WARM UP, then take the MINIMUM of N runs. Background load can only ever ADD
       time, so the minimum is the best estimate of the true cost - for both sides.
    """
    # Warm-up (allocator, pages, BLAS threads) - not measured.
    naive_process(frames[:1], masks[:1], sigma)
    candidate_fn(frames[:1], masks[:1], sigma)

    t_base = float("inf")
    t_cand = float("inf")
    out = None
    for _ in range(TIMING_REPEATS):
        t0 = time.perf_counter()
        naive_process(frames, masks, sigma)
        t_base = min(t_base, time.perf_counter() - t0)

        t0 = time.perf_counter()
        out = candidate_fn(frames, masks, sigma)
        t_cand = min(t_cand, time.perf_counter() - t0)

    return t_base, t_cand, out


def _load(program_path):
    spec = importlib.util.spec_from_file_location("candidate", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "process_sequence"):
        raise AttributeError("program must define process_sequence(frames, masks, sigma)")
    return module


def _check_shapes(out, frames):
    """Structural validation. Returns an error string, or None if OK."""
    if out is None:
        return "process_sequence returned None"
    if len(out) != len(frames):
        return f"expected {len(frames)} output frames, got {len(out)}"
    for i, (o, f) in enumerate(zip(out, frames)):
        o = np.asarray(o)
        if o.shape != f.shape:
            return f"frame {i}: expected shape {f.shape}, got {o.shape}"
        if not np.all(np.isfinite(o)):
            return f"frame {i}: output contains NaN or Inf"
    return None


def _score_quality(out, golden):
    """Returns (mean_ssim, worst_frame_ssim, worst_frame_idx, worst_region_ssim, per_frame)."""
    outs = [np.asarray(o, dtype=np.float32) for o in out]
    per_frame = [ssim(o, g) for o, g in zip(outs, golden)]
    worst_region = min(worst_region_ssim(o, g) for o, g in zip(outs, golden))
    return (
        float(np.mean(per_frame)),
        float(np.min(per_frame)),
        int(np.argmin(per_frame)),
        float(worst_region),
        per_frame,
    )


def _fail(stage, reason, **extra):
    """Rejection. Always carries `ssim` because it is a MAP-Elites feature dimension
    (a program missing a declared feature dimension makes the database raise)."""
    metrics = {"combined_score": 0.0, "ssim": 0.0}
    metrics.update({k: float(v) for k, v in extra.items() if isinstance(v, (int, float))})
    artifacts = {"failure_stage": stage, "failure_reason": reason}
    artifacts.update({k: str(v) for k, v in extra.items()})
    return EvaluationResult(metrics=metrics, artifacts=artifacts)


# ---------------------------------------------------------------------------
# Stage 1 - cheap correctness smoke test
# ---------------------------------------------------------------------------
def evaluate_stage1(program_path):
    """Fast reject: does it run, produce the right shape, and actually blur?"""
    try:
        module = _load(program_path)
        frames, masks, golden = _fixtures()

        # Only the first two frames - this stage must stay cheap.
        f2, m2, g2 = frames[:2], masks[:2], golden[:2]
        out = module.process_sequence(f2, m2, SIGMA)

        err = _check_shapes(out, f2)
        if err:
            return _fail("stage1", err)

        mean_ssim, _, _, _, _ = _score_quality(out, g2)

        # Anti-gaming tripwire: returning the input untouched is fast and scores
        # deceptively well. Reject it here, before it ever reaches the benchmark.
        identity_ssim = float(np.mean([ssim(f, g) for f, g in zip(f2, g2)]))
        if mean_ssim >= NOOP_SSIM_CEILING and mean_ssim <= identity_ssim + 1e-6:
            return _fail(
                "stage1",
                "output is (nearly) the untouched input - the background was not blurred",
                ssim=mean_ssim,
                identity_ssim=identity_ssim,
            )

        return EvaluationResult(
            metrics={"combined_score": 1.0, "ssim": float(mean_ssim)},
            artifacts={"stage1": f"ok (ssim {mean_ssim:.4f} on 2 frames)"},
        )
    except Exception as e:
        return _fail("stage1", f"exception: {e}", traceback=traceback.format_exc())


# ---------------------------------------------------------------------------
# Stage 2 - the HARD QUALITY GATE
# ---------------------------------------------------------------------------
def evaluate_stage2(program_path):
    """The gate. Fidelity is pass/fail; it is never traded against speed."""
    try:
        module = _load(program_path)
        frames, masks, golden = _fixtures()

        out = module.process_sequence(frames, masks, SIGMA)
        err = _check_shapes(out, frames)
        if err:
            return _fail("stage2", err)

        mean_ssim, worst_ssim, worst_i, worst_region, per_frame = _score_quality(out, golden)

        if (
            mean_ssim < MEAN_SSIM_GATE
            or worst_ssim < WORST_SSIM_GATE
            or worst_region < WORST_REGION_GATE
        ):
            return _fail(
                "stage2",
                (
                    f"QUALITY GATE FAILED: mean SSIM {mean_ssim:.4f} (needs >= {MEAN_SSIM_GATE}), "
                    f"worst-frame SSIM {worst_ssim:.4f} on frame {worst_i} "
                    f"(needs >= {WORST_SSIM_GATE}), worst-REGION SSIM {worst_region:.4f} "
                    f"(needs >= {WORST_REGION_GATE}). "
                    "The output no longer matches the reference blur. Speed is only counted "
                    "once the result is faithful on EVERY frame AND in EVERY region - do not "
                    "under-blur, skip frames, blur the person, or reuse a stale background "
                    "(a stale background leaves a person-shaped ghost that wrecks one region "
                    "while barely moving the frame average)."
                ),
                ssim=mean_ssim,
                worst_ssim=worst_ssim,
                worst_frame=worst_i,
                worst_region_ssim=worst_region,
                per_frame_ssim=", ".join(f"{s:.4f}" for s in per_frame),
            )

        return EvaluationResult(
            metrics={
                "combined_score": 1.0,
                "ssim": mean_ssim,
                "worst_ssim": worst_ssim,
                "worst_region_ssim": worst_region,
            },
            artifacts={
                "stage2": (
                    f"quality gate PASSED (mean {mean_ssim:.4f}, worst-frame {worst_ssim:.4f}, "
                    f"worst-region {worst_region:.4f})"
                )
            },
        )
    except Exception as e:
        return _fail("stage2", f"exception: {e}", traceback=traceback.format_exc())


# ---------------------------------------------------------------------------
# Stage 3 - timing benchmark (earned only by candidates that passed the gate)
# ---------------------------------------------------------------------------
def evaluate_stage3(program_path):
    """Measure the speedup. Fitness = speedup."""
    try:
        module = _load(program_path)
        frames, masks, golden = _fixtures()

        # Baseline and candidate are measured interleaved, in this same call, so a
        # noisy machine cannot inflate the ratio. See _benchmark().
        baseline, best, out = _benchmark(module.process_sequence, frames, masks, SIGMA)

        # Re-verify quality on the timed output: speed must never be bought with
        # fidelity, not even by a candidate that behaves differently under timing.
        err = _check_shapes(out, frames)
        if err:
            return _fail("stage3", err)
        mean_ssim, worst_ssim, worst_i, worst_region, _ = _score_quality(out, golden)
        if (
            mean_ssim < MEAN_SSIM_GATE
            or worst_ssim < WORST_SSIM_GATE
            or worst_region < WORST_REGION_GATE
        ):
            return _fail(
                "stage3",
                f"quality regressed under timing: mean {mean_ssim:.4f}, "
                f"worst-frame {worst_ssim:.4f}, worst-region {worst_region:.4f}",
                ssim=mean_ssim,
                worst_ssim=worst_ssim,
                worst_frame=worst_i,
                worst_region_ssim=worst_region,
            )

        speedup = baseline / best
        ms_per_frame = 1000.0 * best / max(1, N_FRAMES)
        baseline_ms = 1000.0 * baseline / max(1, N_FRAMES)

        return EvaluationResult(
            metrics={
                # Fitness IS the speedup - but only reachable through the gate.
                "combined_score": float(speedup),
                "speedup": float(speedup),
                "ssim": float(mean_ssim),
                "worst_ssim": float(worst_ssim),
                "worst_region_ssim": float(worst_region),
                "ms_per_frame": float(ms_per_frame),
            },
            artifacts={
                "result": (
                    f"{speedup:.2f}x speedup | {ms_per_frame:.2f} ms/frame "
                    f"(baseline {baseline_ms:.2f} ms/frame) | mean SSIM {mean_ssim:.4f}, "
                    f"worst-frame {worst_ssim:.4f}, worst-region {worst_region:.4f}"
                )
            },
        )
    except Exception as e:
        return _fail("stage3", f"exception: {e}", traceback=traceback.format_exc())


# ---------------------------------------------------------------------------
# Non-cascade entry point (used when cascade_evaluation: false)
# ---------------------------------------------------------------------------
def evaluate(program_path):
    r1 = evaluate_stage1(program_path)
    if r1.metrics.get("combined_score", 0.0) < 0.5:
        return r1
    r2 = evaluate_stage2(program_path)
    if r2.metrics.get("combined_score", 0.0) < 0.5:
        return r2
    return evaluate_stage3(program_path)
