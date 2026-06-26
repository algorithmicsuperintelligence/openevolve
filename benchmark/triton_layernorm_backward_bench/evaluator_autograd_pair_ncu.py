"""Ncu-aware wrapper around the LayerNorm autograd-pair evaluator.

Runs the normal timing-based evaluator first (cheap, every iteration). If the
candidate is correct, optionally profiles it with `ncu` and folds the result
into both the score (a small penalty for register spilling) and the artifacts
(full hardware metrics + ncu's rule-engine suggestion get rendered into the
*next* LLM prompt by OpenEvolve's prompt sampler, since EvaluationResult
artifacts are surfaced there).

ncu is the expensive part of this pipeline (a single `--metrics` pass still
costs several seconds, even without `--set full`), so by default this only
profiles a candidate when it's a new best score for this run
(AUTOGRAD_PAIR_NCU_MODE=on_improve). Set it to "always" to profile every
correct candidate, or "off" to disable entirely and behave exactly like
evaluator_autograd_pair.py.

Best-effort throughout: any ncu/ncu_report failure (missing binary, no
perf-counter permission, timeout) is recorded as an artifact and otherwise
ignored -- it never blocks or fails the underlying evaluation.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import ncu_profile  # noqa: E402
import task_spec  # noqa: E402
from evaluator_autograd_pair import _json, evaluate as _base_evaluate  # noqa: E402

try:
    from openevolve.evaluation_result import EvaluationResult
except Exception:  # pragma: no cover
    from dataclasses import dataclass, field

    @dataclass
    class EvaluationResult:
        metrics: dict[str, float]
        artifacts: dict[str, str | bytes] = field(default_factory=dict)


NCU_MODE = os.environ.get("AUTOGRAD_PAIR_NCU_MODE", "on_improve")  # off | always | on_improve
NCU_TIMEOUT = int(os.environ.get("AUTOGRAD_PAIR_NCU_TIMEOUT", "120"))
NCU_WARMUP = int(os.environ.get("AUTOGRAD_PAIR_NCU_WARMUP", "5"))
NCU_SPILL_PENALTY = float(os.environ.get("AUTOGRAD_PAIR_NCU_SPILL_PENALTY", "0.1"))
NCU_STATE_PATH = Path(
    os.environ.get("AUTOGRAD_PAIR_NCU_STATE", str(BENCHMARK_DIR / ".autograd_pair_ncu_state.json"))
)
NCU_SHAPE = os.environ.get("AUTOGRAD_PAIR_NCU_SHAPE")  # "rows,cols,dtype" override


def _representative_case():
    if NCU_SHAPE:
        rows, cols, dtype_name = NCU_SHAPE.split(",")
        return task_spec.TestCase(int(rows), int(cols), dtype_name.strip(), 5e-2, 5e-2)
    # Default: the largest case by element count, i.e. the "hot path" workload.
    return max(task_spec.BENCHMARK_CASES, key=lambda c: c.rows * c.cols)


def _load_best_score() -> float:
    try:
        return json.loads(NCU_STATE_PATH.read_text()).get("best_score", float("-inf"))
    except Exception:
        return float("-inf")


def _save_best_score(score: float) -> None:
    try:
        NCU_STATE_PATH.write_text(json.dumps({"best_score": score}))
    except Exception:
        pass  # best-effort bookkeeping; never block evaluation on a write failure


def _should_profile(combined_score: float) -> bool:
    if NCU_MODE == "off":
        return False
    if NCU_MODE == "always":
        return True
    # on_improve (default)
    best = _load_best_score()
    return combined_score > best


def evaluate(program_path: str) -> EvaluationResult:
    result = _base_evaluate(program_path)
    if result.metrics.get("correct", 0.0) != 1.0:
        return result

    combined_score = float(result.metrics.get("combined_score", float("-inf")))
    if not _should_profile(combined_score):
        return result

    case = _representative_case()
    profile = ncu_profile.run_ncu_profile(
        program_path,
        case,
        warmup=NCU_WARMUP,
        timeout=NCU_TIMEOUT,
    )

    artifacts = dict(result.artifacts)
    metrics = dict(result.metrics)

    if not profile.ok:
        artifacts["ncu_profile_error"] = profile.error or "unknown ncu failure"
        # Still record that we attempted it at this score, so a flaky ncu
        # failure doesn't make every subsequent candidate re-attempt forever.
        _save_best_score(combined_score)
        return EvaluationResult(metrics=metrics, artifacts=artifacts)

    flags = ncu_profile.derive_flags(profile.aggregate)
    patterns = ncu_profile.classify_patterns(profile.aggregate)

    metrics["ncu_occupancy_pct"] = flags["occupancy_pct"] if flags["occupancy_pct"] is not None else -1.0
    metrics["ncu_long_scoreboard_stall_ratio"] = (
        flags["long_scoreboard_stall_ratio"] if flags["long_scoreboard_stall_ratio"] is not None else -1.0
    )
    metrics["ncu_sm_throughput_pct"] = flags["sm_throughput_pct"] if flags["sm_throughput_pct"] is not None else -1.0
    metrics["ncu_dram_throughput_pct"] = (
        flags["dram_throughput_pct"] if flags["dram_throughput_pct"] is not None else -1.0
    )
    for name, info in patterns.items():
        metrics[f"ncu_pattern_{name}"] = 1.0 if info["flag"] else 0.0

    adjusted_score = combined_score
    if patterns["register_spill"]["flag"]:
        # Wall-clock timing already captures most performance regressions, but
        # register spilling on the one shape profiled here can hide in the
        # aggregate score across all benchmark shapes. Penalize it directly
        # using a hardware signal wall-clock timing alone wouldn't have
        # flagged. Pattern K is the only one of the four classified patterns
        # treated as unambiguously bad enough to affect the score directly --
        # small_grid/latency_bound/bandwidth_bound/occupancy_limited are left
        # as diagnostic-only (see README): none of them is "always worse,"
        # just worth the LLM reading and reacting to.
        adjusted_score = combined_score * (1.0 - NCU_SPILL_PENALTY)
    metrics["combined_score"] = adjusted_score
    metrics["combined_score_pre_ncu"] = combined_score

    artifacts["ncu_profile"] = _json(
        {
            "shape": [case.rows, case.cols],
            "dtype": case.dtype_name,
            "aggregate": profile.aggregate,
            "derived_flags": flags,
            "patterns": patterns,
            "top_rule_suggestion": profile.top_rule,
        }
    )

    _save_best_score(adjusted_score)
    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"Usage: {argv[0]} PROGRAM_PATH")
        return 2
    result = evaluate(argv[1])
    print(json.dumps({"metrics": result.metrics, "artifacts": result.artifacts}, indent=2))
    return 0 if result.metrics.get("correct", 0.0) == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
