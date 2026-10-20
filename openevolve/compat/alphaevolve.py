"""Adapt local Google Cloud AlphaEvolve evaluators without a Google SDK dependency."""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Optional

from openevolve.evaluation_result import EvaluationResult


def _field(value: Any, name: str) -> Any:
    """Read the same contract from model objects and their model_dump() dictionaries."""
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


@dataclass
class _AlphaEvolveEvaluator:
    evaluator: Callable
    objective_metric: str
    failure_sentinel: Optional[float]
    failure_score: float

    def __call__(self, program_path: str) -> EvaluationResult:
        path = Path(program_path)
        candidate = {
            "content": {"files": [{"path": path.name, "content": path.read_text(encoding="utf-8")}]}
        }
        result = self.evaluator(candidate)
        scores = _field(_field(result, "scores"), "scores")
        if not isinstance(scores, (list, tuple)) or not scores:
            raise ValueError("AlphaEvolve evaluation must contain a non-empty scores.scores list")

        metrics = {}
        failed_metrics = []
        for entry in scores:
            metric, score = _field(entry, "metric"), _field(entry, "score")
            if not isinstance(metric, str) or not metric.strip():
                raise ValueError("AlphaEvolve metric names must be non-empty strings")
            if metric in metrics or metric == "error":
                raise ValueError(f"Duplicate or reserved AlphaEvolve metric: {metric!r}")
            if metric == "combined_score" and self.objective_metric != "combined_score":
                raise ValueError(
                    "Select combined_score as objective_metric to preserve that metric"
                )
            if score is not None and (isinstance(score, bool) or not isinstance(score, Real)):
                raise TypeError(f"AlphaEvolve score for {metric!r} must be a number or None")
            if (
                score is None
                or not math.isfinite(score)
                or (self.failure_sentinel is not None and score == self.failure_sentinel)
            ):
                failed_metrics.append(metric)
                metrics[metric] = self.failure_score
            else:
                metrics[metric] = float(score)

        if self.objective_metric not in metrics:
            raise ValueError(
                f"AlphaEvolve evaluation is missing objective {self.objective_metric!r}"
            )

        artifacts = {}
        insights = _field(_field(result, "insights"), "insights")
        if insights:
            artifacts["alphaevolve_insights"] = json.dumps(
                [
                    {"label": _field(item, "label"), "text": _field(item, "text")}
                    for item in insights
                ],
                ensure_ascii=False,
            )

        # Pick an explicit objective, rather than averaging unrelated feature metrics.
        metrics["combined_score"] = metrics[self.objective_metric]
        if failed_metrics:
            # OpenEvolve error metrics alone have fitness zero, which could beat a
            # valid negative objective (e.g. neg_tour_length). Retain a failure fitness.
            metrics["combined_score"] = self.failure_score
            metrics["error"] = 0.0
            artifacts["error"] = "AlphaEvolve evaluation failed for metrics: " + ", ".join(
                failed_metrics
            )
        return EvaluationResult(metrics=metrics, artifacts=artifacts)


def adapt_evaluator(
    evaluator: Callable,
    *,
    objective_metric: str,
    failure_sentinel: Optional[float] = -1e12,
    failure_score: float = -1e12,
) -> Callable[[str], EvaluationResult]:
    """Wrap a synchronous, single-file AlphaEvolve evaluator for OpenEvolve.

    The callable receives ``content.files[0].content`` instead of a file path.
    Its ``scores.scores`` envelope may be a dictionary or SDK-style object.
    All metrics are retained; ``objective_metric`` is also used as combined_score.
    Insights become JSON in the ``alphaevolve_insights`` artifact.

    None/non-finite scores and the exact configurable failure sentinel mark the
    evaluation failed. Choose a finite ``failure_score`` below all valid scores,
    especially for negative objectives. Set ``failure_sentinel=None`` if -1e12
    is a legitimate score. Exceptions and malformed contracts propagate to the
    existing OpenEvolve evaluator error handling.

    This only adapts evaluation: it does not provide cloud SDK emulation, config
    translation, multi-file candidates, or execution isolation. The wrapped
    evaluator executes trusted candidate code with its existing privileges.
    """
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    if not isinstance(objective_metric, str) or not objective_metric.strip():
        raise ValueError("objective_metric must be a non-empty string")
    if objective_metric == "error":
        raise ValueError("error is reserved for failure handling")
    for name, value in (("failure_score", failure_score), ("failure_sentinel", failure_sentinel)):
        if name == "failure_sentinel" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    return _AlphaEvolveEvaluator(
        evaluator, objective_metric, failure_sentinel, float(failure_score)
    )
