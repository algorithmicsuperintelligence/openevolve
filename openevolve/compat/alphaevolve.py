"""Adapt local Google Cloud AlphaEvolve evaluators without a Google SDK dependency."""

import json
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Optional

from openevolve.config import Config
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


def _config_mapping(value: Any, path: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _warn_unmapped_fields(value: Mapping, supported: set, prefix: str = "") -> None:
    paths = [f"{prefix}{key}" for key in value if key not in supported and value[key] is not None]
    if paths:
        warnings.warn(
            "AlphaEvolve settings not imported: " + ", ".join(paths),
            UserWarning,
            stacklevel=3,
        )


def config_from_alphaevolve(
    experiment: Mapping, *, api_base: str, api_key: Optional[str] = None
) -> Config:
    """Import an experiment dictionary for local, single-file Python evaluation.

    Requires explicit models and a positive run_settings.max_programs; it never
    guesses a model or falls back to OE's 10,000-iteration default. Model names
    and relative weights are preserved. The endpoint and optional key must come
    from the caller, not Google Cloud fields. No files, clients or requests are
    created. Pass SDK models as model_dump() dictionaries.

    max_programs becomes max_iterations (generation attempts, plus OE's seed
    evaluation), not a strict evaluation, request, token, or cost limit. Optional
    settings without mappings warn; max_duration is rejected rather than silently
    losing a wall-clock limit. Other OE settings retain their defaults, except
    cascade evaluation is disabled for the single-stage local evaluator adapter.
    """
    experiment = _config_mapping(experiment, "experiment")
    run = _config_mapping(experiment.get("run_settings"), "run_settings")
    generation = _config_mapping(experiment.get("generation_settings"), "generation_settings")
    if not isinstance(api_base, str) or not api_base.strip():
        raise ValueError("api_base must explicitly name the OpenEvolve model endpoint")
    if experiment.get("program_language") != "python":
        raise ValueError("Only program_language='python' is supported by this local import")
    description = experiment.get("problem_description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("problem_description must be a non-empty string")
    iterations = run.get("max_programs")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("run_settings.max_programs must be an explicit positive integer")
    if run.get("max_duration") is not None:
        raise ValueError("run_settings.max_duration cannot be enforced by this import")

    mixture = generation.get("models")
    if not isinstance(mixture, (list, tuple)) or not mixture:
        raise ValueError("generation_settings.models must be a non-empty list")
    models = []
    for index, entry in enumerate(mixture):
        path = f"generation_settings.models[{index}]"
        entry = _config_mapping(entry, path)
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{path}.name must be a non-empty string")
        weight = entry.get("weight")
        if weight is None:
            weight = 1.0
        if (
            isinstance(weight, bool)
            or not isinstance(weight, Real)
            or not math.isfinite(weight)
            or weight <= 0
        ):
            raise ValueError(f"{path}.weight must be finite and positive")
        models.append({"name": name, "weight": float(weight)})
        _warn_unmapped_fields(entry, {"name", "weight"}, path + ".")
    if not math.isfinite(sum(model["weight"] for model in models)):
        raise ValueError("generation_settings.models weights must have a finite sum")

    # Prefix the text so a description like 'system_message' remains literal,
    # rather than accidentally selecting a PromptSampler template of that name.
    system_message = "Problem description:\n" + description
    context = generation.get("context")
    if context is not None:
        if not isinstance(context, str):
            raise ValueError("generation_settings.context must be a string or None")
        if context:
            system_message += "\n\nAdditional context:\n" + context

    _warn_unmapped_fields(
        experiment,
        {"title", "problem_description", "program_language", "run_settings", "generation_settings"},
    )
    _warn_unmapped_fields(run, {"max_programs", "max_duration"}, "run_settings.")
    _warn_unmapped_fields(generation, {"models", "context"}, "generation_settings.")
    return Config.from_dict(
        {
            "max_iterations": iterations,
            "language": "python",
            "llm": {"api_base": api_base, "api_key": api_key, "models": models},
            "prompt": {"system_message": system_message},
            "evaluator": {"cascade_evaluation": False},
        }
    )
