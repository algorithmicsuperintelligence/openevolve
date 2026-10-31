"""Use synchronous OpenEvolve evaluators with the AlphaEvolve candidate contract."""

import inspect
import math
import tempfile
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Callable

from openevolve.compat.alphaevolve import _field
from openevolve.evaluation_result import EvaluationResult


@dataclass(frozen=True)
class _OpenEvolveEvaluator:
    evaluator: Callable
    objective_metric: str
    failure_score: float

    def _evaluation(self, scores: dict, failure: str = "") -> dict:
        if failure:
            scores = {name: self.failure_score for name in scores}
            scores[self.objective_metric] = self.failure_score
        result = {
            "scores": {
                "scores": [{"metric": name, "score": score} for name, score in scores.items()]
            }
        }
        if failure:
            result["insights"] = {
                "insights": [{"label": "OpenEvolve Evaluation Error", "text": failure}]
            }
        return result

    def __call__(self, candidate: Any) -> dict:
        files = _field(_field(candidate, "content"), "files")
        if not isinstance(files, (list, tuple)) or len(files) != 1:
            raise ValueError("AlphaEvolve candidate must contain exactly one source file")
        source = _field(files[0], "content")
        if not isinstance(source, str):
            raise TypeError("AlphaEvolve source content must be a string")
        if _field(files[0], "program_language") not in (None, "python"):
            raise ValueError("Only single-file Python candidates are supported")

        # Never use the candidate's path for filesystem writes. Close the file
        # before calling the evaluator, including on platforms with file locks.
        with tempfile.TemporaryDirectory(prefix="openevolve-alphaevolve-") as directory:
            path = Path(directory) / "program.py"
            path.write_text(source, encoding="utf-8", newline="")
            try:
                result = self.evaluator(str(path))
            except Exception as exc:
                # Do not copy exception text, local paths or credentials into
                # feedback that a caller may later submit to a cloud service.
                return self._evaluation({}, f"OpenEvolve evaluator raised {type(exc).__name__}")

        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
            raise TypeError("OpenEvolve evaluator must return synchronously")

        artifacts = {}
        if isinstance(result, EvaluationResult):
            metrics, artifacts = result.metrics, result.artifacts
        elif isinstance(result, Mapping):
            metrics = result
        else:
            raise TypeError("OpenEvolve evaluator must return metrics or EvaluationResult")
        if not isinstance(metrics, Mapping) or not metrics:
            raise ValueError("OpenEvolve evaluation must contain a non-empty metrics mapping")
        if not isinstance(artifacts, Mapping):
            raise TypeError("OpenEvolve artifacts must be a mapping")
        if any(name != "error" for name in artifacts):
            warnings.warn(
                "OpenEvolve artifacts are not exported by this adapter",
                UserWarning,
                stacklevel=2,
            )

        failed = "error" in metrics or "error" in artifacts or bool(metrics.get("timeout", False))
        failure = "OpenEvolve evaluation reported an error or timeout" if failed else ""
        scores = {}
        for name, score in metrics.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("OpenEvolve metric names must be non-empty strings")
            if name in ("error", "timeout"):
                continue
            if isinstance(score, bool):
                if name == self.objective_metric:
                    raise TypeError("OpenEvolve objective must be numeric, not boolean")
                continue
            if not isinstance(score, Real):
                raise TypeError(f"OpenEvolve metric {name!r} must be numeric")
            try:
                value = float(score)
            except OverflowError:
                value = math.inf
            if not math.isfinite(value):
                failure = "OpenEvolve evaluation returned a non-finite score"
            scores[name] = value

        if not failure and self.objective_metric not in scores:
            raise ValueError(
                f"OpenEvolve evaluation is missing objective {self.objective_metric!r}"
            )
        return self._evaluation(scores, failure)


def adapt_openevolve_evaluator(
    evaluator: Callable,
    *,
    objective_metric: str,
    failure_score: float = -1e12,
) -> Callable[[Any], dict]:
    """Wrap a synchronous path-based evaluator for AlphaEvolve's dict contract.

    Accept a dictionary or SDK-style single-file Python candidate. Write its
    UTF-8 source to a private temporary program.py, call evaluator(path), and
    clean up on success or failure. Candidate path metadata is ignored.

    Export numeric metrics in scores.scores; omit boolean diagnostic flags and
    the reserved error/timeout fields. Require objective_metric on success.
    Exceptions, explicit error/timeout results and non-finite scores map all
    available metrics plus the objective to failure_score with generic feedback.
    Choose a finite failure_score below every valid score for the problem.
    Malformed candidates/return contracts raise rather than being guessed.

    Artifacts are not exported (a warning contains no artifact names/values).
    This does not configure the service's optimization objective, enforce a
    timeout, sandbox execution, emulate the SDK or export a cloud-run bundle.
    The trusted evaluator retains its existing privileges and dependencies.
    No Google SDK, client or authentication is imported or created here.
    """
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    if inspect.iscoroutinefunction(evaluator) or inspect.iscoroutinefunction(
        getattr(evaluator, "__call__", None)
    ):
        raise TypeError("OpenEvolve evaluator must be synchronous")
    if not isinstance(objective_metric, str) or not objective_metric.strip():
        raise ValueError("objective_metric must be a non-empty string")
    if objective_metric in ("error", "timeout"):
        raise ValueError("error and timeout are reserved for failure handling")
    if isinstance(failure_score, bool) or not isinstance(failure_score, Real):
        raise ValueError("failure_score must be finite")
    try:
        failure_score = float(failure_score)
    except OverflowError:
        raise ValueError("failure_score must be finite") from None
    if not math.isfinite(failure_score):
        raise ValueError("failure_score must be finite")
    return _OpenEvolveEvaluator(evaluator, objective_metric, failure_score)
