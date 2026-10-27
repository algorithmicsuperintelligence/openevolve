"""Dependency-free evaluation envelopes used by local AlphaEvolve examples.

These dataclasses support attribute access and a no-argument ``model_dump``.
They do not emulate Pydantic validation, coercion, schemas or serialization
options. The evaluator adapter validates scores before OpenEvolve uses them.
"""

from dataclasses import asdict, dataclass
from typing import Any, Optional


class _Envelope:
    def model_dump(self) -> dict[str, Any]:
        """Return an independent nested dictionary, including optional None fields."""
        return asdict(self)


@dataclass
class AlphaEvolveEvaluationScore(_Envelope):
    metric: str
    score: Optional[float] = None


@dataclass
class AlphaEvolveEvaluationScores(_Envelope):
    scores: list[AlphaEvolveEvaluationScore]


@dataclass
class AlphaEvolveEvaluationInsight(_Envelope):
    label: str
    text: str


@dataclass
class AlphaEvolveEvaluationInsights(_Envelope):
    insights: list[AlphaEvolveEvaluationInsight]


@dataclass
class AlphaEvolveProgramEvaluation(_Envelope):
    scores: AlphaEvolveEvaluationScores
    insights: Optional[AlphaEvolveEvaluationInsights] = None


__all__ = [
    "AlphaEvolveEvaluationScore",
    "AlphaEvolveEvaluationScores",
    "AlphaEvolveEvaluationInsight",
    "AlphaEvolveEvaluationInsights",
    "AlphaEvolveProgramEvaluation",
]
