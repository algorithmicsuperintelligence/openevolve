"""
Evaluation result structures for OpenEvolve
"""

import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Union


class EvaluatorRepairRequest(Exception):
    """
    Raised by a user evaluator to request an LLM-based code repair attempt.

    Raise this instead of returning a zero score when the generated code has a
    correctable error (e.g. a compilation failure).  OpenEvolve will attempt to
    repair the code using the configured LLM before recording it in the database,
    so that future evolution branches from working code rather than the broken
    original.

    Args:
        message:          Human-readable error description (shown in repair history
                          and logged).
        broken_code:      The full source that failed.  Must be the complete file,
                          not just the error region, so the repair LLM has full
                          context.
        repair_context:   Optional extra information for the repair prompt (e.g.
                          full compiler stderr, runtime traceback).  Defaults to
                          the same text as *message*.
        language:         Source-language identifier used in the prompt code fence
                          (e.g. ``"cpp"``, ``"python"``).  Defaults to
                          ``"python"``.
        fallback_metrics: Metrics dict to use if repair is disabled or all repair
                          attempts are exhausted.  Should include all feature
                          dimensions required by the MAP-Elites database set to
                          appropriate penalty values, plus ``combined_score: 0.0``.
                          When ``None``, a minimal ``{"combined_score": 0.0}`` is
                          used.
    """

    def __init__(
        self,
        message: str,
        broken_code: str,
        repair_context: str = "",
        language: str = "python",
        fallback_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__(message)
        self.broken_code = broken_code
        self.repair_context = repair_context or message
        self.language = language
        self.fallback_metrics: Dict[str, float] = fallback_metrics or {"combined_score": 0.0}


@dataclass
class EvaluationResult:
    """
    Result of program evaluation containing both metrics and optional artifacts

    This maintains backward compatibility with the existing dict[str, float] contract
    while adding a side-channel for arbitrary artifacts (text or binary data).

    IMPORTANT: For custom MAP-Elites features, metrics values must be raw continuous
    scores (e.g., actual counts, percentages, continuous measurements), NOT pre-computed
    bin indices. The database handles all binning internally using min-max scaling.

    Examples:
        ✅ Correct: {"combined_score": 0.85, "prompt_length": 1247, "execution_time": 0.234}
        ❌ Wrong:   {"combined_score": 0.85, "prompt_length": 7, "execution_time": 3}
    """

    metrics: Dict[str, float]  # mandatory - existing contract
    artifacts: Dict[str, Union[str, bytes]] = field(default_factory=dict)  # optional side-channel

    @classmethod
    def from_dict(cls, metrics: Dict[str, float]) -> "EvaluationResult":
        """Auto-wrap dict returns for backward compatibility"""
        return cls(metrics=metrics)

    def to_dict(self) -> Dict[str, float]:
        """Backward compatibility - return just metrics"""
        return self.metrics

    def has_artifacts(self) -> bool:
        """Check if this result contains any artifacts"""
        return bool(self.artifacts)

    def get_artifact_keys(self) -> list:
        """Get list of artifact keys"""
        return list(self.artifacts.keys())

    def get_artifact_size(self, key: str) -> int:
        """Get size of a specific artifact in bytes"""
        if key not in self.artifacts:
            return 0

        value = self.artifacts[key]
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        elif isinstance(value, bytes):
            return len(value)
        else:
            return len(str(value).encode("utf-8"))

    def get_total_artifact_size(self) -> int:
        """Get total size of all artifacts in bytes"""
        return sum(self.get_artifact_size(key) for key in self.artifacts.keys())
