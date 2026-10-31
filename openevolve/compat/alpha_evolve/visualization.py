"""Score lookup for local result envelopes; no plotting dependencies."""

import math
from typing import Any

from openevolve.compat.alphaevolve import _field


def get_score(program: Any, metric_name: str) -> float:
    """Read a finite named score, or -inf for absent/invalid values.

    Accepts mappings or attribute-based envelopes. Unlike a plotting helper,
    this does not execute a candidate or import visualization libraries.
    """
    scores = _field(_field(_field(program, "evaluation"), "scores"), "scores")
    if not isinstance(scores, (list, tuple)):
        return -math.inf
    for entry in scores:
        if _field(entry, "metric") != metric_name:
            continue
        value = _field(entry, "score")
        if value is None:
            continue
        if isinstance(value, bool):
            return -math.inf
        try:
            score = float(value)
        except (TypeError, ValueError, OverflowError):
            return -math.inf
        return score if math.isfinite(score) else -math.inf
    return -math.inf
