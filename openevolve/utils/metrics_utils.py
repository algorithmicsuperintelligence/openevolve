"""
Safe calculation utilities for metrics containing mixed types
"""

import math
from typing import Any, Dict, List, Optional


def safe_numeric_average(metrics: Dict[str, Any]) -> float:
    """
    Calculate the average of numeric values in a metrics dictionary,
    safely ignoring non-numeric values like strings.

    Args:
        metrics: Dictionary of metric names to values

    Returns:
        Average of numeric values, or 0.0 if no numeric values found
    """
    if not metrics:
        return 0.0

    numeric_values = []
    for value in metrics.values():
        # bool is a subclass of int, but a flag is not a score. Averaging it in
        # would let e.g. {"error": 0.0, "timeout": True} score 0.5 instead of 0.0,
        # giving a program that timed out a mid-range fitness.
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                # Convert to float and check if it's a valid number
                float_val = float(value)
                if not (float_val != float_val):  # Check for NaN (NaN != NaN is True)
                    numeric_values.append(float_val)
            except (ValueError, TypeError, OverflowError):
                # Skip invalid numeric values
                continue

    if not numeric_values:
        return 0.0

    total = sum(numeric_values)
    if not math.isfinite(total) and all(math.isfinite(value) for value in numeric_values):
        # A finite average can still have an overflowing intermediate sum.
        # Rescale only that case, retaining ordinary rounding and explicit infinities.
        scale = max(abs(value) for value in numeric_values)
        return (sum(value / scale for value in numeric_values) / len(numeric_values)) * scale
    return total / len(numeric_values)


def safe_numeric_sum(metrics: Dict[str, Any]) -> float:
    """
    Calculate the sum of numeric values in a metrics dictionary,
    safely ignoring non-numeric values like strings and boolean flags.

    In Python, `bool` is a subclass of `int` (`isinstance(True, int)` is True).
    Treating boolean flags (such as `{"timeout": True}` or `{"valid": True}`) as
    numeric scores would corrupt calculations (e.g., adding 1.0 or 0.0 to sums).
    Therefore, boolean values are explicitly excluded.

    Args:
        metrics: Dictionary of metric names to values

    Returns:
        Sum of numeric values, or 0.0 if no numeric values found
    """
    if not metrics:
        return 0.0

    numeric_sum = 0.0
    for value in metrics.values():
        # In Python, bool subclasses int, but boolean flags (e.g., {"timeout": True})
        # represent status metadata rather than numeric scores. Explicitly skip booleans
        # to avoid inflating numeric sums by 1.0 or 0.0.
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                # Convert to float and verify it is not NaN (NaN values are skipped)
                float_val = float(value)
                if not math.isnan(float_val):
                    numeric_sum += float_val
            except (ValueError, TypeError, OverflowError):
                # Skip invalid or overflowing numeric values
                continue

    return numeric_sum


def get_fitness_score(
    metrics: Dict[str, Any], feature_dimensions: Optional[List[str]] = None
) -> float:
    """
    Calculate fitness score, excluding MAP-Elites feature dimensions.

    This ensures that MAP-Elites features (like code complexity or diversity)
    do not pollute the fitness calculation when combined_score is not available.

    Special cases:
      - Boolean `combined_score` values (e.g. `False` on evaluation failure) are
        treated as non-numeric flags to prevent `float(False)` from becoming 0.0
        and falsely beating programs with valid negative scores.
      - A NaN `combined_score` yields -inf so that invalid programs rank below
        every other program.
      - If `combined_score` is absent, averages only non-feature numeric metrics.

    Args:
        metrics: All metrics from evaluation
        feature_dimensions: List of MAP-Elites dimensions to exclude from fitness

    Returns:
        Fitness score (combined_score if available, otherwise average of non-feature metrics).
        A NaN combined_score gives -inf, so that program ranks below every other.
    """
    if not metrics:
        return 0.0

    # Always prefer combined_score if available and genuinely numeric
    if "combined_score" in metrics:
        score_val = metrics["combined_score"]
        # Booleans are flags, not scores. For example, evaluators may return
        # {"combined_score": False, "error": "timeout"} on failure.
        # If treated as numeric, float(False) becomes 0.0, improperly outscoring
        # programs that received negative fitness scores (e.g., loss = -5.0).
        if not isinstance(score_val, bool):
            try:
                combined_score = float(score_val)
                # Every comparison with NaN is False in IEEE 754, which breaks sorting.
                # Assign -inf so that NaN scores rank at the absolute bottom.
                return float("-inf") if math.isnan(combined_score) else combined_score
            except (ValueError, TypeError, OverflowError):
                pass

    # Otherwise, average only non-feature metrics
    feature_dimensions = feature_dimensions or []
    fitness_metrics = {}

    for key, value in metrics.items():
        # Exclude MAP feature dimensions from fitness calculation
        if key not in feature_dimensions:
            # Booleans are flags, not scores - see the note in safe_numeric_average.
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                try:
                    float_val = float(value)
                    if not math.isnan(float_val):
                        fitness_metrics[key] = float_val
                except (ValueError, TypeError, OverflowError):
                    continue

    # If no non-feature metrics, fall back to all metrics (backward compatibility)
    if not fitness_metrics:
        return safe_numeric_average(metrics)

    return safe_numeric_average(fitness_metrics)


def format_feature_coordinates(metrics: Dict[str, Any], feature_dimensions: List[str]) -> str:
    """
    Format feature coordinates for display in prompts

    Args:
        metrics: All metrics from evaluation
        feature_dimensions: List of MAP-Elites feature dimensions

    Returns:
        Formatted string showing feature coordinates (e.g. 'valid=True, complexity=12.50')
    """
    feature_values = []
    for dim in feature_dimensions:
        if dim in metrics:
            value = metrics[dim]
            # In Python, bool subclasses int. Formatting boolean feature flags as floats
            # (e.g., 'valid=1.00') is misleading in LLM prompts. Output boolean literals.
            if isinstance(value, bool):
                feature_values.append(f"{dim}={value}")
            elif isinstance(value, (int, float)):
                try:
                    float_val = float(value)
                    if not math.isnan(float_val):
                        feature_values.append(f"{dim}={float_val:.2f}")
                    else:
                        feature_values.append(f"{dim}={value}")
                except (ValueError, TypeError, OverflowError):
                    feature_values.append(f"{dim}={value}")
            else:
                feature_values.append(f"{dim}={value}")

    if not feature_values:  # No valid feature coordinates found will return empty string
        return ""

    return ", ".join(feature_values)
