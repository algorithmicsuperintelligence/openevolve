"""
Utility functions for formatting output
"""

from typing import Any, Dict

import numpy as np


def format_metrics_safe(metrics: Dict[str, Any]) -> str:
    """
    Safely format metrics dictionary for logging, handling both numeric and string values.

    Args:
        metrics: Dictionary of metric names to values

    Returns:
        Formatted string representation of metrics
    """
    if not metrics:
        return ""

    formatted_parts = []
    for name, value in metrics.items():
        # Check if value is numeric (int, float)
        if isinstance(value, (int, float)):
            try:
                # Only apply float formatting to numeric values
                formatted_parts.append(f"{name}={value:.4f}")
            except (ValueError, TypeError):
                # Fallback to string representation if formatting fails
                formatted_parts.append(f"{name}={value}")
        else:
            # For non-numeric values (strings, etc.), just convert to string
            formatted_parts.append(f"{name}={value}")

    return ", ".join(formatted_parts)


DTYPES = (int, float, tuple, list, np.ndarray)


def format_improvement_safe(parent_metrics: Dict[str, Any], child_metrics: Dict[str, Any]) -> str:
    """
    Safely format improvement metrics for logging.

    Args:
        parent_metrics: Parent program metrics
        child_metrics: Child program metrics

    Returns:
        Formatted string representation of improvements
    """
    if not parent_metrics or not child_metrics:
        return ""

    improvement_parts = []
    for metric, child_value in child_metrics.items():
        if metric in parent_metrics:
            parent_value = parent_metrics[metric]
            # Only calculate improvement for numeric values
            if isinstance(child_value, DTYPES) and isinstance(parent_value, DTYPES):
                try:
                    diff = np.subtract(child_value, parent_value)
                    improvement_parts.append(f"{metric}={format_score(diff)}")
                except (ValueError, TypeError):
                    # Skip non-numeric comparisons
                    continue

    return ", ".join(improvement_parts)


def format_score(x: tuple[float, ...] | list[float] | float) -> str:
    try:
        if isinstance(x, (tuple, list, np.ndarray)):
            return "(" + ", ".join(f"{float(v):.4f}" for v in x) + ")"
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return str(x)
