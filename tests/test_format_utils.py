from openevolve.utils.format_utils import format_improvement_safe, format_metrics_safe


def test_format_metrics_safe_preserves_boolean_values():
    assert format_metrics_safe({"valid": True, "score": 0.25}) == "valid=True, score=0.2500"


def test_format_improvement_safe_ignores_boolean_diffs():
    assert (
        format_improvement_safe(
            {"valid": False, "score": 0.25},
            {"valid": True, "score": 0.5},
        )
        == "score=+0.2500"
    )
