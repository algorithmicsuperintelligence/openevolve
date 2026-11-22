"""
Unit tests for extended boolean metric handling and fitness score ranking.

Validates that:
- safe_numeric_sum ignores boolean flags and non-finite values.
- get_fitness_score safely handles boolean combined_score flags and NaNs.
- format_feature_coordinates renders boolean coordinates as literals.
- Evaluator._passes_threshold guards against boolean flags and NaNs.
- PromptSampler._format_evolution_history renders boolean flags as literals
  and excludes them from numeric metric improvement comparisons.
- ProcessParallelController island program ranking handles NaNs and MAP-Elites
  feature dimension exclusions consistently with database selection.
"""

import math
import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test")

from openevolve.config import PromptConfig
from openevolve.database import Program
from openevolve.evaluator import Evaluator
from openevolve.prompt.sampler import PromptSampler
from openevolve.utils.metrics_utils import (
    format_feature_coordinates,
    get_fitness_score,
    safe_numeric_average,
    safe_numeric_sum,
)


class TestSafeNumericSum(unittest.TestCase):
    """Tests for safe_numeric_sum utility function."""

    def test_bool_excluded_from_numeric_sum(self):
        """Boolean flags (True/False) should not be counted as 1.0 or 0.0 in numeric sum."""
        metrics = {"score1": 10.0, "score2": 5.0, "valid": True, "timeout": False}
        self.assertAlmostEqual(safe_numeric_sum(metrics), 15.0)

    def test_all_booleans_sum_to_zero(self):
        """A dictionary containing only boolean flags should sum to 0.0."""
        metrics = {"valid": True, "timeout": False, "passed": True}
        self.assertEqual(safe_numeric_sum(metrics), 0.0)

    def test_empty_metrics_sum_to_zero(self):
        """An empty metrics dictionary should sum to 0.0."""
        self.assertEqual(safe_numeric_sum({}), 0.0)

    def test_nan_and_invalid_values_skipped(self):
        """NaN floats and string metric values should be safely skipped."""
        metrics = {"score": 7.5, "bad": float("nan"), "str_metric": "error"}
        self.assertAlmostEqual(safe_numeric_sum(metrics), 7.5)


class TestGetFitnessScoreExtended(unittest.TestCase):
    """Tests for get_fitness_score with boolean flags, NaNs, and feature dimensions."""

    def test_boolean_combined_score_ignored(self):
        """
        Boolean combined_score must not be converted to 0.0 or 1.0.

        When evaluation fails, evaluators may emit {"combined_score": False, "error": ...}.
        If cast to float, 0.0 would improperly outscore programs with valid negative scores.
        """
        metrics_failed = {"combined_score": False, "score": -10.0}
        self.assertAlmostEqual(get_fitness_score(metrics_failed), -10.0)

        metrics_true = {"combined_score": True, "accuracy": 0.8}
        self.assertAlmostEqual(get_fitness_score(metrics_true), 0.8)

    def test_nan_combined_score_returns_neg_inf(self):
        """A NaN combined_score must return -inf so invalid programs rank at the bottom."""
        metrics = {"combined_score": float("nan"), "score": 100.0}
        self.assertEqual(get_fitness_score(metrics), float("-inf"))

    def test_feature_dimensions_excluded_when_no_combined_score(self):
        """MAP-Elites feature dimensions must not inflate fitness when combined_score is absent."""
        metrics = {"reward": 5.0, "complexity": 999.0}
        score = get_fitness_score(metrics, feature_dimensions=["complexity"])
        self.assertAlmostEqual(score, 5.0)


class TestFormatFeatureCoordinates(unittest.TestCase):
    """Tests for formatting MAP-Elites feature coordinates for prompt display."""

    def test_boolean_coordinates_formatted_as_bool(self):
        """Boolean coordinates should display as True/False rather than 1.00/0.00."""
        metrics = {"valid": True, "deterministic": False, "runtime": 0.1234}
        dims = ["valid", "deterministic", "runtime"]
        formatted = format_feature_coordinates(metrics, dims)
        self.assertIn("valid=True", formatted)
        self.assertIn("deterministic=False", formatted)
        self.assertIn("runtime=0.12", formatted)

    def test_empty_coordinates(self):
        """Empty metrics or dimensions should return an empty string."""
        self.assertEqual(format_feature_coordinates({}, ["dim1"]), "")

    def test_missing_and_nan_coordinates(self):
        """Missing dimensions should be omitted, and NaN dimensions preserved as nan."""
        metrics = {"dim1": float("nan"), "dim2": 2.5}
        formatted = format_feature_coordinates(metrics, ["dim1", "dim2", "dim3"])
        self.assertIn("dim1=nan", formatted)
        self.assertIn("dim2=2.50", formatted)
        self.assertNotIn("dim3", formatted)


class TestEvaluatorThresholdCheck(unittest.TestCase):
    """Tests for cascade threshold checks in Evaluator."""

    def setUp(self):
        self.evaluator = Evaluator.__new__(Evaluator)

    def test_boolean_combined_score_does_not_pass_threshold(self):
        """A boolean combined_score flag (False) must not pass threshold 0.0 via float conversion."""
        self.assertFalse(
            self.evaluator._passes_threshold(metrics={"combined_score": False}, threshold=0.0)
        )

    def test_nan_combined_score_does_not_pass_threshold(self):
        """A NaN combined_score must not pass threshold evaluation."""
        self.assertFalse(
            self.evaluator._passes_threshold(
                metrics={"combined_score": float("nan")}, threshold=0.0
            )
        )

    def test_valid_combined_score_passes_threshold(self):
        """Valid numeric combined_score meeting or exceeding threshold should pass."""
        self.assertTrue(
            self.evaluator._passes_threshold(metrics={"combined_score": 0.8}, threshold=0.5)
        )


class TestPromptSamplerHistoryBooleanHandling(unittest.TestCase):
    """Tests for evolution history prompt formatting and metric comparison."""

    def test_boolean_flag_flip_not_treated_as_numeric_improvement(self):
        """
        Verify boolean flag flipping False -> True is not counted as a numeric metric improvement.

        If a program's numeric score regresses from 10.0 to 5.0, flipping a flag to True
        must not cause the prompt outcome to report 'All metrics improved'.
        """
        config = PromptConfig()
        sampler = PromptSampler(config)

        # Parent and child where only a boolean flag flipped and real numeric metric regressed
        program = {
            "id": "p2",
            "metrics": {"score": 5.0, "valid": True},
            "parent_id": "p1",
            "changes_description": "Fix bug",
            "metadata": {
                "parent_metrics": {"score": 10.0, "valid": False},
            },
        }

        history_str = sampler._format_evolution_history(
            previous_programs=[program],
            top_programs=[],
            inspirations=[],
            language="python",
        )

        # Boolean metric should be rendered as boolean literal, not float
        self.assertIn("valid: True", history_str)
        self.assertNotIn("valid: 1.0000", history_str)

        # Since score regressed (5.0 < 10.0), outcome should NOT report all metrics improved!
        self.assertNotIn("All metrics improved", history_str)


class TestProcessParallelProgramRanking(unittest.TestCase):
    """Tests for program ranking in parallel workers."""

    def test_program_ranking_handles_nan_and_feature_dimensions(self):
        """
        Verify sorting island programs by get_fitness_score(p.metrics, feature_dims)
        correctly handles NaN values, negative scores, and excludes MAP-Elites feature dimensions.
        """
        p_normal = Program(
            id="p1",
            code="code1",
            metrics={"combined_score": 10.0},
        )
        p_nan = Program(
            id="p2",
            code="code2",
            metrics={"combined_score": float("nan")},
        )
        p_negative = Program(
            id="p3",
            code="code3",
            metrics={"combined_score": -100.0},
        )
        p_no_combined_high_feature = Program(
            id="p4",
            code="code4",
            metrics={"score": 2.0, "complexity": 10000.0},
        )
        p_no_combined_better_score = Program(
            id="p5",
            code="code5",
            metrics={"score": 8.0, "complexity": 10.0},
        )

        programs = [
            p_nan,
            p_negative,
            p_normal,
            p_no_combined_high_feature,
            p_no_combined_better_score,
        ]
        feature_dims = ["complexity"]

        programs.sort(
            key=lambda p: get_fitness_score(p.metrics, feature_dims),
            reverse=True,
        )

        # Expected ranking order:
        # p_normal (10.0) > p_no_combined_better_score (8.0) > p_no_combined_high_feature (2.0)
        # > p_negative (-100.0) > p_nan (-inf)
        self.assertEqual(programs[0].id, "p1")
        self.assertEqual(programs[1].id, "p5")
        self.assertEqual(programs[2].id, "p4")
        self.assertEqual(programs[3].id, "p3")
        self.assertEqual(programs[4].id, "p2")  # NaN ranked at the very bottom


if __name__ == "__main__":
    unittest.main()
