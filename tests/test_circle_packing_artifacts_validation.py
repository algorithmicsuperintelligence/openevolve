"""Regression tests for numeric validation in the artifact packing evaluator."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

EVALUATOR_PATH = (
    Path(__file__).resolve().parent.parent
    / "examples"
    / "circle_packing_with_artifacts"
    / "evaluator.py"
)


def load_evaluator():
    spec = importlib.util.spec_from_file_location("circle_packing_artifacts", EVALUATOR_PATH)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    return evaluator


def valid_packing():
    """26 separated circles with a hand-checked radius sum of 0.65."""
    centers = np.array([(0.1 + 0.15 * (i % 6), 0.1 + 0.15 * (i // 6)) for i in range(26)])
    return centers, np.full(26, 0.025)


class TestCirclePackingArtifactsValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = load_evaluator()

    def test_rejects_nonfinite_centers(self):
        for coordinate in (0, 1):
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(coordinate=coordinate, value=value):
                    centers, radii = valid_packing()
                    centers[0, coordinate] = value

                    valid, details = self.evaluator.validate_packing(centers, radii)

                    self.assertFalse(valid)
                    self.assertFalse(details["is_valid"])
                    self.assertIn("invalid_values", details)
                    diagnostic = " ".join(details["invalid_values"]).lower()
                    self.assertIn("centers", diagnostic)
                    self.assertIn("finite", diagnostic)

    def test_rejects_nonfinite_radii(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                centers, radii = valid_packing()
                radii[0] = value

                valid, details = self.evaluator.validate_packing(centers, radii)

                self.assertFalse(valid)
                self.assertFalse(details["is_valid"])
                self.assertIn("invalid_values", details)
                diagnostic = " ".join(details["invalid_values"]).lower()
                self.assertIn("radii", diagnostic)
                self.assertIn("finite", diagnostic)

    def test_rejects_negative_radii(self):
        centers, radii = valid_packing()
        radii[0] = -0.025

        valid, details = self.evaluator.validate_packing(centers, radii)

        self.assertFalse(valid)
        self.assertFalse(details["is_valid"])
        self.assertIn("invalid_values", details)
        diagnostic = " ".join(details["invalid_values"]).lower()
        self.assertIn("radii", diagnostic)
        self.assertIn("non-negative", diagnostic)

    def test_accepts_finite_nonoverlapping_circles(self):
        valid, details = self.evaluator.validate_packing(*valid_packing())

        self.assertTrue(valid)
        self.assertTrue(details["is_valid"])
        self.assertEqual(details["total_circles"], 26)
        self.assertEqual(details["boundary_violations"], [])
        self.assertEqual(details["overlaps"], [])
        self.assertAlmostEqual(details["min_radius"], 0.025)
        self.assertAlmostEqual(details["max_radius"], 0.025)
        self.assertAlmostEqual(details["avg_radius"], 0.025)

    def test_accepts_zero_radius(self):
        centers, radii = valid_packing()
        radii[0] = 0.0

        valid, details = self.evaluator.validate_packing(centers, radii)

        self.assertTrue(valid)
        self.assertTrue(details["is_valid"])
        self.assertEqual(details["min_radius"], 0.0)

    def test_reports_boundary_violations(self):
        centers, radii = valid_packing()
        centers[0, 0] = 0.0

        valid, details = self.evaluator.validate_packing(centers, radii)

        self.assertFalse(valid)
        self.assertTrue(details["boundary_violations"])
        self.assertEqual(details["overlaps"], [])

    def test_reports_overlaps(self):
        centers, radii = valid_packing()
        centers[0] = centers[1]

        valid, details = self.evaluator.validate_packing(centers, radii)

        self.assertFalse(valid)
        self.assertEqual(details["boundary_violations"], [])
        self.assertTrue(details["overlaps"])

    def test_evaluation_paths_reject_nonfinite_centers(self):
        for coordinate in (0, 1):
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(coordinate=coordinate, value=value):
                    centers, radii = valid_packing()
                    centers[0, coordinate] = value
                    self.assert_invalid_evaluation(centers, radii, "centers", "finite")

    def test_evaluation_paths_reject_nonfinite_radii(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                centers, radii = valid_packing()
                radii[0] = value
                self.assert_invalid_evaluation(centers, radii, "radii", "finite")

    def test_evaluation_paths_reject_negative_radii(self):
        centers, radii = valid_packing()
        radii[0] = -0.025
        self.assert_invalid_evaluation(centers, radii, "radii", "non-negative")

    def assert_invalid_evaluation(self, centers, radii, *diagnostic_terms):
        for name, failure_stage in (
            ("evaluate", "geometric_validation"),
            ("evaluate_stage1", "stage1_geometric_validation"),
        ):
            with self.subTest(evaluator=name):
                # Only replace candidate execution; validation and scoring remain real.
                with patch.object(
                    self.evaluator,
                    "run_with_timeout",
                    return_value=(centers, radii, 0.65),
                ):
                    result = getattr(self.evaluator, name)("candidate.py")

                self.assertEqual(result.artifacts.get("failure_stage"), failure_stage)
                for metric in (
                    "validity",
                    "sum_radii",
                    "target_ratio",
                    "combined_score",
                    "radius_variance",
                    "spatial_spread",
                ):
                    self.assertEqual(result.metrics[metric], 0.0)
                self.assertTrue(all(np.isfinite(value) for value in result.metrics.values()))
                self.assertIn("invalid_values", result.artifacts)
                for term in diagnostic_terms:
                    self.assertIn(term, result.artifacts["invalid_values"].lower())

    def test_evaluation_paths_score_valid_packings(self):
        for name in ("evaluate", "evaluate_stage1"):
            for first_radius, expected_sum in ((0.025, 0.65), (0.0, 0.625)):
                with self.subTest(evaluator=name, first_radius=first_radius):
                    centers, radii = valid_packing()
                    radii[0] = first_radius
                    with patch.object(
                        self.evaluator,
                        "run_with_timeout",
                        return_value=(centers, radii, expected_sum),
                    ):
                        result = getattr(self.evaluator, name)("candidate.py")

                    self.assertEqual(result.metrics["validity"], 1.0)
                    self.assertAlmostEqual(result.metrics["sum_radii"], expected_sum)
                    self.assertAlmostEqual(result.metrics["target_ratio"], expected_sum / 2.635)
                    self.assertAlmostEqual(result.metrics["combined_score"], expected_sum / 2.635)
                    self.assertGreater(result.metrics["spatial_spread"], 0.0)
                    self.assertTrue(all(np.isfinite(value) for value in result.metrics.values()))
                    self.assertNotIn("failure_stage", result.artifacts)

    def test_evaluation_paths_preserve_geometry_diagnostics(self):
        for name, failure_stage, boundary_key, overlap_key in (
            ("evaluate", "geometric_validation", "boundary_violations", "overlap_violations"),
            ("evaluate_stage1", "stage1_geometric_validation", "boundary_issues", "overlap_issues"),
        ):
            for defect in ("boundary", "overlap"):
                with self.subTest(evaluator=name, defect=defect):
                    centers, radii = valid_packing()
                    if defect == "boundary":
                        centers[0, 0] = 0.0
                        diagnostic_key = boundary_key
                    else:
                        centers[0] = centers[1]
                        diagnostic_key = overlap_key
                    with patch.object(
                        self.evaluator,
                        "run_with_timeout",
                        return_value=(centers, radii, 0.65),
                    ):
                        result = getattr(self.evaluator, name)("candidate.py")

                    self.assertEqual(result.artifacts.get("failure_stage"), failure_stage)
                    self.assertEqual(result.metrics["validity"], 0.0)
                    self.assertEqual(result.metrics["combined_score"], 0.0)
                    self.assertTrue(result.artifacts[diagnostic_key])


if __name__ == "__main__":
    unittest.main()
