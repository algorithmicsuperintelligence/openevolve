"""
Tests for visualization data sanitization.
Ensures -inf, +inf, and NaN values are properly sanitized for JSON serialization.
"""

import json
import math
import os
import sys
import unittest


class TestCheckJsonFloat(unittest.TestCase):
    """Tests for the check_json_float helper function"""

    @classmethod
    def setUpClass(cls):
        """Add scripts directory to path for importing visualizer"""
        cls.scripts_path = os.path.join(
            os.path.dirname(__file__), "..", "scripts"
        )
        sys.path.insert(0, cls.scripts_path)

    @classmethod
    def tearDownClass(cls):
        """Remove scripts directory from path"""
        if cls.scripts_path in sys.path:
            sys.path.remove(cls.scripts_path)

    def test_valid_positive_float(self):
        """Test that positive floats are valid"""
        try:
            from visualizer import check_json_float
            self.assertTrue(check_json_float(1.5))
            self.assertTrue(check_json_float(100.0))
            self.assertTrue(check_json_float(0.001))
        except ImportError:
            self.skipTest("Visualizer module not available")

    def test_valid_negative_float(self):
        """Test that negative floats are valid"""
        try:
            from visualizer import check_json_float
            self.assertTrue(check_json_float(-1.5))
            self.assertTrue(check_json_float(-100.0))
        except ImportError:
            self.skipTest("Visualizer module not available")

    def test_valid_zero(self):
        """Test that zero is valid"""
        try:
            from visualizer import check_json_float
            self.assertTrue(check_json_float(0))
            self.assertTrue(check_json_float(0.0))
        except ImportError:
            self.skipTest("Visualizer module not available")

    def test_valid_integer(self):
        """Test that integers are valid"""
        try:
            from visualizer import check_json_float
            self.assertTrue(check_json_float(1))
            self.assertTrue(check_json_float(-5))
            self.assertTrue(check_json_float(1000))
        except ImportError:
            self.skipTest("Visualizer module not available")

    def test_invalid_positive_infinity(self):
        """Test that positive infinity is invalid"""
        try:
            from visualizer import check_json_float
            self.assertFalse(check_json_float(float('inf')))
        except ImportError:
            self.skipTest("Visualizer module not available")

    def test_invalid_negative_infinity(self):
        """Test that negative infinity is invalid"""
        try:
            from visualizer import check_json_float
            self.assertFalse(check_json_float(float('-inf')))
        except ImportError:
            self.skipTest("Visualizer module not available")

    def test_invalid_nan(self):
        """Test that NaN is invalid"""
        try:
            from visualizer import check_json_float
            self.assertFalse(check_json_float(float('nan')))
        except ImportError:
            self.skipTest("Visualizer module not available")

    def test_invalid_none(self):
        """Test that None is invalid (not a number)"""
        try:
            from visualizer import check_json_float
            self.assertFalse(check_json_float(None))
        except ImportError:
            self.skipTest("Visualizer module not available")


class TestSanitizeProgramForVisualization(unittest.TestCase):
    """Tests for the sanitize_program_for_visualization function"""

    @classmethod
    def setUpClass(cls):
        """Add scripts directory to path for importing visualizer"""
        cls.scripts_path = os.path.join(
            os.path.dirname(__file__), "..", "scripts"
        )
        sys.path.insert(0, cls.scripts_path)

    @classmethod
    def tearDownClass(cls):
        """Remove scripts directory from path"""
        if cls.scripts_path in sys.path:
            sys.path.remove(cls.scripts_path)

    def test_sanitize_negative_infinity_in_metrics(self):
        """Test that -inf in metrics is sanitized to None"""
        try:
            from visualizer import sanitize_program_for_visualization

            program = {
                "metrics": {"combined_score": float('-inf')},
                "metadata": {}
            }
            sanitize_program_for_visualization(program)
            self.assertIsNone(program["metrics"]["combined_score"])
        except ImportError:
            self.skipTest("Visualizer module not available")

    def test_sanitize_positive_infinity_in_metrics(self):
        """Test that +inf in metrics is sanitized to None"""
        try:
            from visualizer import sanitize_program_for_visualization

            program = {
                "metrics": {"score": float('inf')},
                "metadata": {}
            }
            sanitize_program_for_visualization(program)
            self.assertIsNone(program["metrics"]["score"])
        except ImportError:
            self.skipTest("Visualizer module not available")

    def test_sanitize_nan_in_metrics(self):
        """Test that NaN in metrics is sanitized to None"""
        try:
            from visualizer import sanitize_program_for_visualization

            program = {
                "metrics": {"validity": float('nan')},
                "metadata": {}
            }
            sanitize_program_for_visualization(program)
            self.assertIsNone(program["metrics"]["validity"])
        except ImportError:
            self.skipTest("Visualizer module not available")

    def test_valid_metrics_unchanged(self):
        """Test that valid metrics are not changed"""
        try:
            from visualizer import sanitize_program_for_visualization

            program = {
                "metrics": {"score": 0.85, "validity": 1.0},
                "metadata": {}
            }
            sanitize_program_for_visualization(program)
            self.assertEqual(program["metrics"]["score"], 0.85)
            self.assertEqual(program["metrics"]["validity"], 1.0)
        except ImportError:
            self.skipTest("Visualizer module not available")

    def test_sanitize_parent_metrics(self):
        """Test that parent_metrics are also sanitized"""
        try:
            from visualizer import sanitize_program_for_visualization

            program = {
                "metrics": {"score": 0.5},
                "metadata": {
                    "parent_metrics": {
                        "score": float('inf'),
                        "other": 1.0,
                    }
                }
            }
            sanitize_program_for_visualization(program)
            self.assertIsNone(program["metadata"]["parent_metrics"]["score"])
            self.assertEqual(program["metadata"]["parent_metrics"]["other"], 1.0)
        except ImportError:
            self.skipTest("Visualizer module not available")

    def test_mixed_valid_and_invalid_values(self):
        """Test sanitization with mix of valid and invalid values"""
        try:
            from visualizer import sanitize_program_for_visualization

            program = {
                "metrics": {
                    "score": 0.5,
                    "combined_score": float('-inf'),
                    "validity": float('nan'),
                    "eval_time": 1.23,
                },
                "metadata": {}
            }
            sanitize_program_for_visualization(program)

            self.assertEqual(program["metrics"]["score"], 0.5)
            self.assertIsNone(program["metrics"]["combined_score"])
            self.assertIsNone(program["metrics"]["validity"])
            self.assertEqual(program["metrics"]["eval_time"], 1.23)
        except ImportError:
            self.skipTest("Visualizer module not available")


class TestJsonSerialization(unittest.TestCase):
    """Tests for JSON serialization of sanitized data"""

    def test_sanitized_data_is_json_serializable(self):
        """Test that sanitized data can be JSON serialized"""
        data = {
            "metrics": {
                "score": None,  # Sanitized from -inf
                "other": 1.0,
            }
        }
        # This should not raise an error
        json_str = json.dumps(data)
        self.assertIn('"score": null', json_str)

    def test_unsanitized_infinity_fails_strict_json(self):
        """Test that unsanitized infinity fails strict JSON serialization"""
        data = {"score": float('inf')}
        # With allow_nan=False (strict JSON compliance), this should fail
        with self.assertRaises(ValueError):
            json.dumps(data, allow_nan=False)

    def test_unsanitized_nan_fails_strict_json(self):
        """Test that unsanitized NaN fails strict JSON serialization"""
        data = {"score": float('nan')}
        # With allow_nan=False (strict JSON compliance), this should fail
        with self.assertRaises(ValueError):
            json.dumps(data, allow_nan=False)

    def test_sanitized_none_works_with_strict_json(self):
        """Test that sanitized None values work with strict JSON"""
        data = {"score": None}  # Properly sanitized
        # Should work with strict JSON
        json_str = json.dumps(data, allow_nan=False)
        self.assertIn('"score": null', json_str)


if __name__ == "__main__":
    unittest.main()
