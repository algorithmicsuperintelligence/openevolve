"""
Tests for early stopping configuration and behavior.
"""

import unittest

from openevolve.config import Config


class TestEarlyStoppingConfigDefaults(unittest.TestCase):
    """Tests for early stopping configuration defaults"""

    def test_patience_default_is_none(self):
        """Test that early_stopping_patience defaults to None (disabled)"""
        config = Config()
        self.assertIsNone(config.early_stopping_patience)

    def test_convergence_threshold_default(self):
        """Test that convergence_threshold defaults to 0.001"""
        config = Config()
        self.assertEqual(config.convergence_threshold, 0.001)

    def test_metric_default(self):
        """Test that early_stopping_metric defaults to combined_score"""
        config = Config()
        self.assertEqual(config.early_stopping_metric, "combined_score")


class TestEarlyStoppingConfigFromDict(unittest.TestCase):
    """Tests for loading early stopping config from dict"""

    def test_custom_patience(self):
        """Test setting custom early_stopping_patience"""
        config_dict = {
            "llm": {"primary_model": "gpt-4"},
            "early_stopping_patience": 50,
        }
        config = Config.from_dict(config_dict)
        self.assertEqual(config.early_stopping_patience, 50)

    def test_custom_convergence_threshold(self):
        """Test setting custom convergence_threshold"""
        config_dict = {
            "llm": {"primary_model": "gpt-4"},
            "convergence_threshold": 0.01,
        }
        config = Config.from_dict(config_dict)
        self.assertEqual(config.convergence_threshold, 0.01)

    def test_custom_metric(self):
        """Test setting custom early_stopping_metric"""
        config_dict = {
            "llm": {"primary_model": "gpt-4"},
            "early_stopping_metric": "score",
        }
        config = Config.from_dict(config_dict)
        self.assertEqual(config.early_stopping_metric, "score")

    def test_all_early_stopping_options(self):
        """Test setting all early stopping options together"""
        config_dict = {
            "llm": {"primary_model": "gpt-4"},
            "early_stopping_patience": 100,
            "convergence_threshold": 0.005,
            "early_stopping_metric": "validity",
        }
        config = Config.from_dict(config_dict)
        self.assertEqual(config.early_stopping_patience, 100)
        self.assertEqual(config.convergence_threshold, 0.005)
        self.assertEqual(config.early_stopping_metric, "validity")

    def test_zero_patience_disables_early_stopping(self):
        """Test that patience=0 effectively disables early stopping"""
        config_dict = {
            "llm": {"primary_model": "gpt-4"},
            "early_stopping_patience": 0,
        }
        config = Config.from_dict(config_dict)
        self.assertEqual(config.early_stopping_patience, 0)

    def test_negative_patience_allowed(self):
        """Test that negative patience is allowed (but probably shouldn't be used)"""
        config_dict = {
            "llm": {"primary_model": "gpt-4"},
            "early_stopping_patience": -1,
        }
        # Should not raise an error during loading
        config = Config.from_dict(config_dict)
        self.assertEqual(config.early_stopping_patience, -1)


class TestEarlyStoppingWithYaml(unittest.TestCase):
    """Tests for early stopping config from YAML"""

    def test_config_to_dict_includes_early_stopping(self):
        """Test that to_dict includes early stopping settings"""
        config = Config()
        config_dict = config.to_dict()

        self.assertIn("early_stopping_patience", config_dict)
        self.assertIn("convergence_threshold", config_dict)
        self.assertIn("early_stopping_metric", config_dict)


if __name__ == "__main__":
    unittest.main()
