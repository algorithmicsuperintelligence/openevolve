"""
Tests for optional LLM parameters (temperature, top_p).
Ensures Anthropic model compatibility where both params cannot be specified together.
"""

import unittest

from openevolve.config import Config, LLMConfig, LLMModelConfig


class TestOptionalTemperatureTopP(unittest.TestCase):
    """Tests for optional temperature and top_p parameters"""

    def test_llm_config_temperature_default(self):
        """Test that temperature defaults to 0.7 in LLMConfig"""
        config = LLMConfig()
        self.assertEqual(config.temperature, 0.7)

    def test_llm_config_top_p_default_is_none(self):
        """Test that top_p defaults to None in LLMConfig (for Anthropic compatibility)"""
        config = LLMConfig()
        self.assertIsNone(config.top_p)

    def test_model_config_temperature_none_by_default(self):
        """Test that LLMModelConfig temperature is None by default"""
        config = LLMModelConfig()
        self.assertIsNone(config.temperature)

    def test_model_config_top_p_none_by_default(self):
        """Test that LLMModelConfig top_p is None by default"""
        config = LLMModelConfig()
        self.assertIsNone(config.top_p)

    def test_type_annotation_allows_none(self):
        """Test that temperature and top_p can be set to None"""
        config = LLMModelConfig(temperature=None, top_p=None)
        self.assertIsNone(config.temperature)
        self.assertIsNone(config.top_p)

    def test_type_annotation_allows_float(self):
        """Test that temperature and top_p can be set to float values"""
        config = LLMModelConfig(temperature=0.5, top_p=0.9)
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.top_p, 0.9)


class TestConfigFromDictWithOptionalParams(unittest.TestCase):
    """Tests for loading config with optional temperature/top_p from dict"""

    def test_config_with_null_temperature_uses_default(self):
        """Test loading config with null temperature uses default"""
        config_dict = {
            "llm": {
                "primary_model": "claude-sonnet",
                "api_base": "https://api.anthropic.com/v1",
                "temperature": None,
            }
        }
        config = Config.from_dict(config_dict)
        # None is stripped, so default 0.7 is used
        self.assertEqual(config.llm.temperature, 0.7)

    def test_config_with_null_top_p(self):
        """Test loading config with null top_p"""
        config_dict = {
            "llm": {
                "primary_model": "gpt-4",
                "top_p": None,
            }
        }
        config = Config.from_dict(config_dict)
        self.assertIsNone(config.llm.top_p)

    def test_config_with_only_temperature(self):
        """Test config with only temperature set (typical for Anthropic)"""
        config_dict = {
            "llm": {
                "primary_model": "claude-sonnet",
                "temperature": 0.9,
            }
        }
        config = Config.from_dict(config_dict)
        self.assertEqual(config.llm.temperature, 0.9)
        self.assertIsNone(config.llm.top_p)

    def test_config_with_only_top_p(self):
        """Test config with only top_p set"""
        config_dict = {
            "llm": {
                "primary_model": "gpt-4",
                "temperature": None,
                "top_p": 0.95,
            }
        }
        config = Config.from_dict(config_dict)
        self.assertEqual(config.llm.top_p, 0.95)

    def test_config_with_both_params(self):
        """Test config with both temperature and top_p set (OpenAI compatible)"""
        config_dict = {
            "llm": {
                "primary_model": "gpt-4",
                "temperature": 0.8,
                "top_p": 0.9,
            }
        }
        config = Config.from_dict(config_dict)
        self.assertEqual(config.llm.temperature, 0.8)
        self.assertEqual(config.llm.top_p, 0.9)

    def test_models_inherit_optional_params(self):
        """Test that models inherit temperature/top_p from parent config"""
        config_dict = {
            "llm": {
                "primary_model": "gpt-4",
                "temperature": 0.5,
                "top_p": None,
            }
        }
        config = Config.from_dict(config_dict)
        # Check that models inherited the temperature
        for model in config.llm.models:
            self.assertEqual(model.temperature, 0.5)


if __name__ == "__main__":
    unittest.main()
