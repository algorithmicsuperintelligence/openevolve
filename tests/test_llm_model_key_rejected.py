"""
Tests for rejection of the legacy/mistaken 'llm.model' config key (issue #427).

The LLM config has no 'model' field - models are configured via 'llm.models'
or 'llm.primary_model'. dacite silently drops unknown keys, so a config using
'model:' (as several README provider examples taught) produced an empty model
ensemble that later crashed every LLM call with "IndexError: list index out
of range". Config.from_dict must reject the key with an actionable error.
"""

import unittest

from openevolve.config import Config


class TestLLMModelKeyRejected(unittest.TestCase):
    """The 'llm.model' key must fail loudly instead of building an empty ensemble"""

    def test_model_key_raises_helpful_error(self):
        """A bare llm.model key raises ValueError pointing at the right keys"""
        with self.assertRaises(ValueError) as ctx:
            Config.from_dict({"llm": {"model": "glm-ocr"}})
        self.assertIn("primary_model", str(ctx.exception))
        self.assertIn("llm.models", str(ctx.exception))

    def test_model_key_with_other_keys_still_raises(self):
        """llm.model raises even when mixed into an otherwise valid config"""
        config_dict = {
            "max_iterations": 100,
            "llm": {
                "api_base": "http://localhost:3000/api/",
                "api_key": "test",
                "model": "glm-ocr",
                "temperature": 0.7,
            },
        }
        with self.assertRaises(ValueError) as ctx:
            Config.from_dict(config_dict)
        self.assertIn("llm.model", str(ctx.exception))

    def test_primary_model_config_still_valid(self):
        """The documented primary_model spelling keeps working"""
        config = Config.from_dict(
            {"llm": {"primary_model": "glm-ocr", "api_base": "http://localhost:3000/api/"}}
        )
        self.assertEqual(len(config.llm.models), 1)
        self.assertEqual(config.llm.models[0].name, "glm-ocr")

    def test_models_array_config_still_valid(self):
        """The models array spelling keeps working"""
        config = Config.from_dict({"llm": {"models": [{"name": "gpt-4o-mini", "weight": 1.0}]}})
        self.assertEqual(len(config.llm.models), 1)
        self.assertEqual(config.llm.models[0].name, "gpt-4o-mini")

    def test_code_constructed_config_unaffected(self):
        """Internally constructed Config() (which never goes through from_dict)
        keeps its empty default ensemble - that path is not a user error"""
        config = Config()
        self.assertEqual(config.llm.models, [])


if __name__ == "__main__":
    unittest.main()
