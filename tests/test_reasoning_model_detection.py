"""Tests for reasoning model detection logic (is_reasoning_model function)"""

import unittest

from openevolve.llm.openai import OPENAI_REASONING_MODEL_PREFIXES, is_reasoning_model


class TestIsReasoningModel(unittest.TestCase):
    """Test the is_reasoning_model() function"""

    # Auto-detect (config_flag=None) -- OpenAI models
    def test_openai_o_series_auto_detected(self):
        for model in ["o1", "o1-mini", "o3", "o3-mini", "o3-pro", "o4-mini"]:
            with self.subTest(model=model):
                self.assertTrue(is_reasoning_model(model))

    def test_openai_gpt5_auto_detected(self):
        for model in ["gpt-5", "gpt-5-mini", "gpt-5-nano"]:
            with self.subTest(model=model):
                self.assertTrue(is_reasoning_model(model))

    def test_openai_gpt_oss_auto_detected(self):
        for model in ["gpt-oss-120b", "gpt-oss-20b", "gpt-oss-30b"]:
            with self.subTest(model=model):
                self.assertTrue(is_reasoning_model(model))

    def test_openai_non_reasoning_not_detected(self):
        for model in ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]:
            with self.subTest(model=model):
                self.assertFalse(is_reasoning_model(model))

    # Auto-detect -- Non-OpenAI models should NOT be auto-detected
    def test_gemini_not_auto_detected(self):
        for model in ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"]:
            with self.subTest(model=model):
                self.assertFalse(is_reasoning_model(model))

    def test_claude_not_auto_detected(self):
        for model in ["claude-sonnet-4-5-20250929", "claude-opus-4-5-20251101"]:
            with self.subTest(model=model):
                self.assertFalse(is_reasoning_model(model))

    def test_deepseek_not_auto_detected(self):
        self.assertFalse(is_reasoning_model("deepseek-r1"))

    # Explicit config_flag=True -- forces reasoning model
    def test_explicit_true_overrides_auto_detect(self):
        self.assertTrue(is_reasoning_model("gemini-2.5-flash", config_flag=True))
        self.assertTrue(is_reasoning_model("deepseek-r1", config_flag=True))
        self.assertTrue(is_reasoning_model("any-unknown-model", config_flag=True))

    # Explicit config_flag=False -- forces non-reasoning model
    def test_explicit_false_overrides_auto_detect(self):
        # Even OpenAI reasoning models can be forced to non-reasoning
        self.assertFalse(is_reasoning_model("o3-mini", config_flag=False))
        self.assertFalse(is_reasoning_model("gpt-5", config_flag=False))

    # Case insensitivity
    def test_case_insensitive(self):
        self.assertTrue(is_reasoning_model("O3-MINI"))
        self.assertTrue(is_reasoning_model("GPT-5-MINI"))

    # Backward compatibility
    def test_none_config_flag_is_default(self):
        """None config_flag should behave exactly like the old hardcoded logic"""
        self.assertTrue(is_reasoning_model("o3-mini", config_flag=None))
        self.assertFalse(is_reasoning_model("gpt-4o", config_flag=None))


class TestReasoningModelPrefixes(unittest.TestCase):
    """Test that the prefix constant is properly defined"""

    def test_prefixes_is_tuple(self):
        self.assertIsInstance(OPENAI_REASONING_MODEL_PREFIXES, tuple)

    def test_prefixes_contains_o_series(self):
        # At minimum, o1 and o3 should be in the prefixes
        prefixes_str = " ".join(OPENAI_REASONING_MODEL_PREFIXES)
        self.assertIn("o1", prefixes_str)
        self.assertIn("o3", prefixes_str)


if __name__ == "__main__":
    unittest.main()
