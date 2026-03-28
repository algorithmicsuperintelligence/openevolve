"""
Test OpenAI reasoning model detection logic

Updated to use the extracted is_reasoning_model() function instead of
duplicating detection logic locally.
"""

import unittest

from openevolve.llm.openai import OPENAI_REASONING_MODEL_PREFIXES, is_reasoning_model


class TestOpenAIReasoningModelDetection(unittest.TestCase):
    """Test that OpenAI reasoning models are correctly identified via auto-detection"""

    def test_reasoning_model_detection(self):
        """Test various model names to ensure correct reasoning model detection"""
        test_cases = [
            # Reasoning models - should return True (auto-detect)
            ("o1", True, "Base o1 model"),
            ("o1-mini", True, "o1-mini model"),
            ("o1-preview", True, "o1-preview model"),
            ("o1-mini-2025-01-31", True, "o1-mini with date"),
            ("o3", True, "Base o3 model"),
            ("o3-mini", True, "o3-mini model"),
            ("o3-pro", True, "o3-pro model"),
            ("o4-mini", True, "o4-mini model"),
            ("gpt-5", True, "Base gpt-5 model"),
            ("gpt-5-mini", True, "gpt-5-mini model"),
            ("gpt-5-nano", True, "gpt-5-nano model"),
            ("gpt-oss-120b", True, "gpt-oss-120b model"),
            ("gpt-oss-20b", True, "gpt-oss-20b model"),
            # Non-reasoning models - should return False (auto-detect)
            ("gpt-4o-mini", False, "gpt-4o-mini (not reasoning)"),
            ("gpt-4o", False, "gpt-4o (not reasoning)"),
            ("gpt-4", False, "gpt-4 (not reasoning)"),
            ("gpt-3.5-turbo", False, "gpt-3.5-turbo (not reasoning)"),
            ("claude-3", False, "Non-OpenAI model"),
            ("gemini-pro", False, "Non-OpenAI model"),
            # Case insensitivity
            ("O1-MINI", True, "Uppercase o1-mini"),
            ("GPT-5-MINI", True, "Uppercase gpt-5-mini"),
        ]

        for model_name, expected, description in test_cases:
            with self.subTest(model=model_name, desc=description):
                result = is_reasoning_model(model_name)
                self.assertEqual(
                    result,
                    expected,
                    f"Model '{model_name}' ({description}): expected {expected}, got {result}",
                )

    def test_non_openai_models_not_auto_detected(self):
        """Non-OpenAI models should not be auto-detected as reasoning models"""
        non_openai_models = [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "claude-sonnet-4-5-20250929",
            "claude-opus-4-5-20251101",
            "deepseek-r1",
        ]
        for model_name in non_openai_models:
            with self.subTest(model=model_name):
                self.assertFalse(
                    is_reasoning_model(model_name),
                    f"Non-OpenAI model '{model_name}' should not be auto-detected",
                )

    def test_explicit_override_ignores_api_base(self):
        """Explicit config_flag overrides auto-detection regardless of model origin"""
        # Even non-OpenAI models can be forced to reasoning mode
        self.assertTrue(is_reasoning_model("gemini-2.5-flash", config_flag=True))
        # Even OpenAI reasoning models can be forced to standard mode
        self.assertFalse(is_reasoning_model("o3-mini", config_flag=False))


if __name__ == "__main__":
    unittest.main()
