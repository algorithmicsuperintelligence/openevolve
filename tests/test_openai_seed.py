"""
Tests that the seed parameter is skipped for Google AI Studio's OpenAI-compatible
endpoint, which rejects it (#105), however api_base is spelled
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

from openevolve.llm.openai import OpenAILLM

GEMINI = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _make_llm(api_base):
    model_cfg = Mock()
    model_cfg.name = "test-model"
    model_cfg.system_message = "system"
    model_cfg.temperature = 0.7
    model_cfg.top_p = None
    model_cfg.max_tokens = 100
    model_cfg.timeout = 60
    model_cfg.retries = 0
    model_cfg.retry_delay = 0
    model_cfg.api_base = api_base
    model_cfg.api_key = None  # openai.AsyncOpenAI is patched
    model_cfg.random_seed = 42
    model_cfg.reasoning_effort = None
    with patch("openai.AsyncOpenAI"):
        llm = OpenAILLM(model_cfg)
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = "ok"
    llm.client.chat.completions.create = AsyncMock(return_value=response)
    return llm


def _sent_params(llm):
    asyncio.run(
        llm.generate_with_context(
            system_message="sys", messages=[{"role": "user", "content": "hi"}]
        )
    )
    return llm.client.chat.completions.create.call_args.kwargs


class TestSeedParameter(unittest.TestCase):
    def test_seed_is_sent_to_other_endpoints(self):
        self.assertEqual(_sent_params(_make_llm("https://api.openai.com/v1"))["seed"], 42)

    def test_seed_is_skipped_for_gemini_with_or_without_trailing_slash(self):
        for api_base in (GEMINI, GEMINI.rstrip("/")):
            with self.subTest(api_base=api_base):
                self.assertNotIn("seed", _sent_params(_make_llm(api_base)))

    def test_skip_warning_is_logged_once(self):
        llm = _make_llm(GEMINI)
        with self.assertLogs("openevolve.llm.openai", level="WARNING") as logs:
            _sent_params(llm)
            _sent_params(llm)
        self.assertEqual(sum("Skipping seed parameter" in line for line in logs.output), 1)


if __name__ == "__main__":
    unittest.main()
