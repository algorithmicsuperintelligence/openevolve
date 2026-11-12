"""
Tests that a response cut off at the token limit (finish_reason="length") is reported
instead of being passed on as if it were complete
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

from openevolve.llm.openai import OpenAILLM


def _make_llm():
    model_cfg = Mock()
    model_cfg.name = "test-model"
    model_cfg.system_message = "system"
    model_cfg.temperature = 0.7
    model_cfg.top_p = None
    model_cfg.max_tokens = 100
    model_cfg.timeout = 60
    model_cfg.retries = 0
    model_cfg.retry_delay = 0
    model_cfg.api_base = "https://example.com/v1"
    model_cfg.api_key = None  # openai.AsyncOpenAI is patched
    model_cfg.random_seed = None
    model_cfg.reasoning_effort = None
    with patch("openai.AsyncOpenAI"):
        return OpenAILLM(model_cfg)


def _response(finish_reason):
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = "Plan: keep the quiet window in onSuccess to"
    response.choices[0].finish_reason = finish_reason
    # Numbers from a real cut-off reply: most of the limit went to thinking tokens
    response.usage.prompt_tokens = 8411
    response.usage.completion_tokens = 322
    response.usage.total_tokens = 16598
    return response


class TestFinishReason(unittest.TestCase):
    def test_cut_off_response_logs_a_warning(self):
        llm = _make_llm()
        llm.client.chat.completions.create = AsyncMock(return_value=_response("length"))
        params = {"model": "test-model", "messages": [], "max_tokens": 8192}

        with self.assertLogs("openevolve.llm.openai", level="WARNING") as logs:
            asyncio.run(llm._call_api(params))
        self.assertIn("max_tokens=8192", logs.output[0])
        self.assertEqual(llm.last_usage["finish_reason"], "length")

    def test_complete_response_does_not_warn(self):
        llm = _make_llm()
        llm.client.chat.completions.create = AsyncMock(return_value=_response("stop"))
        params = {"model": "test-model", "messages": [], "max_tokens": 8192}

        with self.assertNoLogs("openevolve.llm.openai", level="WARNING"):
            asyncio.run(llm._call_api(params))
        self.assertEqual(llm.last_usage["finish_reason"], "stop")


if __name__ == "__main__":
    unittest.main()
