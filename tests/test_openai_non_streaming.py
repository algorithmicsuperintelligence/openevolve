"""
Tests that chat completions are requested without streaming (#436)
"""

import asyncio
import unittest
from unittest.mock import Mock, patch

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
    model_cfg.api_key = None  # openai.OpenAI is patched
    model_cfg.random_seed = None
    model_cfg.reasoning_effort = None
    with patch("openai.OpenAI"):
        return OpenAILLM(model_cfg)


class TestNonStreamingRequests(unittest.TestCase):
    def test_stream_false_is_sent(self):
        llm = _make_llm()
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.content = "ok"
        llm.client.chat.completions.create.return_value = response

        params = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}
        self.assertEqual(asyncio.run(llm._call_api(params)), "ok")
        llm.client.chat.completions.create.assert_called_once_with(**params, stream=False)
        # The caller's params are not mutated
        self.assertNotIn("stream", params)

    def test_raw_sse_string_gives_clear_error(self):
        llm = _make_llm()
        llm.client.chat.completions.create.return_value = 'data: {"object":"chat.completion.chunk"}'

        with self.assertRaisesRegex(ValueError, "streaming responses are not supported"):
            asyncio.run(llm._call_api({"model": "test-model", "messages": []}))


if __name__ == "__main__":
    unittest.main()
