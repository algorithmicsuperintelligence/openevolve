"""
Tests handling of a list of chunks returned from reasoning models
As returned by Mistral's API for reasoning models
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

from openevolve.llm.openai import OpenAILLM

MISTRAL = "https://api.mistral.ai/v1"


def _make_llm(api_base=MISTRAL, content="ok"):
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
    response.choices[0].message.content = content
    llm.client.chat.completions.create = AsyncMock(return_value=response)
    return llm


class TestListContentBlocks(unittest.TestCase):
    def test_thinking_and_text_blocks_are_joined(self):
        content = [
            {
                "type": "thinking",
                "thinking": [
                    {
                        "type": "text",
                        "text": "Internal thinking...",
                    }
                ],
                "closed": True,
            },
            {
                "type": "text",
                "text": "```python\nprint(hi)\n\n```",
            },
        ]
        llm = _make_llm(content=content)
        result = asyncio.run(
            llm.generate_with_context(
                system_message="sys", messages=[{"role": "user", "content": "hi"}]
            )
        )
        self.assertEqual(result, "```python\nprint(hi)\n\n```")

    def test_multiple_text_blocks_are_concatenated(self):
        content = [
            {
                "type": "thinking",
                "thinking": [
                    {
                        "type": "text",
                        "text": "Thinking one...",
                    }
                ],
                "closed": True,
            },
            {"type": "text", "text": "part one "},
            {
                "type": "thinking",
                "thinking": [
                    {
                        "type": "text",
                        "text": "Thinking two...",
                    }
                ],
                "closed": True,
            },
            {"type": "text", "text": "part two"},
        ]
        llm = _make_llm(content=content)
        result = asyncio.run(
            llm.generate_with_context(
                system_message="sys", messages=[{"role": "user", "content": "hi"}]
            )
        )
        self.assertEqual(result, "part one part two")

    def test_only_thinking_blocks_raises(self):
        content = [
            {
                "type": "thinking",
                "thinking": [{"type": "text", "text": "reasoning only"}],
                "closed": True,
            },
        ]
        llm = _make_llm(content=content)
        with self.assertRaises(ValueError):
            asyncio.run(
                llm.generate_with_context(
                    system_message="sys", messages=[{"role": "user", "content": "hi"}]
                )
            )

    def test_plain_string_content_is_unchanged(self):
        llm = _make_llm(content="plain response")
        result = asyncio.run(
            llm.generate_with_context(
                system_message="sys", messages=[{"role": "user", "content": "hi"}]
            )
        )
        self.assertEqual(result, "plain response")


if __name__ == "__main__":
    unittest.main()
