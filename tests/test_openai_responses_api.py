"""Tests for OpenAI Responses API support (#405, #374)."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, Mock, patch

try:
    import httpx
except ImportError:
    import httpx2 as httpx
import openai

from openevolve.config import Config, LLMModelConfig
from openevolve.llm.openai import OpenAILLM


def _make_llm(api_type="responses", model="responses-test-model", **overrides):
    values = {
        "name": model,
        "system_message": "system",
        "temperature": 0.7,
        "top_p": None,
        "max_tokens": 100,
        "timeout": 60,
        "retries": 0,
        "retry_delay": 0,
        "api_base": "https://example.com/v1",
        "api_key": "test-key",
        "api_type": api_type,
    }
    values.update(overrides)
    with patch("openai.AsyncOpenAI"):
        return OpenAILLM(LLMModelConfig(**values))


def _response(text="ok", status="completed", incomplete_reason=None):
    response = Mock()
    response.output_text = text
    response.status = status
    response.incomplete_details = (
        None if incomplete_reason is None else Mock(reason=incomplete_reason)
    )
    response.usage.input_tokens = 11
    response.usage.output_tokens = 7
    response.usage.total_tokens = 18
    return response


class TestResponsesAPI(unittest.TestCase):
    def test_request_text_and_usage_mapping(self):
        llm = _make_llm()
        llm.client.responses.create = AsyncMock(return_value=_response())
        llm.client.chat.completions.create = AsyncMock()

        result = asyncio.run(
            llm.generate_with_context("system", [{"role": "user", "content": "hi"}])
        )

        self.assertEqual(result, "ok")
        llm.client.responses.create.assert_awaited_once_with(
            model="responses-test-model",
            instructions="system",
            input=[{"role": "user", "content": "hi"}],
            max_output_tokens=100,
            temperature=0.7,
            stream=False,
        )
        llm.client.chat.completions.create.assert_not_called()
        self.assertEqual(
            llm.last_usage,
            {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
                "model": "responses-test-model",
                "finish_reason": "stop",
            },
        )

    def test_reasoning_parameters_are_mapped(self):
        llm = _make_llm(model="gpt-5.1-codex-max", reasoning_effort="high")
        llm.client.responses.create = AsyncMock(return_value=_response())

        asyncio.run(
            llm.generate_with_context(
                "system", [{"role": "user", "content": "hi"}], verbosity="low"
            )
        )

        params = llm.client.responses.create.await_args.kwargs
        self.assertEqual(params["max_output_tokens"], 100)
        self.assertEqual(params["reasoning"], {"effort": "high"})
        self.assertEqual(params["text"], {"verbosity": "low"})
        self.assertNotIn("temperature", params)
        self.assertNotIn("reasoning_effort", params)
        self.assertNotIn("verbosity", params)

    def test_incomplete_max_output_tokens_is_reported_as_length(self):
        llm = _make_llm()
        llm.client.responses.create = AsyncMock(
            return_value=_response("partial", "incomplete", "max_output_tokens")
        )

        with self.assertLogs("openevolve.llm.openai", level="WARNING") as logs:
            result = asyncio.run(
                llm.generate_with_context("system", [{"role": "user", "content": "hi"}])
            )

        self.assertEqual(result, "partial")
        self.assertEqual(llm.last_usage["finish_reason"], "length")
        self.assertIn("max_output_tokens=100", "\n".join(logs.output))

    def test_empty_output_text_is_rejected(self):
        llm = _make_llm()
        llm.client.responses.create = AsyncMock(return_value=_response(text=""))

        with self.assertRaisesRegex(ValueError, "no text content"):
            asyncio.run(llm._call_api({"model": "responses-test-model", "messages": []}))

    def test_seed_is_not_sent_to_unsupported_responses_endpoint(self):
        llm = _make_llm(random_seed=42)
        llm.client.responses.create = AsyncMock(return_value=_response())

        with self.assertLogs("openevolve.llm.openai", level="WARNING") as logs:
            asyncio.run(llm.generate_with_context("system", [{"role": "user", "content": "hi"}]))

        self.assertNotIn("seed", llm.client.responses.create.await_args.kwargs)
        self.assertIn("does not support seed", "\n".join(logs.output))

    def test_real_sdk_uses_responses_endpoint(self):
        captured = {}

        def handle_request(request):
            captured["path"] = request.url.path
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "id": "resp_test",
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": "responses-test-model",
                    "output": [
                        {
                            "id": "msg_test",
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                        }
                    ],
                    "usage": {
                        "input_tokens": 11,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 7,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 18,
                    },
                },
            )

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle_request))
        client = openai.AsyncOpenAI(
            api_key="test-key",
            base_url="https://example.com/v1",
            http_client=http_client,
            max_retries=0,
        )
        llm = _make_llm()
        llm.client = client

        async def run_test():
            try:
                return await llm.generate_with_context(
                    "system", [{"role": "user", "content": "hi"}]
                )
            finally:
                await client.close()

        self.assertEqual(asyncio.run(run_test()), "ok")
        self.assertEqual(captured["path"], "/v1/responses")
        self.assertEqual(captured["body"]["instructions"], "system")
        self.assertEqual(captured["body"]["input"], [{"role": "user", "content": "hi"}])
        self.assertEqual(captured["body"]["max_output_tokens"], 100)
        self.assertIs(captured["body"]["stream"], False)
        self.assertEqual(llm.last_usage["total_tokens"], 18)


class TestResponsesAPIConfig(unittest.TestCase):
    def test_default_api_type_preserves_chat_completions(self):
        config = Config.from_dict({"llm": {"primary_model": "test-model"}})
        self.assertEqual(config.llm.models[0].api_type, "chat_completions")

    def test_top_level_api_type_reaches_model_and_evaluator(self):
        config = Config.from_dict({"llm": {"primary_model": "test-model", "api_type": "responses"}})
        self.assertEqual(config.llm.models[0].api_type, "responses")
        self.assertEqual(config.llm.evaluator_models[0].api_type, "responses")

    def test_explicit_per_model_api_type_wins(self):
        config = Config.from_dict(
            {"llm": {"models": [{"name": "test-model", "api_type": "responses"}]}}
        )
        self.assertEqual(config.llm.models[0].api_type, "responses")

    def test_invalid_api_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "api_type"):
            Config.from_dict({"llm": {"primary_model": "test-model", "api_type": "automatic"}})


if __name__ == "__main__":
    unittest.main()
