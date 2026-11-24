"""Tests for the Amazon Bedrock (Converse API) LLM backend."""

import asyncio
import os
import unittest
from unittest.mock import MagicMock, patch

from openevolve.config import Config, LLMModelConfig
from openevolve.llm.bedrock import BedrockLLM, init_bedrock_client
from openevolve.llm.ensemble import _create_model


def _ok_response(text="hello", stop="end_turn"):
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": stop,
        "usage": {"inputTokens": 11, "outputTokens": 7, "totalTokens": 18},
    }


class TestBedrockLLM(unittest.TestCase):
    def _cfg(self, **overrides):
        cfg = LLMModelConfig(name=overrides.pop("name", "anthropic.test-model-v1:0"))
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def _llm(self, client=None, **overrides):
        client = client or MagicMock()
        return BedrockLLM(self._cfg(**overrides), client=client), client

    def test_requires_model_name(self):
        with self.assertRaises(ValueError):
            BedrockLLM(LLMModelConfig(name=None), client=MagicMock())

    def test_defaults(self):
        llm, _ = self._llm()
        self.assertEqual(llm.max_tokens, 4096)
        self.assertEqual(llm.retries, 3)
        self.assertIsNone(llm.endpoint_url)

    def test_region_from_config_then_env(self):
        llm, _ = self._llm(aws_region="eu-west-1")
        self.assertEqual(llm.region, "eu-west-1")
        with patch.dict(os.environ, {"AWS_REGION": "eu-central-1"}, clear=False):
            llm, _ = self._llm()
            self.assertEqual(llm.region, "eu-central-1")

    def test_env_var_references_are_resolved(self):
        env = {"OE_TEST_BEDROCK_MODEL": "anthropic.from-env-v1:0", "OE_TEST_REGION": "sa-east-1"}
        with patch.dict(os.environ, env, clear=False):
            llm, _ = self._llm(name="${OE_TEST_BEDROCK_MODEL}", aws_region="${OE_TEST_REGION}")
        self.assertEqual(llm.model, "anthropic.from-env-v1:0")
        self.assertEqual(llm.region, "sa-east-1")

    def test_missing_env_var_raises(self):
        os.environ.pop("OE_TEST_UNSET_MODEL", None)
        with self.assertRaises(ValueError):
            self._llm(name="${OE_TEST_UNSET_MODEL}")

    def test_openai_api_base_is_ignored_but_bedrock_endpoint_is_used(self):
        llm, _ = self._llm(api_base="https://api.openai.com/v1")
        self.assertIsNone(llm.endpoint_url)
        url = "https://bedrock-runtime.us-east-1.amazonaws.com"
        llm, _ = self._llm(api_base=url)
        self.assertEqual(llm.endpoint_url, url)

    def test_request_shape_and_text(self):
        client = MagicMock()
        client.converse.return_value = _ok_response("improved code")
        llm, _ = self._llm(client=client, max_tokens=1234, temperature=0.5)
        out = asyncio.run(
            llm.generate_with_context(
                "you are helpful",
                [
                    {"role": "user", "content": "a"},
                    {"role": "user", "content": "b"},
                    {"role": "assistant", "content": "c"},
                ],
            )
        )
        self.assertEqual(out, "improved code")
        req = client.converse.call_args.kwargs
        self.assertEqual(req["modelId"], "anthropic.test-model-v1:0")
        self.assertEqual(req["system"], [{"text": "you are helpful"}])
        # consecutive user turns are merged so roles alternate
        self.assertEqual([m["role"] for m in req["messages"]], ["user", "assistant"])
        self.assertEqual(req["messages"][0]["content"][0]["text"], "a\n\nb")
        self.assertEqual(req["inferenceConfig"], {"maxTokens": 1234, "temperature": 0.5})

    def test_unset_sampling_params_are_not_sent(self):
        client = MagicMock()
        client.converse.return_value = _ok_response()
        llm, _ = self._llm(client=client)
        asyncio.run(llm.generate("hi"))
        inference = client.converse.call_args.kwargs["inferenceConfig"]
        self.assertNotIn("temperature", inference)
        self.assertNotIn("topP", inference)

    def test_rejected_temperature_is_dropped_and_remembered(self):
        client = MagicMock()
        client.converse.side_effect = [
            Exception(
                "An error occurred (ValidationException) when calling the Converse operation: "
                "`temperature` is deprecated for this model."
            ),
            _ok_response("ok"),
            _ok_response("again"),
        ]
        llm, _ = self._llm(client=client, temperature=0.7, retries=0)
        self.assertEqual(asyncio.run(llm.generate("hi")), "ok")
        self.assertNotIn("temperature", client.converse.call_args.kwargs["inferenceConfig"])
        # a later call does not send temperature at all
        self.assertEqual(asyncio.run(llm.generate("hi")), "again")
        self.assertNotIn("temperature", client.converse.call_args.kwargs["inferenceConfig"])
        self.assertEqual(client.converse.call_count, 3)

    def test_other_validation_errors_are_raised(self):
        client = MagicMock()
        client.converse.side_effect = Exception(
            "ValidationException: The provided model identifier is invalid."
        )
        llm, _ = self._llm(client=client, temperature=0.7, retries=0)
        with self.assertRaises(Exception):
            asyncio.run(llm.generate("hi"))
        self.assertEqual(client.converse.call_count, 1)

    def test_usage_is_recorded(self):
        client = MagicMock()
        client.converse.return_value = _ok_response()
        llm, _ = self._llm(client=client)
        asyncio.run(llm.generate("hi"))
        self.assertEqual(llm.last_usage["prompt_tokens"], 11)
        self.assertEqual(llm.last_usage["completion_tokens"], 7)
        self.assertEqual(llm.last_usage["total_tokens"], 18)

    def test_reasoning_blocks_are_skipped(self):
        client = MagicMock()
        resp = _ok_response()
        resp["output"]["message"]["content"] = [
            {"reasoningContent": {"reasoningText": {"text": "thinking..."}}},
            {"text": "answer"},
        ]
        client.converse.return_value = resp
        llm, _ = self._llm(client=client)
        self.assertEqual(asyncio.run(llm.generate("hi")), "answer")

    def test_retries_then_succeeds(self):
        client = MagicMock()
        client.converse.side_effect = [RuntimeError("throttled"), _ok_response("ok")]
        llm, _ = self._llm(client=client, retries=1, retry_delay=0)
        self.assertEqual(asyncio.run(llm.generate("hi")), "ok")
        self.assertEqual(client.converse.call_count, 2)

    def test_raises_after_retries(self):
        client = MagicMock()
        client.converse.side_effect = RuntimeError("boom")
        llm, _ = self._llm(client=client, retries=1, retry_delay=0)
        with self.assertRaises(RuntimeError):
            asyncio.run(llm.generate("hi"))

    def test_empty_response_raises(self):
        client = MagicMock()
        client.converse.return_value = _ok_response(text="")
        llm, _ = self._llm(client=client, retries=0)
        with self.assertRaises(RuntimeError):
            asyncio.run(llm.generate("hi"))

    def test_factory_and_registry(self):
        with patch("openevolve.llm.bedrock.BedrockLLM._make_client", return_value=MagicMock()):
            self.assertIsInstance(init_bedrock_client(self._cfg()), BedrockLLM)
            cfg = self._cfg(provider="bedrock")
            self.assertIsInstance(_create_model(cfg), BedrockLLM)

    def test_top_level_provider_and_region_reach_models(self):
        config = Config.from_dict(
            {
                "llm": {
                    "provider": "bedrock",
                    "aws_region": "us-west-2",
                    "models": [{"name": "anthropic.test-model-v1:0", "weight": 1.0}],
                }
            }
        )
        model = config.llm.models[0]
        self.assertEqual(model.provider, "bedrock")
        self.assertEqual(model.aws_region, "us-west-2")


if __name__ == "__main__":
    unittest.main()
