"""
Tests for LLMEnsemble in openevolve.llm.ensemble
"""

from typing import Any, Dict, List
import unittest
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.config import LLMModelConfig
from openevolve.llm.base import LLMInterface


class TestLLMEnsemble(unittest.TestCase):
    def test_weighted_sampling(self):
        models = [
            LLMModelConfig(name="a", weight=0.0, api_key="test", api_base="http://test"),
            LLMModelConfig(name="b", weight=1.0, api_key="test", api_base="http://test"),
        ]
        ensemble = LLMEnsemble(models)
        # Should always sample model 'b'
        for _ in range(10):
            self.assertEqual(ensemble._sample_model().model, "b")

        models = [
            LLMModelConfig(name="a", weight=0.3, api_key="test", api_base="http://test"),
            LLMModelConfig(name="b", weight=0.3, api_key="test", api_base="http://test"),
            LLMModelConfig(name="c", weight=0.3, api_key="test", api_base="http://test"),
        ]
        ensemble = LLMEnsemble(models)
        # Should sample both models. Track sampled models in a set
        sampled_models = set()
        for _ in range(1000):
            sampled_models.add(ensemble._sample_model().model)
            # Cancel once we have both models
            if len(sampled_models) == len(models):
                break
        self.assertEqual(len(sampled_models), len(models))


class TestEnsembleInit(unittest.TestCase):
    class MyCustomLLM(LLMInterface):
        def __init__(self, model, some_field):
            self.model = model
            self.some_field = some_field

        async def generate(self, prompt: str, **kwargs) -> str:
            return "custom-generate"

        async def generate_with_context(
            self, system_message: str, messages: List[Dict[str, str]], **kwargs
        ) -> str:
            return "custom-generate-with-context"

    def init_custom_llm(self, model_cfg):
        return self.MyCustomLLM(model=model_cfg.name, some_field="value")

    def test_ensemble_initialization(self):
        models = [
            LLMModelConfig(name="a"),
            LLMModelConfig(name="b", init_client=self.init_custom_llm),
        ]
        ensemble = LLMEnsemble(models)
        self.assertEqual(len(ensemble.models), len(models))
        self.assertEqual(ensemble.models[0].model, "a")
        self.assertEqual(ensemble.models[1].model, "b")
        self.assertEqual(ensemble.models[1].some_field, "value")

    def test_last_model_tracks_the_sampled_model(self):
        models = [LLMModelConfig(name="a", weight=1.0), LLMModelConfig(name="b", weight=1.0)]
        ensemble = LLMEnsemble(models)
        self.assertIsNone(ensemble.last_model)
        for _ in range(20):
            index = ensemble._sample_index()
            self.assertEqual(ensemble.last_model, models[index].name)


class TestTokenUsageLogging(unittest.IsolatedAsyncioTestCase):
    async def test_openai_llm_token_usage_logging_and_propagation(self):
        from unittest.mock import AsyncMock, MagicMock
        from openevolve.llm.openai import OpenAILLM

        cfg = LLMModelConfig(
            name="gpt-4o-mini",
            api_key="test",
            api_base="http://test",
            retries=0,
            retry_delay=0.1,
            timeout=10,
        )
        llm = OpenAILLM(cfg)

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "evolved code response"
        mock_response.usage.prompt_tokens = 1234
        mock_response.usage.completion_tokens = 567
        mock_response.usage.total_tokens = 1801

        llm.client.chat.completions.create = AsyncMock(return_value=mock_response)

        with self.assertLogs("openevolve.llm.openai", level="INFO") as log_ctx:
            result = await llm.generate_with_context(
                system_message="sys",
                messages=[{"role": "user", "content": "user"}],
            )

        self.assertEqual(result, "evolved code response")
        self.assertIsNotNone(llm.last_usage)
        self.assertEqual(llm.last_usage["prompt_tokens"], 1234)
        self.assertEqual(llm.last_usage["completion_tokens"], 567)
        self.assertEqual(llm.last_usage["total_tokens"], 1801)
        self.assertEqual(llm.last_usage["model"], "gpt-4o-mini")
        self.assertTrue(
            any(
                "LLM Token Usage (gpt-4o-mini): prompt_tokens=1234, completion_tokens=567, total_tokens=1801"
                in msg
                for msg in log_ctx.output
            )
        )

        # Also verify LLMEnsemble propagates last_usage
        ensemble = LLMEnsemble([cfg])
        ensemble.models = [llm]
        await ensemble.generate_with_context(
            system_message="sys",
            messages=[{"role": "user", "content": "user"}],
        )
        self.assertEqual(ensemble.last_usage, llm.last_usage)
        self.assertEqual(ensemble.last_model, cfg.name)


class TestCustomClientSampling(unittest.IsolatedAsyncioTestCase):
    class InterfaceOnlyLLM(LLMInterface):
        def __init__(self):
            self.calls = []

        async def generate(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            return "custom answer"

        async def generate_with_context(self, system_message, messages, **kwargs):
            self.calls.append((system_message, messages, kwargs))
            return "custom context answer"

    class PropertyModelLLM(InterfaceOnlyLLM):
        @property
        def model(self):
            return "custom model"

    async def check_custom_client(self, client):
        ensemble = LLMEnsemble(
            [LLMModelConfig(name="custom model", init_client=lambda _cfg: client)]
        )
        messages = [{"role": "user", "content": "question"}]
        with self.assertLogs("openevolve.llm.ensemble", level="INFO") as logs:
            self.assertEqual(await ensemble.generate("question", temperature=0), "custom answer")
            self.assertEqual(
                await ensemble.generate_with_context("system", messages, max_tokens=8),
                "custom context answer",
            )
        self.assertEqual(
            client.calls,
            [("question", {"temperature": 0}), ("system", messages, {"max_tokens": 8})],
        )
        self.assertTrue(all("custom model" in message for message in logs.output))

    async def test_interface_only_client_can_generate(self):
        await self.check_custom_client(self.InterfaceOnlyLLM())

    async def test_property_backed_model_client_can_generate(self):
        await self.check_custom_client(self.PropertyModelLLM())


class TestUsageSum(unittest.IsolatedAsyncioTestCase):
    """take_usage() sums every call, including each model generate_all_with_context asks"""

    class UsageLLM(LLMInterface):
        def __init__(self, prompt_tokens):
            self.prompt_tokens = prompt_tokens
            self.last_usage = None

        async def generate(self, prompt, **kwargs):
            return await self.generate_with_context("", [], **kwargs)

        async def generate_with_context(self, system_message, messages, **kwargs):
            self.last_usage = {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": 1,
                "total_tokens": self.prompt_tokens + 1,
                "model": "m",
                "finish_reason": "stop",
            }
            return "answer"

    def ensemble(self, *clients):
        return LLMEnsemble(
            [
                LLMModelConfig(name=f"m{i}", init_client=lambda _cfg, c=client: c)
                for i, client in enumerate(clients)
            ]
        )

    async def test_feedback_from_every_model_is_summed(self):
        ensemble = self.ensemble(self.UsageLLM(10), self.UsageLLM(20))
        await ensemble.generate_all_with_context("system", [{"role": "user", "content": "q"}])
        self.assertEqual(
            ensemble.take_usage(),
            {"prompt_tokens": 30, "completion_tokens": 2, "total_tokens": 32, "calls": 2},
        )
        self.assertIsNone(ensemble.take_usage())

    async def test_single_calls_are_summed_until_taken(self):
        ensemble = self.ensemble(self.UsageLLM(10))
        await ensemble.generate("q")
        await ensemble.generate_with_context("system", [])
        self.assertEqual(ensemble.last_usage["prompt_tokens"], 10)
        self.assertEqual(
            ensemble.take_usage(),
            {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22, "calls": 2},
        )

    def test_add_usage(self):
        from unittest.mock import MagicMock
        from openevolve.llm.ensemble import add_usage, take_usage

        generation = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
        extra = {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8, "calls": 2}
        self.assertEqual(
            add_usage(dict(generation, finish_reason="stop"), extra),
            {
                "prompt_tokens": 16,
                "completion_tokens": 10,
                "total_tokens": 26,
                "finish_reason": "stop",
                "calls": 3,
            },
        )
        # No generation usage (a backend that reports none): only the extra calls count
        self.assertEqual(add_usage(None, extra), extra)
        # Nothing to add, or an LLM that does not sum its usage
        self.assertIs(add_usage(generation, None), generation)
        self.assertIs(add_usage(generation, take_usage(MagicMock())), generation)
        self.assertIsNone(take_usage(object()))


if __name__ == "__main__":
    unittest.main()
