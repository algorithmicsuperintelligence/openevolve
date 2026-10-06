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

        async def generate_with_context(self, system_message: str, messages: List[Dict[str, str]], **kwargs) -> str:
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


class TestTokenUsageLogging(unittest.IsolatedAsyncioTestCase):
    async def test_openai_llm_token_usage_logging_and_propagation(self):
        from unittest.mock import MagicMock
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

        llm.client.chat.completions.create = MagicMock(return_value=mock_response)

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


if __name__ == "__main__":
    unittest.main()

