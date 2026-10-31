"""Legacy model weights must retain the ensemble's weighted-selection semantics."""

import tempfile
import unittest
from pathlib import Path

import yaml

from openevolve.config import Config, LLMConfig
from openevolve.llm.base import LLMInterface
from openevolve.llm.ensemble import LLMEnsemble


class RecordingLLM(LLMInterface):
    def __init__(self, model, calls):
        self.model = model
        self.calls = calls

    async def generate(self, prompt, **kwargs):
        self.calls.append(self.model)
        return self.model

    async def generate_with_context(self, system_message, messages, **kwargs):
        self.calls.append(self.model)
        return self.model


class TestPrimaryModelWeights(unittest.IsolatedAsyncioTestCase):
    async def assert_weighted_generation(self, config, expected_weight):
        self.assertEqual([model.name for model in config.models], ["primary", "secondary"])
        calls = []

        def create_client(model_config):
            return RecordingLLM(model_config.name, calls)

        for model in config.models:
            model.init_client = create_client

        ensemble = LLMEnsemble(config.models)
        # Seed 1 draws 0.134... first. With a positive primary weight of 0.25
        # or 1.0 it selects primary; a zero weight must always select secondary.
        expected_model = "secondary" if expected_weight == 0 else "primary"
        result = await ensemble.generate_with_context(
            "system", [{"role": "user", "content": "prompt"}]
        )
        self.assertEqual(result, expected_model)
        self.assertEqual(calls, [expected_model])
        self.assertEqual([model.weight for model in config.models], [expected_weight, 1.0])
        self.assertEqual(
            [model.weight for model in config.evaluator_models], [expected_weight, 1.0]
        )

    async def test_constructor_preserves_primary_weight(self):
        for weight in (None, 0.0, 0.25):
            with self.subTest(weight=weight):
                config = LLMConfig(
                    primary_model="primary",
                    primary_model_weight=weight,
                    secondary_model="secondary",
                    secondary_model_weight=1.0,
                    random_seed=1,
                )
                await self.assert_weighted_generation(config, 1.0 if weight is None else weight)

    async def test_yaml_preserves_primary_weight(self):
        for weight in (None, 0.0, 0.25):
            with self.subTest(weight=weight):
                llm = {
                    "primary_model": "primary",
                    "secondary_model": "secondary",
                    "secondary_model_weight": 1.0,
                    "random_seed": 1,
                }
                if weight is not None:
                    llm["primary_model_weight"] = weight
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "config.yaml"
                    path.write_text(yaml.safe_dump({"llm": llm}), encoding="utf-8")
                    config = Config.from_yaml(path).llm
                await self.assert_weighted_generation(config, 1.0 if weight is None else weight)

    async def test_rebuild_preserves_primary_weight(self):
        for weight in (None, 0.0, 0.25):
            with self.subTest(weight=weight):
                config = LLMConfig(random_seed=1)
                config.primary_model = "primary"
                config.primary_model_weight = weight
                config.secondary_model = "secondary"
                config.secondary_model_weight = 1.0
                config.rebuild_models()
                await self.assert_weighted_generation(config, 1.0 if weight is None else weight)


if __name__ == "__main__":
    unittest.main()
