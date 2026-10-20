"""
Tests that a top-level `llm.provider` reaches the per-model configs.

Regression test: LLMConfig pushes shared settings down to each LLMModelConfig via a
`shared_config` dict, but that dict omitted `provider`. Every model therefore kept
`provider=None`, and LLMEnsemble._create_model (which routes on `provider`) sent them
all to the OpenAI backend regardless of what the config asked for. A config with
`provider: claude_code` would crash with a missing-OPENAI_API_KEY error.

Reported in https://github.com/algorithmicsuperintelligence/openevolve/pull/472
"""

import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test")

from openevolve.config import Config, LLMConfig


class TestProviderPropagation(unittest.TestCase):
    def test_top_level_provider_reaches_evolution_models(self):
        config = Config.from_dict(
            {"llm": {"provider": "claude_code", "models": [{"name": "a"}, {"name": "b"}]}}
        )
        self.assertEqual([m.provider for m in config.llm.models], ["claude_code", "claude_code"])

    def test_top_level_provider_reaches_evaluator_models(self):
        config = Config.from_dict(
            {"llm": {"provider": "claude_code", "models": [{"name": "a"}, {"name": "b"}]}}
        )
        self.assertEqual(
            [m.provider for m in config.llm.evaluator_models], ["claude_code", "claude_code"]
        )

    def test_explicit_per_model_provider_wins(self):
        """Propagation must not clobber a provider set explicitly on a model."""
        config = Config.from_dict(
            {
                "llm": {
                    "provider": "claude_code",
                    "models": [{"name": "a", "provider": "openai"}, {"name": "b"}],
                }
            }
        )
        self.assertEqual([m.provider for m in config.llm.models], ["openai", "claude_code"])

    def test_default_provider_is_none(self):
        """No provider configured -> still None, i.e. the OpenAI default path."""
        config = Config.from_dict({"llm": {"models": [{"name": "a"}]}})
        self.assertEqual([m.provider for m in config.llm.models], [None])

    def test_provider_survives_rebuild_models(self):
        """rebuild_models() re-derives the shared config; it must not drop provider."""
        llm = LLMConfig(provider="claude_code")
        llm.rebuild_models()
        for model in llm.models + llm.evaluator_models:
            self.assertEqual(model.provider, "claude_code")


class TestEnsembleRoutesOnProvider(unittest.TestCase):
    """End-to-end: the ensemble must actually construct the provider's backend."""

    def test_ensemble_builds_provider_backend(self):
        from openevolve.llm.ensemble import LLMEnsemble

        config = Config.from_dict(
            {"llm": {"provider": "claude_code", "models": [{"name": "sonnet"}]}}
        )
        ensemble = LLMEnsemble(config.llm.models)
        # Before the fix this silently produced an OpenAILLM.
        self.assertEqual(type(ensemble.models[0]).__name__, "ClaudeCodeLLM")

    def test_ensemble_defaults_to_openai_without_provider(self):
        from openevolve.llm.ensemble import LLMEnsemble

        config = Config.from_dict({"llm": {"models": [{"name": "gpt-4"}]}})
        ensemble = LLMEnsemble(config.llm.models)
        self.assertEqual(type(ensemble.models[0]).__name__, "OpenAILLM")


if __name__ == "__main__":
    unittest.main()
