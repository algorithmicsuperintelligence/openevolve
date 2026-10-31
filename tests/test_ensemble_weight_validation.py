"""Ensemble weights must remain usable before any model client is initialized."""

import math
import unittest

from openevolve.config import LLMModelConfig
from openevolve.llm.ensemble import LLMEnsemble


class OfflineClient:
    def __init__(self, config):
        self.model = config.name


class TestEnsembleWeightValidation(unittest.TestCase):
    def make_configs(self, weights, init_client=OfflineClient):
        return [
            LLMModelConfig(name=f"model-{index}", weight=weight, init_client=init_client)
            for index, weight in enumerate(weights)
        ]

    def test_rejects_invalid_weights_before_initializing_clients(self):
        invalid = (
            [float("nan"), 1.0],
            [float("inf"), 1.0],
            [float("-inf"), 1.0],
            [-1.0, 2.0],
            [-1.0, -2.0],
        )
        for weights in invalid:
            with self.subTest(weights=weights):
                initialized = []

                def initialize(config):
                    initialized.append(config.name)
                    return OfflineClient(config)

                with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                    LLMEnsemble(self.make_configs(weights, initialize))
                self.assertEqual(initialized, [])

    def test_rejects_an_ensemble_with_no_positive_weight(self):
        for weights in ([0.0], [0.0, 0.0]):
            with self.subTest(weights=weights):
                initialized = []

                def initialize(config):
                    initialized.append(config.name)
                    return OfflineClient(config)

                with self.assertRaisesRegex(ValueError, "positive weight"):
                    LLMEnsemble(self.make_configs(weights, initialize))
                self.assertEqual(initialized, [])

    def test_preserves_empty_placeholder_ensembles(self):
        ensemble = LLMEnsemble([])
        self.assertEqual(ensemble.models, [])
        self.assertEqual(ensemble.weights, [])

    def test_normalizes_finite_weights_when_their_sum_overflows(self):
        configs = self.make_configs([1e308, 1e308, 0.0])
        ensemble = LLMEnsemble(configs)
        self.assertEqual(ensemble.weights, [0.5, 0.5, 0.0])
        ensemble.random_state.seed(42)
        sampled = {ensemble._sample_model().model for _ in range(30)}
        self.assertEqual(sampled, {"model-0", "model-1"})
        self.assertEqual([config.weight for config in configs], [1e308, 1e308, 0.0])

    def test_preserves_ratios_for_unequal_overflowing_weights(self):
        ensemble = LLMEnsemble(self.make_configs([1e308, 5e307, 5e307]))
        self.assertEqual(ensemble.weights, [0.5, 0.25, 0.25])
        self.assertTrue(all(math.isfinite(weight) for weight in ensemble.weights))
        self.assertEqual(sum(ensemble.weights), 1.0)

    def test_preserves_ordinary_normalization_and_disabled_models(self):
        for weights in ([1.0, 3.0, 0.0], [0.0, 1.0], [0.3, 0.3, 0.3], [1, 2, 3]):
            with self.subTest(weights=weights):
                ensemble = LLMEnsemble(self.make_configs(weights))
                self.assertEqual(ensemble.weights, [weight / sum(weights) for weight in weights])
                self.assertEqual(len(ensemble.models), len(weights))

    def test_preserves_subnormal_positive_weights(self):
        ensemble = LLMEnsemble(self.make_configs([5e-324, 5e-324]))
        self.assertEqual(ensemble.weights, [0.5, 0.5])
