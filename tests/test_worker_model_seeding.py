"""
Tests that ensemble model sampling in worker processes is seeded per iteration
(issue #554): workers built from the same llm_seed must not all draw the same
sequence of models, and draws must not depend on which worker runs an iteration.
"""

import os
import unittest
from collections import Counter
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test")

from openevolve import process_parallel as pp
from openevolve.config import Config, LLMModelConfig
from openevolve.llm.ensemble import LLMEnsemble

LLM_SEED = 1234567


def _config(random_seed):
    config = Config()
    config.llm.models = [
        LLMModelConfig(
            name="fast", weight=0.8, api_key="test", api_base="http://test", random_seed=random_seed
        ),
        LLMModelConfig(
            name="strong",
            weight=0.2,
            api_key="test",
            api_base="http://test",
            random_seed=random_seed,
        ),
    ]
    return config


class _Worker:
    """What _worker_init/_lazy_init_worker_components leave in one worker process."""

    def __init__(self, config):
        self.config = config
        self.ensemble = LLMEnsemble(config.llm.models)

    def draw(self, iteration):
        with (
            patch.object(pp, "_worker_config", self.config, create=True),
            patch.object(pp, "_worker_llm_ensemble", self.ensemble, create=True),
        ):
            pp._seed_worker_ensemble(iteration)
            return self.ensemble._sample_model().model


def _run(n_workers, iterations, random_seed=LLM_SEED):
    """Round-robin iterations over n_workers fresh workers; model drawn per iteration."""
    config = _config(random_seed)
    workers = [_Worker(config) for _ in range(n_workers)]
    return {i: workers[k % n_workers].draw(i) for k, i in enumerate(iterations)}


class TestWorkerModelSeeding(unittest.TestCase):
    def test_draws_do_not_depend_on_worker_count_or_assignment(self):
        iterations = list(range(1, 61))
        one = _run(1, iterations)
        self.assertEqual(_run(3, iterations), one)
        self.assertEqual(_run(4, list(reversed(iterations))), one)

    def test_workers_do_not_repeat_the_same_sequence(self):
        # Before the fix every worker drew the same sequence, so with 3 workers
        # iterations 1, 2, 3 (one per worker) always got the same model.
        draws = _run(3, list(range(1, 301)))
        per_worker = [[draws[i] for i in range(w, 301, 3)] for w in (1, 2, 3)]
        self.assertNotEqual(per_worker[0], per_worker[1])
        self.assertNotEqual(per_worker[1], per_worker[2])

    def test_realized_mix_follows_weights(self):
        counts = Counter(_run(3, list(range(1, 501))).values())
        self.assertAlmostEqual(counts["fast"] / 500, 0.8, delta=0.06)

    def test_recycled_worker_continues_the_sequence(self):
        # max_tasks_per_child: a fresh worker taking over at iteration 30 draws
        # what a long-lived worker would have.
        config = _config(LLM_SEED)
        long_lived = _Worker(config)
        expected = [long_lived.draw(i) for i in range(1, 60)]
        recycled = [_Worker(config).draw(i) for i in range(1, 60)]
        self.assertEqual(recycled, expected)

    def test_unseeded_sampler_is_left_alone(self):
        worker = _Worker(_config(None))
        state = worker.ensemble.random_state.getstate()
        with (
            patch.object(pp, "_worker_config", worker.config, create=True),
            patch.object(pp, "_worker_llm_ensemble", worker.ensemble, create=True),
        ):
            pp._seed_worker_ensemble(5)
        self.assertEqual(worker.ensemble.random_state.getstate(), state)

    def test_iteration_worker_seeds_before_generating(self):
        with (
            patch.object(pp, "_lazy_init_worker_components"),
            patch.object(pp, "_seed_worker_ensemble") as seed,
        ):
            pp._run_iteration_worker(7, {"programs": {}}, "missing", [])
        seed.assert_called_once_with(7)


if __name__ == "__main__":
    unittest.main()
