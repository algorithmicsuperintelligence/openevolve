"""
Tests that the ensemble model picked in a worker process depends on the
iteration being run, not on which worker process runs it (issue #554)
"""

import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test")

from openevolve import process_parallel as process_parallel_module
from openevolve.config import Config, LLMModelConfig
from openevolve.database import Program, ProgramDatabase
from openevolve.llm.openai import OpenAILLM
from openevolve.process_parallel import ProcessParallelController

# Worker globals set by _worker_init; restored after each test
_WORKER_GLOBALS = (
    "_worker_config",
    "_worker_evaluation_file",
    "_worker_evaluator",
    "_worker_llm_ensemble",
    "_worker_prompt_sampler",
)
_MISSING = object()
ITERATIONS = 16


class TestEnsembleWorkerSeed(unittest.TestCase):
    """With a seed set, model selection is a function of the iteration alone"""

    def setUp(self):
        self._saved_globals = {
            name: getattr(process_parallel_module, name, _MISSING) for name in _WORKER_GLOBALS
        }

        parent = Program(id="parent", code="x = 1", metrics={}, metadata={"island": 0})
        self.snapshot = {
            "programs": {"parent": parent.to_dict()},
            "artifacts": {},
            "islands": [["parent"]],
            "current_island": 0,
            "sampling_island": 0,
            "feature_dimensions": [],
        }

    def tearDown(self):
        for name, value in self._saved_globals.items():
            if value is _MISSING:
                if hasattr(process_parallel_module, name):
                    delattr(process_parallel_module, name)
            else:
                setattr(process_parallel_module, name, value)

    def _config_dict(self, seed):
        """Worker config as shipped by the controller, which seeds every model alike"""
        config = Config()
        config.llm.models = [
            LLMModelConfig(name=name, weight=0.5, api_base="http://localhost:1/v1", api_key="x")
            for name in ("fast", "strong")
        ]
        for model_cfg in config.llm.models:
            model_cfg.random_seed = seed
        controller = ProcessParallelController(
            config, "evaluator.py", ProgramDatabase(config.database)
        )
        return controller._serialize_config(config)

    def _worker(self, config_dict, iterations):
        """Run `iterations` in a fresh worker process; return the model sampled for each"""
        iterations = list(iterations)
        process_parallel_module._worker_init(config_dict, "evaluator.py", None)
        process_parallel_module._worker_prompt_sampler = MagicMock()
        process_parallel_module._worker_prompt_sampler.build_prompt.return_value = {
            "system": "sys",
            "user": "usr",
        }
        process_parallel_module._worker_evaluator = MagicMock()

        sampled = []

        async def record_model(llm, system_message, messages, **kwargs):
            sampled.append(llm.model)
            return None

        with patch.object(OpenAILLM, "generate_with_context", record_model):
            for iteration in iterations:
                process_parallel_module._run_iteration_worker(
                    iteration, self.snapshot, "parent", []
                )

        self.assertEqual(len(sampled), len(iterations))
        return sampled

    def test_workers_running_different_iterations_do_not_replay_the_same_picks(self):
        config_dict = self._config_dict(seed=123)
        even = self._worker(config_dict, range(0, 2 * ITERATIONS, 2))
        odd = self._worker(config_dict, range(1, 2 * ITERATIONS, 2))

        self.assertNotEqual(even, odd)

    def test_recycled_workers_do_not_restart_the_same_picks(self):
        config_dict = self._config_dict(seed=123)
        picks = [self._worker(config_dict, [i])[0] for i in range(ITERATIONS)]

        self.assertGreater(len(set(picks)), 1)

    def test_same_iteration_picks_same_model_on_any_worker(self):
        config_dict = self._config_dict(seed=123)
        forward = list(range(ITERATIONS))
        backward = forward[::-1]
        late_joiner = [ITERATIONS + 1, 5]

        picked_forward = dict(zip(forward, self._worker(config_dict, forward)))
        picked_backward = dict(zip(backward, self._worker(config_dict, backward)))
        picked_late = dict(zip(late_joiner, self._worker(config_dict, late_joiner)))

        self.assertEqual(picked_forward, picked_backward)
        self.assertEqual(picked_late[5], picked_forward[5])

    def test_unseeded_selection_stays_random(self):
        config_dict = self._config_dict(seed=None)
        picks = self._worker(config_dict, [7] * 64)

        # One iteration number repeated: a seeded run would pin it to a single model
        self.assertEqual(set(picks), {"fast", "strong"})


if __name__ == "__main__":
    unittest.main()
