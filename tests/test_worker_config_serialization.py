"""
Tests that the config sent to worker processes survives the
_serialize_config -> _worker_init round trip (issue #516)
"""

import os
import unittest
from dataclasses import fields

os.environ.setdefault("OPENAI_API_KEY", "test")

from openevolve import process_parallel as process_parallel_module
from openevolve.config import Config, EvolutionTraceConfig
from openevolve.database import ProgramDatabase
from openevolve.process_parallel import ProcessParallelController

CUSTOM_DIFF_PATTERN = r"<<<< SEARCH\n(.*?)====\n(.*?)>>>> REPLACE"

# Worker globals set by _worker_init; restored after each test
_WORKER_GLOBALS = (
    "_worker_config",
    "_worker_evaluation_file",
    "_worker_evaluator",
    "_worker_llm_ensemble",
    "_worker_prompt_sampler",
)
_MISSING = object()


class TestWorkerConfigSerialization(unittest.TestCase):
    """Config fields must reach worker processes, not fall back to defaults"""

    def setUp(self):
        self._saved_globals = {
            name: getattr(process_parallel_module, name, _MISSING) for name in _WORKER_GLOBALS
        }

        # A non-default value for every top-level field, so a field that is
        # dropped in transit shows up as a mismatch against the default
        self.config = Config()
        self.config.max_iterations = 7
        self.config.checkpoint_interval = 3
        self.config.log_level = "DEBUG"
        self.config.log_dir = "/tmp/openevolve-test-logs"
        self.config.logger_prefix = "run1"
        self.config.random_seed = 123
        self.config.language = "rust"
        self.config.file_suffix = ".rs"
        self.config.diff_based_evolution = False
        self.config.max_code_length = 1234
        self.config.enforce_evolve_blocks = True
        self.config.diff_pattern = CUSTOM_DIFF_PATTERN
        self.config.early_stopping_patience = 5
        self.config.convergence_threshold = 0.5
        self.config.early_stopping_metric = "accuracy"
        self.config.max_tasks_per_child = 4
        self.config.evolution_trace = EvolutionTraceConfig(enabled=True, format="json")

    def tearDown(self):
        for name, value in self._saved_globals.items():
            if value is _MISSING:
                if hasattr(process_parallel_module, name):
                    delattr(process_parallel_module, name)
            else:
                setattr(process_parallel_module, name, value)

    def _round_trip(self) -> Config:
        """Serialize the config as the controller does and rebuild it as a worker does"""
        controller = ProcessParallelController(
            self.config,
            "evaluator.py",
            ProgramDatabase(self.config.database),
            file_suffix=self.config.file_suffix,
        )
        config_dict = controller._serialize_config(self.config)
        process_parallel_module._worker_init(config_dict, "evaluator.py", None)
        return process_parallel_module._worker_config

    def test_enforce_evolve_blocks_reaches_worker(self):
        worker_config = self._round_trip()
        self.assertTrue(worker_config.enforce_evolve_blocks)

    def test_diff_pattern_reaches_worker(self):
        worker_config = self._round_trip()
        self.assertEqual(worker_config.diff_pattern, CUSTOM_DIFF_PATTERN)

    def test_all_top_level_fields_round_trip(self):
        """Every Config field except llm (rebuilt from its models) must match in the worker"""
        worker_config = self._round_trip()
        default_config = Config()
        nested_sections = {"prompt", "database", "evaluator"}

        for field in fields(Config):
            if field.name == "llm":
                continue
            with self.subTest(field=field.name):
                if field.name not in nested_sections:
                    # Guard: keeps this test meaningful when new fields are added
                    self.assertNotEqual(
                        getattr(self.config, field.name),
                        getattr(default_config, field.name),
                        f"set a non-default value for '{field.name}' in setUp",
                    )
                self.assertEqual(
                    getattr(worker_config, field.name), getattr(self.config, field.name)
                )


if __name__ == "__main__":
    unittest.main()
