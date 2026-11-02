"""
Tests for mixing diff-based iterations with full rewrites via
full_rewrite_probability (issue #116)
"""

import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test")

from openevolve import process_parallel as process_parallel_module
from openevolve.config import Config
from openevolve.database import Program
from openevolve.process_parallel import (
    EVOLUTION_MODE_DIFF,
    EVOLUTION_MODE_FULL_REWRITE,
    _choose_evolution_mode,
    _prompt_template_key,
)

PARENT_CODE = "x = 1\ny = 1"
DIFF_RESPONSE = "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE\n"
REWRITE_RESPONSE = "```python\nx = 3\ny = 3\n```"


def _config(**kwargs):
    config = Config()
    config.language = "python"
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


class TestConfigValidation(unittest.TestCase):
    def _from_dict(self, **kwargs):
        return Config.from_dict(kwargs)

    def test_default_is_zero(self):
        self.assertEqual(Config().full_rewrite_probability, 0.0)
        self.assertEqual(self._from_dict().full_rewrite_probability, 0.0)

    def test_accepts_bounds(self):
        for p in (0, 0.0, 0.2, 1, 1.0):
            self.assertEqual(
                self._from_dict(full_rewrite_probability=p).full_rewrite_probability, p
            )

    def test_rejects_out_of_range(self):
        for p in (-0.1, 1.5, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                self._from_dict(full_rewrite_probability=p)

    def test_rejects_with_diff_based_evolution_off(self):
        with self.assertRaisesRegex(ValueError, "only applies when diff_based_evolution=true"):
            self._from_dict(diff_based_evolution=False, full_rewrite_probability=0.2)
        # Zero is fine: it is the default
        self._from_dict(diff_based_evolution=False, full_rewrite_probability=0.0)

    def test_rejects_with_changes_description(self):
        with self.assertRaisesRegex(ValueError, "full_rewrite_probability=0"):
            self._from_dict(
                full_rewrite_probability=0.2,
                prompt={"programs_as_changes_description": True},
            )
        self._from_dict(prompt={"programs_as_changes_description": True})

    def test_existing_changes_description_check_still_applies(self):
        with self.assertRaisesRegex(ValueError, "requires diff_based_evolution=true"):
            self._from_dict(
                diff_based_evolution=False, prompt={"programs_as_changes_description": True}
            )

    def test_config_built_in_code_is_validated_by_controller(self):
        from openevolve.controller import OpenEvolve

        config = _config(full_rewrite_probability=2.0)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            OpenEvolve("initial_program.py", "evaluator.py", config=config)


class TestChooseEvolutionMode(unittest.TestCase):
    def test_fixed_modes(self):
        self.assertEqual(
            _choose_evolution_mode(_config(full_rewrite_probability=0.0), 1), EVOLUTION_MODE_DIFF
        )
        self.assertEqual(
            _choose_evolution_mode(_config(full_rewrite_probability=1.0), 1),
            EVOLUTION_MODE_FULL_REWRITE,
        )
        self.assertEqual(
            _choose_evolution_mode(_config(diff_based_evolution=False), 1),
            EVOLUTION_MODE_FULL_REWRITE,
        )

    def test_seeded_choice_depends_on_iteration_only(self):
        config = _config(full_rewrite_probability=0.3, random_seed=7)
        first = [_choose_evolution_mode(config, i) for i in range(200)]
        # Same iterations in reverse order give the same modes
        second = [_choose_evolution_mode(config, i) for i in reversed(range(200))][::-1]
        self.assertEqual(first, second)

    def test_does_not_touch_global_random_state(self):
        import random

        config = _config(full_rewrite_probability=0.5, random_seed=3)
        state = random.getstate()
        for i in range(10):
            _choose_evolution_mode(config, i)
        self.assertEqual(random.getstate(), state)

    def test_template_key_matches_mode(self):
        self.assertEqual(_prompt_template_key(EVOLUTION_MODE_DIFF), "diff_user")
        self.assertEqual(_prompt_template_key(EVOLUTION_MODE_FULL_REWRITE), "full_rewrite_user")

    def test_different_seeds_differ(self):
        a = [
            _choose_evolution_mode(_config(full_rewrite_probability=0.5, random_seed=1), i)
            for i in range(64)
        ]
        b = [
            _choose_evolution_mode(_config(full_rewrite_probability=0.5, random_seed=2), i)
            for i in range(64)
        ]
        self.assertNotEqual(a, b)

    def test_rate_roughly_matches_probability(self):
        config = _config(full_rewrite_probability=0.2, random_seed=42)
        modes = [_choose_evolution_mode(config, i) for i in range(5000)]
        rate = modes.count(EVOLUTION_MODE_FULL_REWRITE) / len(modes)
        self.assertAlmostEqual(rate, 0.2, delta=0.03)

    def test_unseeded_still_mixes(self):
        config = _config(full_rewrite_probability=0.5, random_seed=None)
        modes = {_choose_evolution_mode(config, i) for i in range(200)}
        self.assertEqual(modes, {EVOLUTION_MODE_DIFF, EVOLUTION_MODE_FULL_REWRITE})


class TestWorkerUsesChosenMode(unittest.TestCase):
    def setUp(self):
        parent = Program(
            id="parent", code=PARENT_CODE, metrics={"combined_score": 0.5}, metadata={"island": 0}
        )
        self.snapshot = {
            "programs": {"parent": parent.to_dict()},
            "artifacts": {},
            "islands": [["parent"]],
            "current_island": 0,
            "sampling_island": 0,
            "feature_dimensions": [],
        }
        self.evaluator = MagicMock()
        self.evaluator.evaluate_program = AsyncMock(return_value={"combined_score": 1.0})
        self.evaluator.get_pending_artifacts.return_value = None
        self.sampler = MagicMock()
        self.sampler.build_prompt.return_value = {"system": "sys", "user": "usr"}

    def _run(self, config, llm_response):
        llm = MagicMock()
        llm.generate_with_context = AsyncMock(return_value=llm_response)
        llm.last_usage = None
        with patch.multiple(
            process_parallel_module,
            create=True,
            _worker_config=config,
            _worker_llm_ensemble=llm,
            _worker_prompt_sampler=self.sampler,
            _worker_evaluator=self.evaluator,
        ):
            return process_parallel_module._run_iteration_worker(3, self.snapshot, "parent", [])

    def _prompted_diff_based(self):
        return self.sampler.build_prompt.call_args.kwargs["diff_based_evolution"]

    def test_full_rewrite_iteration(self):
        result = self._run(_config(full_rewrite_probability=1.0), REWRITE_RESPONSE)
        self.assertIsNone(result.error)
        self.assertFalse(self._prompted_diff_based())
        self.assertEqual(result.evolution_mode, EVOLUTION_MODE_FULL_REWRITE)
        child = result.child_program_dict
        self.assertEqual(child["code"], "x = 3\ny = 3")
        self.assertEqual(child["metadata"]["evolution_mode"], EVOLUTION_MODE_FULL_REWRITE)

    def test_diff_iteration(self):
        result = self._run(_config(full_rewrite_probability=0.0), DIFF_RESPONSE)
        self.assertIsNone(result.error)
        self.assertTrue(self._prompted_diff_based())
        self.assertEqual(result.evolution_mode, EVOLUTION_MODE_DIFF)
        self.assertEqual(
            result.child_program_dict["metadata"]["evolution_mode"], EVOLUTION_MODE_DIFF
        )

    def test_mode_choice_failure_is_returned_not_raised(self):
        with patch.object(
            process_parallel_module, "_choose_evolution_mode", side_effect=RuntimeError("boom")
        ):
            result = self._run(_config(full_rewrite_probability=0.5), DIFF_RESPONSE)
        self.assertEqual(result.error, "boom")
        self.assertIsNone(result.evolution_mode)
        self.evaluator.evaluate_program.assert_not_called()

    def test_mode_recorded_on_failure(self):
        # An empty response is rejected by the rewrite parser
        result = self._run(_config(full_rewrite_probability=1.0), "")
        self.assertIsNotNone(result.error)
        self.assertEqual(result.evolution_mode, EVOLUTION_MODE_FULL_REWRITE)

        result = self._run(_config(full_rewrite_probability=0.0), "no diffs here")
        self.assertIn("No valid diffs", result.error)
        self.assertEqual(result.evolution_mode, EVOLUTION_MODE_DIFF)


if __name__ == "__main__":
    unittest.main()
