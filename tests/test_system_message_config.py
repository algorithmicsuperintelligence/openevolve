"""
Tests for where a custom system message is read from.
The prompt sampler only reads prompt.system_message, so a message set under llm must not be dropped silently.
"""

import os
import tempfile
import unittest

import yaml

# Dummy API key so LLM ensembles initialize without a real key
os.environ.setdefault("OPENAI_API_KEY", "test")

from openevolve.config import Config, load_config


class TestSystemMessagePlacement(unittest.TestCase):
    def test_prompt_system_message_is_used_without_warning(self):
        with self.assertNoLogs("openevolve.config", level="WARNING"):
            config = Config.from_dict({"prompt": {"system_message": "From prompt."}})
        self.assertEqual(config.prompt.system_message, "From prompt.")

    def test_llm_system_message_fills_an_unset_prompt_system_message(self):
        with self.assertLogs("openevolve.config", level="WARNING") as logs:
            config = Config.from_dict({"llm": {"system_message": "From llm."}})
        self.assertEqual(config.prompt.system_message, "From llm.")
        self.assertIn("prompt.system_message", logs.output[0])

    def test_prompt_system_message_wins_when_both_are_set(self):
        with self.assertLogs("openevolve.config", level="WARNING") as logs:
            config = Config.from_dict(
                {
                    "llm": {"system_message": "From llm."},
                    "prompt": {"system_message": "From prompt."},
                }
            )
        self.assertEqual(config.prompt.system_message, "From prompt.")
        self.assertIn("llm.system_message", logs.output[0])

    def test_models_get_the_system_message_set_under_llm(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.yaml")
            with open(path, "w") as f:
                yaml.safe_dump(
                    {"llm": {"primary_model": "test-model", "system_message": "From llm."}}, f
                )
            with self.assertLogs("openevolve.config", level="WARNING"):
                config = load_config(path)
        self.assertEqual(config.llm.models[0].system_message, "From llm.")

    def test_resolving_again_does_not_warn_again(self):
        for config_dict in (
            {"llm": {"system_message": "From llm."}},
            {"llm": {"system_message": "From llm."}, "prompt": {"system_message": "From prompt."}},
        ):
            with self.assertLogs("openevolve.config", level="WARNING"):
                config = Config.from_dict(config_dict)
            with self.assertNoLogs("openevolve.config", level="WARNING"):
                config.resolve_system_message()

    def test_config_built_in_code_is_resolved_by_controller(self):
        from openevolve.controller import OpenEvolve

        config = Config()
        config.llm.system_message = "From llm."
        with tempfile.TemporaryDirectory() as tmp:
            program = os.path.join(tmp, "initial_program.py")
            with open(program, "w") as f:
                f.write("# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END\n")
            evaluator = os.path.join(tmp, "evaluator.py")
            with open(evaluator, "w") as f:
                f.write("def evaluate(program_path):\n    return {'combined_score': 0.0}\n")
            with self.assertLogs("openevolve.config", level="WARNING"):
                controller = OpenEvolve(
                    program, evaluator, config=config, output_dir=os.path.join(tmp, "out")
                )
        prompt = controller.prompt_sampler.build_prompt(
            current_program="x = 1", program_metrics={"combined_score": 0.0}, language="python"
        )
        self.assertEqual(prompt["system"], "From llm.")


if __name__ == "__main__":
    unittest.main()
