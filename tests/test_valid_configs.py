"""
Confirming the validity of configuration files in project directories
"""

import os
import unittest
import itertools
from unittest.mock import MagicMock, patch

from openevolve.config import Config, load_config


class TestConfigValidity(unittest.TestCase):
    """Tests that all config files in the configs/ and examples/ directories are valid"""

    def collect_files(self):
        """Collect all config/*config*.yaml and examples/**/*config*.yaml files"""
        config_dir = os.path.join(os.path.dirname(__file__), "../configs")
        examples_dir = os.path.join(os.path.dirname(__file__), "../examples")
        config_files = []
        for root, _, files in itertools.chain(os.walk(config_dir), os.walk(examples_dir)):
            for file in files:
                if "config" in file and file.endswith(".yaml"):
                    config_files.append(os.path.join(root, file))
        return config_files

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key-for-validation"})
    def test_import_config_files(self):
        """Attempt to import all config files"""
        config_files = self.collect_files()
        for config_file in config_files:
            print(f"Testing config file: {config_file}")
            config = load_config(config_file)
            self.assertIsInstance(
                config, Config, f"Config file {config_file} did not load correctly"
            )
            self.assertTrue(len(config.llm.models) > 0)

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key-for-validation"})
    def test_template_configs_name_served_gemini_models(self):
        """The configs/ templates name no Gemini model Google has shut down

        Shutdown dates: https://ai.google.dev/gemini-api/docs/deprecations
        """
        shut_down = {
            "gemini-2.0-flash",
            "gemini-2.0-flash-001",
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash-lite-001",
        }
        config_dir = os.path.join(os.path.dirname(__file__), "../configs")
        for file in sorted(os.listdir(config_dir)):
            if not file.endswith(".yaml"):
                continue
            config = load_config(os.path.join(config_dir, file))
            for model in config.llm.models + config.llm.evaluator_models:
                self.assertNotIn(model.name, shut_down, f"{file} names {model.name}")


if __name__ == "__main__":
    unittest.main()
