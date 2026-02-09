"""
Tests for template_dir path resolution
"""

import unittest
import tempfile
import yaml
from pathlib import Path

from openevolve.config import Config


class TestTemplateDirResolution(unittest.TestCase):
    """Test that template_dir paths are resolved relative to config file location"""

    def test_relative_template_dir_resolved_to_config_location(self):
        """Relative paths in template_dir should resolve relative to config file"""
        # Create a temporary directory structure
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config_subdir"
            config_dir.mkdir()
            config_file = config_dir / "test_config.yaml"

            # Write config with relative template_dir
            config_data = {
                "prompt": {"template_dir": "templates"},
                "llm": {"models": [{"name": "gpt-4", "weight": 1.0}]},
            }
            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            # Load config
            config = Config.from_yaml(config_file)

            # Template_dir should be resolved relative to config file location
            expected_path = str((config_dir / "templates").resolve())
            self.assertEqual(config.prompt.template_dir, expected_path)

    def test_absolute_template_dir_unchanged(self):
        """Absolute paths in template_dir should remain unchanged"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "test_config.yaml"
            absolute_template_path = "/absolute/path/to/templates"

            # Write config with absolute template_dir
            config_data = {
                "prompt": {"template_dir": absolute_template_path},
                "llm": {"models": [{"name": "gpt-4", "weight": 1.0}]},
            }
            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            # Load config
            config = Config.from_yaml(config_file)

            # Absolute path should remain unchanged
            self.assertEqual(config.prompt.template_dir, absolute_template_path)

    def test_null_template_dir_unchanged(self):
        """Null template_dir should remain None"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "test_config.yaml"

            # Write config with null template_dir
            config_data = {
                "prompt": {"template_dir": None},
                "llm": {"models": [{"name": "gpt-4", "weight": 1.0}]},
            }
            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            # Load config
            config = Config.from_yaml(config_file)

            # None should remain None
            self.assertIsNone(config.prompt.template_dir)

    def test_nested_relative_template_dir(self):
        """Nested relative paths should resolve correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "configs"
            config_dir.mkdir()
            config_file = config_dir / "test_config.yaml"

            # Write config with nested relative path
            config_data = {
                "prompt": {"template_dir": "../templates/custom"},
                "llm": {"models": [{"name": "gpt-4", "weight": 1.0}]},
            }
            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            # Load config
            config = Config.from_yaml(config_file)

            # Should resolve to <tmpdir>/templates/custom
            expected_path = str((config_dir / "../templates/custom").resolve())
            self.assertEqual(config.prompt.template_dir, expected_path)

    def test_real_example_config(self):
        """Test with real example config file"""
        # This test uses the actual llm_prompt_optimization example
        config_path = "examples/llm_prompt_optimization/config.yaml"
        if not Path(config_path).exists():
            self.skipTest(f"Example config not found: {config_path}")

        config = Config.from_yaml(config_path)

        # Should resolve to examples/llm_prompt_optimization/templates
        expected_dir = Path("examples/llm_prompt_optimization/templates").resolve()
        actual_dir = Path(config.prompt.template_dir)

        self.assertEqual(actual_dir, expected_dir)
        # Verify the resolved path is absolute
        self.assertTrue(actual_dir.is_absolute())


if __name__ == "__main__":
    unittest.main()
