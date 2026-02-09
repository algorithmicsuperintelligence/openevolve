"""
Tests for large codebase support via changes description.
Programs can be represented as compact change descriptions instead of full code.
"""

import unittest

from openevolve.config import Config, PromptConfig


class TestChangesDescriptionConfigDefaults(unittest.TestCase):
    """Tests for changes description configuration defaults"""

    def test_programs_as_changes_description_default_false(self):
        """Test that programs_as_changes_description defaults to False"""
        config = Config()
        self.assertFalse(config.prompt.programs_as_changes_description)

    def test_system_message_changes_description_default_none(self):
        """Test that system_message_changes_description defaults to None"""
        config = Config()
        self.assertIsNone(config.prompt.system_message_changes_description)

    def test_initial_changes_description_default_empty(self):
        """Test that initial_changes_description defaults to empty string"""
        config = Config()
        self.assertEqual(config.prompt.initial_changes_description, "")


class TestChangesDescriptionValidation(unittest.TestCase):
    """Tests for changes description validation rules"""

    def test_requires_diff_based_evolution(self):
        """Test that programs_as_changes_description requires diff_based_evolution"""
        config_dict = {
            "llm": {"primary_model": "gpt-4"},
            "diff_based_evolution": False,
            "prompt": {
                "programs_as_changes_description": True,
            }
        }
        with self.assertRaises(ValueError) as context:
            Config.from_dict(config_dict)
        self.assertIn("diff_based_evolution", str(context.exception))

    def test_works_with_diff_based_evolution_enabled(self):
        """Test that changes description works when diff_based_evolution=True"""
        config_dict = {
            "llm": {"primary_model": "gpt-4"},
            "diff_based_evolution": True,
            "prompt": {
                "programs_as_changes_description": True,
            }
        }
        config = Config.from_dict(config_dict)
        self.assertTrue(config.prompt.programs_as_changes_description)
        self.assertTrue(config.diff_based_evolution)

    def test_disabled_without_diff_based_evolution_is_ok(self):
        """Test that disabled changes description works without diff_based_evolution"""
        config_dict = {
            "llm": {"primary_model": "gpt-4"},
            "diff_based_evolution": False,
            "prompt": {
                "programs_as_changes_description": False,
            }
        }
        config = Config.from_dict(config_dict)
        self.assertFalse(config.prompt.programs_as_changes_description)


class TestChangesDescriptionFromDict(unittest.TestCase):
    """Tests for loading changes description config from dict"""

    def test_custom_system_message(self):
        """Test setting custom system_message_changes_description"""
        config_dict = {
            "llm": {"primary_model": "gpt-4"},
            "diff_based_evolution": True,
            "prompt": {
                "programs_as_changes_description": True,
                "system_message_changes_description": "You are optimizing a large codebase.",
            }
        }
        config = Config.from_dict(config_dict)
        self.assertEqual(
            config.prompt.system_message_changes_description,
            "You are optimizing a large codebase."
        )

    def test_custom_initial_description(self):
        """Test setting custom initial_changes_description"""
        config_dict = {
            "llm": {"primary_model": "gpt-4"},
            "diff_based_evolution": True,
            "prompt": {
                "programs_as_changes_description": True,
                "initial_changes_description": "Initial implementation with basic algorithm.",
            }
        }
        config = Config.from_dict(config_dict)
        self.assertEqual(
            config.prompt.initial_changes_description,
            "Initial implementation with basic algorithm."
        )

    def test_all_changes_description_options(self):
        """Test setting all changes description options together"""
        config_dict = {
            "llm": {"primary_model": "gpt-4"},
            "diff_based_evolution": True,
            "prompt": {
                "programs_as_changes_description": True,
                "system_message_changes_description": "Custom system message",
                "initial_changes_description": "Initial state description",
            }
        }
        config = Config.from_dict(config_dict)
        self.assertTrue(config.prompt.programs_as_changes_description)
        self.assertEqual(
            config.prompt.system_message_changes_description,
            "Custom system message"
        )
        self.assertEqual(
            config.prompt.initial_changes_description,
            "Initial state description"
        )


class TestPromptConfigChangesDescription(unittest.TestCase):
    """Tests for PromptConfig changes description fields"""

    def test_prompt_config_defaults(self):
        """Test PromptConfig defaults for changes description"""
        prompt_config = PromptConfig()
        self.assertFalse(prompt_config.programs_as_changes_description)
        self.assertIsNone(prompt_config.system_message_changes_description)
        self.assertEqual(prompt_config.initial_changes_description, "")

    def test_prompt_config_custom_values(self):
        """Test PromptConfig with custom changes description values"""
        prompt_config = PromptConfig(
            programs_as_changes_description=True,
            system_message_changes_description="Custom message",
            initial_changes_description="Initial state",
        )
        self.assertTrue(prompt_config.programs_as_changes_description)
        self.assertEqual(prompt_config.system_message_changes_description, "Custom message")
        self.assertEqual(prompt_config.initial_changes_description, "Initial state")


if __name__ == "__main__":
    unittest.main()
