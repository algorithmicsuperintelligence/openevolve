"""Tests for per-run proposal-model budget configuration."""

import tempfile
import unittest
from pathlib import Path

from openevolve.config import Config, load_config


class TestLLMBudgetConfig(unittest.TestCase):
    def test_defaults_preserve_unlimited_legacy_behavior(self):
        config = Config()

        self.assertIsNone(config.max_llm_calls)
        self.assertIsNone(config.max_total_provider_tokens)
        config.validate()

    def test_dict_yaml_and_serialization_round_trip(self):
        config = Config.from_dict(
            {
                "max_llm_calls": 12,
                "max_total_provider_tokens": 240_000,
            }
        )

        self.assertEqual(config.max_llm_calls, 12)
        self.assertEqual(config.max_total_provider_tokens, 240_000)
        self.assertEqual(config.to_dict()["max_llm_calls"], 12)
        self.assertEqual(
            config.to_dict()["max_total_provider_tokens"],
            240_000,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(
                "max_llm_calls: 7\nmax_total_provider_tokens: 9000\n",
                encoding="utf-8",
            )
            loaded = load_config(config_path)

        self.assertEqual(loaded.max_llm_calls, 7)
        self.assertEqual(loaded.max_total_provider_tokens, 9000)

    def test_programmatic_validation_accepts_positive_integers(self):
        config = Config()
        config.max_llm_calls = 1
        config.max_total_provider_tokens = 1

        config.validate()

    def test_programmatic_validation_rejects_invalid_limits(self):
        for field_name in ("max_llm_calls", "max_total_provider_tokens"):
            for invalid_value in (0, -1, True, 1.5, "10"):
                with self.subTest(field_name=field_name, invalid_value=invalid_value):
                    config = Config()
                    setattr(config, field_name, invalid_value)
                    with self.assertRaisesRegex(
                        ValueError,
                        rf"^{field_name} must be a positive integer or None$",
                    ):
                        config.validate()


if __name__ == "__main__":
    unittest.main()
