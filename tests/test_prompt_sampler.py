"""
Tests for PromptSampler in openevolve.prompt.sampler
"""

import json
import tempfile
import unittest
from pathlib import Path

from openevolve.config import Config
from openevolve.prompt.sampler import PromptSampler


class TestPromptSampler(unittest.TestCase):
    """Tests for prompt sampler"""

    def setUp(self):
        """Set up test prompt sampler"""
        config = Config()
        self.prompt_sampler = PromptSampler(config.prompt)

    def test_build_prompt(self):
        """Test building a prompt"""
        current_program = "def test(): pass"
        parent_program = "def test(): pass"
        program_metrics = {"score": 0.5}
        previous_programs = [
            {
                "id": "prev1",
                "code": "def prev1(): pass",
                "metrics": {"score": 0.4},
            }
        ]
        top_programs = [
            {
                "id": "top1",
                "code": "def top1(): pass",
                "metrics": {"score": 0.6},
            }
        ]

        prompt = self.prompt_sampler.build_prompt(
            current_program=current_program,
            parent_program=parent_program,
            program_metrics=program_metrics,
            previous_programs=previous_programs,
            top_programs=top_programs,
        )

        self.assertIn("system", prompt)
        self.assertIn("user", prompt)
        self.assertIn("def test(): pass", prompt["user"])
        # Check that the score value appears in the prompt (either as fitness or metrics)
        self.assertIn("0.5", prompt["user"])

    def test_top_program_metrics_use_neutral_descriptions(self):
        """Report metric values without inferring that they indicate good performance."""
        cases = [
            ("score", 0.0, "0.0000"),
            ("score", -1.0, "-1.0000"),
            ("error", "evaluation failed", "evaluation failed"),
            ("latency", 10000.0, "10000.0000"),
        ]
        for name, value, formatted_value in cases:
            with self.subTest(metric=name, value=value):
                prompt = self.prompt_sampler.build_prompt(
                    top_programs=[{"id": "top1", "metrics": {name: value}}],
                )

                self.assertIn(f"Key features: Metric: {name} ({formatted_value})", prompt["user"])
                self.assertNotIn("Performs well on", prompt["user"])

    def test_top_program_explicit_key_features_are_preserved(self):
        """Keep caller-provided descriptions instead of generating metric descriptions."""
        prompt = self.prompt_sampler.build_prompt(
            top_programs=[
                {
                    "id": "top1",
                    "metrics": {"score": 0.0},
                    "key_features": ["Uses memoization", "Avoids repeated work"],
                }
            ],
        )

        self.assertIn("Key features: Uses memoization, Avoids repeated work", prompt["user"])
        self.assertNotIn("Metric: score", prompt["user"])

    def test_top_program_metrics_prefix_can_be_overridden(self):
        """Load a custom metric prefix through the configured template directory."""
        with tempfile.TemporaryDirectory() as template_dir:
            fragments_path = Path(template_dir) / "fragments.json"
            fragments_path.write_text(
                json.dumps({"top_program_metrics_prefix": "Observed:"}), encoding="utf-8"
            )
            config = Config()
            config.prompt.template_dir = template_dir
            sampler = PromptSampler(config.prompt)

            prompt = sampler.build_prompt(
                top_programs=[{"id": "top1", "metrics": {"score": 0.0}}],
            )

        self.assertIn("Key features: Observed: score (0.0000)", prompt["user"])
        self.assertNotIn("Metric: score", prompt["user"])


if __name__ == "__main__":
    unittest.main()
