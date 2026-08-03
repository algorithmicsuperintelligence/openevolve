"""
Comprehensive tests for PromptSampler including inspirations and feature_dimensions
"""

import unittest
from unittest.mock import MagicMock, patch
from openevolve.config import Config
from openevolve.prompt.sampler import PromptSampler


class TestPromptSamplerComprehensive(unittest.TestCase):
    """Comprehensive tests for prompt sampler edge cases"""

    def setUp(self):
        """Set up test prompt sampler"""
        config = Config()
        # Add feature dimensions to config for testing
        config.database.feature_dimensions = ["complexity", "memory_usage"]
        self.prompt_sampler = PromptSampler(config.prompt)
        self.feature_dimensions = config.database.feature_dimensions

    def test_build_prompt_with_inspirations(self):
        """Test building a prompt with inspiration programs"""
        current_program = "def optimized(): pass"
        parent_program = "def original(): pass"
        program_metrics = {
            "combined_score": 0.85,
            "accuracy": 0.9,
            "speed": 0.8,
            "complexity": 5,
            "memory_usage": 100,
        }

        # Create inspirations with diverse characteristics
        inspirations = [
            {
                "id": "insp1",
                "code": "def fast_implementation(): pass",
                "metrics": {
                    "combined_score": 0.75,
                    "accuracy": 0.7,
                    "speed": 0.95,
                    "complexity": 3,
                    "memory_usage": 50,
                },
                "metadata": {"diverse": True},
            },
            {
                "id": "insp2",
                "code": "def memory_efficient(): pass",
                "metrics": {
                    "combined_score": 0.65,
                    "accuracy": 0.8,
                    "speed": 0.5,
                    "complexity": 7,
                    "memory_usage": 20,
                },
                "metadata": {"migrant": True},
            },
        ]

        # Build prompt with inspirations and feature_dimensions
        prompt = self.prompt_sampler.build_prompt(
            current_program=current_program,
            parent_program=parent_program,
            program_metrics=program_metrics,
            inspirations=inspirations,
            feature_dimensions=self.feature_dimensions,
        )

        # Verify prompt was built successfully
        self.assertIn("system", prompt)
        self.assertIn("user", prompt)

        # Check that inspirations are included
        self.assertIn("fast_implementation", prompt["user"])
        self.assertIn("memory_efficient", prompt["user"])

        # Verify fitness scores are calculated correctly (excluding feature dimensions)
        # The inspirations should show their fitness scores, not including complexity/memory_usage
        self.assertIn("0.75", prompt["user"])  # insp1's combined_score
        self.assertIn("0.65", prompt["user"])  # insp2's combined_score

    def test_format_inspirations_section_with_feature_dimensions(self):
        """Test _format_inspirations_section directly with feature_dimensions"""
        inspirations = [
            {
                "id": "test1",
                "code": "def test_func(): return 42",
                "metrics": {
                    "combined_score": 0.9,
                    "accuracy": 0.95,
                    "complexity": 10,  # Feature dimension
                    "memory_usage": 200,  # Feature dimension
                },
                "metadata": {"diverse": True},
            }
        ]

        # Call the method directly
        result = self.prompt_sampler._format_inspirations_section(
            inspirations, "python", feature_dimensions=["complexity", "memory_usage"]
        )

        # Should not raise NameError
        self.assertIsInstance(result, str)
        self.assertIn("test_func", result)
        self.assertIn("0.9000", result)  # The fitness score

    def test_format_inspirations_section_without_feature_dimensions(self):
        """Test _format_inspirations_section works without feature_dimensions"""
        inspirations = [
            {
                "id": "test2",
                "code": "def another_func(): pass",
                "metrics": {"score": 0.7, "time": 1.2},
                "metadata": {},
            }
        ]

        # Call without feature_dimensions (should use default of None)
        result = self.prompt_sampler._format_inspirations_section(inspirations, "python")

        self.assertIsInstance(result, str)
        self.assertIn("another_func", result)

    def test_determine_program_type_with_feature_dimensions(self):
        """Test _determine_program_type with feature_dimensions parameter"""
        program = {
            "metrics": {"combined_score": 0.85, "complexity": 5, "memory_usage": 100},
            "metadata": {},
        }

        # Test with feature_dimensions
        program_type = self.prompt_sampler._determine_program_type(
            program, feature_dimensions=["complexity", "memory_usage"]
        )

        self.assertEqual(program_type, "High-Performer")  # Based on combined_score of 0.85

    def test_extract_unique_features_calls_determine_program_type(self):
        """Test that _extract_unique_features correctly handles program_type determination"""
        program = {
            "code": "",  # Empty code to trigger default features
            "metrics": {"score": 0.5},
            "metadata": {},
        }

        # This should not raise NameError when calling _determine_program_type
        features = self.prompt_sampler._extract_unique_features(program)

        self.assertIsInstance(features, str)
        self.assertIn("approach to the problem", features)

    def test_build_prompt_with_all_optional_parameters(self):
        """Test build_prompt with all optional parameters including inspirations"""
        current_program = "def main(): pass"

        # Comprehensive test data
        previous_programs = [{"id": "prev1", "code": "def v1(): pass", "metrics": {"score": 0.3}}]
        top_programs = [
            {"id": "top1", "code": "def best(): pass", "metrics": {"combined_score": 0.95}}
        ]
        inspirations = [{"id": "insp1", "code": "def creative(): pass", "metrics": {"score": 0.6}}]

        prompt = self.prompt_sampler.build_prompt(
            current_program=current_program,
            parent_program="def parent(): pass",
            program_metrics={"combined_score": 0.7, "feature1": 10},
            previous_programs=previous_programs,
            top_programs=top_programs,
            inspirations=inspirations,
            language="python",
            evolution_round=5,
            diff_based_evolution=True,
            feature_dimensions=["feature1"],
            program_artifacts={"output": "test output"},
        )

        self.assertIn("system", prompt)
        self.assertIn("user", prompt)
        # Verify all components are included
        self.assertIn("main", prompt["user"])
        self.assertIn("best", prompt["user"])
        self.assertIn("creative", prompt["user"])

    def test_artifact_include_names_limit_and_order_prompt_context(self):
        config = Config()
        config.prompt.artifact_include_names = ["search-map.txt", "guidance"]
        sampler = PromptSampler(config.prompt)

        prompt = sampler.build_prompt(
            current_program="def main(): pass",
            program_artifacts={
                "large-receipt.json": "omit me",
                "guidance": "short direction",
                "search-map.txt": "compact map",
            },
        )

        self.assertNotIn("large-receipt.json", prompt["user"])
        self.assertNotIn("omit me", prompt["user"])
        self.assertLess(
            prompt["user"].index("search-map.txt"),
            prompt["user"].index("guidance"),
        )

    def test_fitness_calculation_consistency(self):
        """Test that fitness calculation is consistent across all methods"""
        metrics = {
            "combined_score": 0.8,
            "accuracy": 0.9,
            "speed": 0.7,
            "complexity": 5,  # Feature dimension
            "memory_usage": 100,  # Feature dimension
        }
        feature_dimensions = ["complexity", "memory_usage"]

        # Build a prompt with these metrics
        prompt = self.prompt_sampler.build_prompt(
            current_program="def test(): pass",
            program_metrics=metrics,
            inspirations=[{"id": "i1", "code": "pass", "metrics": metrics}],
            feature_dimensions=feature_dimensions,
        )

        # The fitness score should be 0.8 (combined_score), not an average including features
        self.assertIn("0.8000", prompt["user"])  # Fitness score in prompt

    def test_empty_inspirations_list(self):
        """Test that empty inspirations list doesn't break anything"""
        prompt = self.prompt_sampler.build_prompt(
            current_program="def empty(): pass",
            inspirations=[],  # Empty list
            feature_dimensions=["test_feature"],
        )

        self.assertIn("system", prompt)
        self.assertIn("user", prompt)
        # Should complete without errors

    def test_inspirations_with_missing_metrics(self):
        """Test handling of inspirations with missing or invalid metrics"""
        inspirations = [
            {
                "id": "bad1",
                "code": "def bad(): pass",
                "metrics": {},  # Empty metrics
            },
            {
                "id": "bad2",
                "code": "def worse(): pass",
                # No metrics key at all
            },
        ]

        # Should handle gracefully without errors
        result = self.prompt_sampler._format_inspirations_section(
            inspirations, "python", feature_dimensions=["test"]
        )

        self.assertIsInstance(result, str)

    def test_feature_dimensions_none_vs_empty_list(self):
        """Test that None and empty list for feature_dimensions are handled correctly"""
        program = {"metrics": {"score": 0.5}}

        # Test with None
        type_none = self.prompt_sampler._determine_program_type(program, None)

        # Test with empty list
        type_empty = self.prompt_sampler._determine_program_type(program, [])

        # Both should work and give same result
        self.assertEqual(type_none, type_empty)

    def test_feature_coordinates_formatting_in_prompt(self):
        """Test that feature coordinates are formatted correctly in the prompt"""
        metrics = {"combined_score": 0.75, "complexity": 8, "memory_usage": 150, "cpu_usage": 0.3}

        prompt = self.prompt_sampler.build_prompt(
            current_program="def test(): pass",
            program_metrics=metrics,
            feature_dimensions=["complexity", "memory_usage", "cpu_usage"],
        )

        # Check that feature coordinates are included
        user_msg = prompt["user"]
        self.assertIn("complexity", user_msg)
        self.assertIn("memory_usage", user_msg)
        self.assertIn("cpu_usage", user_msg)


if __name__ == "__main__":
    unittest.main()
