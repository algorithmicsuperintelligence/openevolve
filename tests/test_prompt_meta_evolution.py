"""
Tests for prompt meta-evolution in openevolve.prompt.meta_evolution
"""

import unittest

from openevolve.prompt.meta_evolution import (
    PromptTemplate,
    PromptArchive,
    evolve_prompt,
    _extract_between_tags,
)


class TestPromptTemplate(unittest.TestCase):
    """Tests for PromptTemplate dataclass"""

    def test_initial_score(self):
        """Test that new templates have neutral score"""
        template = PromptTemplate(
            id="test1",
            system_template="You are a helpful assistant.",
            user_template="Improve this code: {code}",
        )
        # With 0 uses, should return 0.5 (neutral prior)
        self.assertEqual(template.score, 0.5)

    def test_score_calculation(self):
        """Test score calculation with usage data"""
        template = PromptTemplate(
            id="test1",
            system_template="System",
            user_template="User",
            uses=10,
            successes=8,  # 80% success rate
            improvements=6,  # 60% improvement rate
            total_fitness_delta=0.5,  # avg delta = 0.05
        )

        # success_rate = 0.8
        # improvement_rate = 0.6
        # avg_fitness_delta = 0.05, normalized = min(1.0, 0.05 + 0.5) = 0.55
        # score = 0.3 * 0.8 + 0.4 * 0.6 + 0.3 * 0.55 = 0.24 + 0.24 + 0.165 = 0.645
        expected_score = 0.3 * 0.8 + 0.4 * 0.6 + 0.3 * 0.55
        self.assertAlmostEqual(template.score, expected_score, places=3)

    def test_record_use(self):
        """Test recording usage outcomes"""
        template = PromptTemplate(
            id="test1",
            system_template="System",
            user_template="User",
        )

        # Record successful improvement
        template.record_use(accepted=True, fitness_delta=0.1)
        self.assertEqual(template.uses, 1)
        self.assertEqual(template.successes, 1)
        self.assertEqual(template.improvements, 1)
        self.assertAlmostEqual(template.total_fitness_delta, 0.1)

        # Record accepted but no improvement
        template.record_use(accepted=True, fitness_delta=-0.05)
        self.assertEqual(template.uses, 2)
        self.assertEqual(template.successes, 2)
        self.assertEqual(template.improvements, 1)  # No improvement
        self.assertAlmostEqual(template.total_fitness_delta, 0.05)

        # Record rejection
        template.record_use(accepted=False, fitness_delta=0.0)
        self.assertEqual(template.uses, 3)
        self.assertEqual(template.successes, 2)
        self.assertEqual(template.improvements, 1)

    def test_serialization(self):
        """Test to_dict and from_dict"""
        template = PromptTemplate(
            id="test1",
            system_template="System message",
            user_template="User message",
            uses=5,
            successes=3,
            improvements=2,
            total_fitness_delta=0.25,
            parent_id="parent1",
            generation=1,
            metadata={"source": "test"},
        )

        data = template.to_dict()
        restored = PromptTemplate.from_dict(data)

        self.assertEqual(restored.id, template.id)
        self.assertEqual(restored.system_template, template.system_template)
        self.assertEqual(restored.user_template, template.user_template)
        self.assertEqual(restored.uses, template.uses)
        self.assertEqual(restored.successes, template.successes)
        self.assertEqual(restored.improvements, template.improvements)
        self.assertAlmostEqual(restored.total_fitness_delta, template.total_fitness_delta)
        self.assertEqual(restored.parent_id, template.parent_id)
        self.assertEqual(restored.generation, template.generation)
        self.assertEqual(restored.metadata, template.metadata)


class TestPromptArchive(unittest.TestCase):
    """Tests for PromptArchive"""

    def setUp(self):
        """Set up test archive"""
        self.archive = PromptArchive(
            max_size=5,
            min_uses_for_evolution=3,
            elite_fraction=0.4,
            exploration_rate=0.0,  # Disable exploration for deterministic tests
        )

    def test_add_template(self):
        """Test adding templates"""
        template = self.archive.add_template(
            system_template="System",
            user_template="User",
        )

        self.assertIn(template.id, self.archive.templates)
        self.assertEqual(self.archive.default_template_id, template.id)
        self.assertEqual(len(self.archive.templates), 1)

    def test_add_child_template(self):
        """Test adding child template with parent"""
        parent = self.archive.add_template(
            system_template="Parent system",
            user_template="Parent user",
        )
        child = self.archive.add_template(
            system_template="Child system",
            user_template="Child user",
            parent_id=parent.id,
        )

        self.assertEqual(child.parent_id, parent.id)
        self.assertEqual(child.generation, 1)

    def test_sample_template(self):
        """Test template sampling"""
        template = self.archive.add_template(
            system_template="System",
            user_template="User",
        )

        sampled = self.archive.sample_template()
        self.assertEqual(sampled.id, template.id)

    def test_sample_prefers_higher_score(self):
        """Test that sampling prefers higher-scoring templates"""
        # Add low-scoring template
        low = self.archive.add_template(
            system_template="Low",
            user_template="Low",
        )
        low.uses = 10
        low.successes = 1
        low.improvements = 0

        # Add high-scoring template
        high = self.archive.add_template(
            system_template="High",
            user_template="High",
        )
        high.uses = 10
        high.successes = 9
        high.improvements = 8
        high.total_fitness_delta = 1.0

        # Sample multiple times and check distribution
        high_count = 0
        for _ in range(100):
            sampled = self.archive.sample_template()
            if sampled.id == high.id:
                high_count += 1

        # High-scoring template should be sampled more often
        self.assertGreater(high_count, 50)

    def test_record_outcome(self):
        """Test recording outcomes"""
        template = self.archive.add_template(
            system_template="System",
            user_template="User",
        )

        self.archive.record_outcome(template.id, accepted=True, fitness_delta=0.1)

        self.assertEqual(template.uses, 1)
        self.assertEqual(template.successes, 1)

    def test_get_templates_for_evolution(self):
        """Test getting templates ready for evolution"""
        template1 = self.archive.add_template(
            system_template="System1",
            user_template="User1",
        )
        template1.uses = 5  # Above min_uses_for_evolution (3)

        template2 = self.archive.add_template(
            system_template="System2",
            user_template="User2",
        )
        template2.uses = 2  # Below threshold

        ready = self.archive.get_templates_for_evolution()
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].id, template1.id)

    def test_pruning(self):
        """Test that archive prunes when over capacity"""
        # Add 6 templates (max_size is 5)
        for i in range(6):
            t = self.archive.add_template(
                system_template=f"System{i}",
                user_template=f"User{i}",
            )
            t.uses = 10
            t.successes = i  # Different scores

        # Should have pruned to max_size
        self.assertEqual(len(self.archive.templates), 5)

    def test_serialization(self):
        """Test archive serialization"""
        t1 = self.archive.add_template(
            system_template="System1",
            user_template="User1",
        )
        t1.uses = 5
        t1.successes = 3

        t2 = self.archive.add_template(
            system_template="System2",
            user_template="User2",
            parent_id=t1.id,
        )

        data = self.archive.to_dict()
        restored = PromptArchive.from_dict(data)

        self.assertEqual(len(restored.templates), 2)
        self.assertEqual(restored.default_template_id, self.archive.default_template_id)
        self.assertEqual(restored.templates[t1.id].uses, 5)
        self.assertEqual(restored.templates[t2.id].parent_id, t1.id)

    def test_serialization_with_scoring_config(self):
        """Test that scoring config is preserved during serialization"""
        # Create archive with custom scoring config
        archive = PromptArchive(
            max_size=10,
            score_weight_success=0.2,
            score_weight_improvement=0.5,
            score_weight_fitness_delta=0.3,
            score_min_uses=10,
            score_neutral_prior=0.6,
        )
        archive.add_template(system_template="Test", user_template="Test")

        # Serialize and restore
        data = archive.to_dict()
        restored = PromptArchive.from_dict(data)

        # Verify scoring config is preserved
        self.assertEqual(restored.score_weight_success, 0.2)
        self.assertEqual(restored.score_weight_improvement, 0.5)
        self.assertEqual(restored.score_weight_fitness_delta, 0.3)
        self.assertEqual(restored.score_min_uses, 10)
        self.assertEqual(restored.score_neutral_prior, 0.6)

    def test_get_statistics(self):
        """Test archive statistics"""
        t1 = self.archive.add_template(
            system_template="System1",
            user_template="User1",
        )
        t1.uses = 10
        t1.successes = 8

        t2 = self.archive.add_template(
            system_template="System2",
            user_template="User2",
            parent_id=t1.id,
        )
        t2.uses = 5
        t2.successes = 2

        stats = self.archive.get_statistics()

        self.assertEqual(stats["size"], 2)
        self.assertEqual(stats["total_uses"], 15)
        self.assertEqual(stats["total_successes"], 10)
        self.assertAlmostEqual(stats["overall_success_rate"], 10 / 15)
        self.assertEqual(stats["max_generation"], 1)


class TestExtractBetweenTags(unittest.TestCase):
    """Tests for tag extraction helper"""

    def test_extract_simple(self):
        """Test simple tag extraction"""
        text = "<tag>content</tag>"
        result = _extract_between_tags(text, "tag")
        self.assertEqual(result, "content")

    def test_extract_with_whitespace(self):
        """Test extraction with whitespace"""
        text = "<tag>  content with spaces  </tag>"
        result = _extract_between_tags(text, "tag")
        self.assertEqual(result, "content with spaces")

    def test_extract_multiline(self):
        """Test multiline extraction"""
        text = """<template>
line 1
line 2
</template>"""
        result = _extract_between_tags(text, "template")
        self.assertEqual(result, "line 1\nline 2")

    def test_extract_not_found(self):
        """Test extraction when tag not found"""
        text = "no tags here"
        result = _extract_between_tags(text, "tag")
        self.assertIsNone(result)


class TestEvolvePrompt(unittest.TestCase):
    """Tests for evolve_prompt function"""

    def test_evolve_prompt_success(self):
        """Test successful prompt evolution"""
        template = PromptTemplate(
            id="test1",
            system_template="Old system",
            user_template="Old user",
            uses=10,
            successes=5,
            improvements=3,
            total_fitness_delta=0.2,
        )

        # Mock LLM that returns valid evolved templates
        def mock_llm(system: str, user: str) -> str:
            return """
Here's an improved version:

<system_template>
New improved system template
</system_template>

<user_template>
New improved user template
</user_template>

I made these changes because...
"""

        result = evolve_prompt(template, [], mock_llm)

        self.assertIsNotNone(result)
        new_system, new_user = result
        self.assertEqual(new_system, "New improved system template")
        self.assertEqual(new_user, "New improved user template")

    def test_evolve_prompt_failure(self):
        """Test prompt evolution when LLM returns invalid format"""
        template = PromptTemplate(
            id="test1",
            system_template="Old system",
            user_template="Old user",
            uses=10,
        )

        # Mock LLM that returns invalid format
        def mock_llm(system: str, user: str) -> str:
            return "This response doesn't have the expected tags"

        result = evolve_prompt(template, [], mock_llm)

        self.assertIsNone(result)

    def test_evolve_prompt_exception(self):
        """Test prompt evolution when LLM raises exception"""
        template = PromptTemplate(
            id="test1",
            system_template="Old system",
            user_template="Old user",
            uses=10,
        )

        # Mock LLM that raises exception
        def mock_llm(system: str, user: str) -> str:
            raise RuntimeError("LLM error")

        result = evolve_prompt(template, [], mock_llm)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
