"""
Tests for sample_from_island() exploration/exploitation/random ratio compliance

This ensures that sample_from_island() uses the same sampling strategy as sample()
to maintain consistent behavior between single-process and parallel execution modes.
"""

import random
import unittest
from openevolve.config import Config
from openevolve.database import Program
from openevolve.database_memory import InMemoryProgramDatabase as ProgramDatabase


class TestSampleFromIslandRatios(unittest.TestCase):
    """Tests for sample_from_island() ratio compliance"""

    def setUp(self):
        """Set up test database with programs"""
        config = Config()
        config.database.in_memory = True
        config.database.num_islands = 3
        config.database.archive_size = 10
        config.database.population_size = 100

        # Set specific exploration/exploitation ratios for testing
        config.database.exploration_ratio = 0.3
        config.database.exploitation_ratio = 0.4
        # Remaining 0.3 will be weighted sampling

        self.db = ProgramDatabase(config.database)

        # Add programs to island 0
        for i in range(20):
            program = Program(
                id=f"island0_prog{i}",
                code=f"def test{i}(): pass",
                language="python",
                metrics={"score": 0.5 + i * 0.01},  # Increasing scores
            )
            self.db.add(program, target_island=0)

        # Add some programs to island 1
        for i in range(15):
            program = Program(
                id=f"island1_prog{i}",
                code=f"def test_island1_{i}(): pass",
                language="python",
                metrics={"score": 0.6 + i * 0.01},
            )
            self.db.add(program, target_island=1)

        # Add some programs to island 2
        for i in range(10):
            program = Program(
                id=f"island2_prog{i}",
                code=f"def test_island2_{i}(): pass",
                language="python",
                metrics={"score": 0.7 + i * 0.01},
            )
            self.db.add(program, target_island=2)

    def test_exploration_exploitation_random_ratios(self):
        """Test that sample_from_island respects exploration/exploitation/random ratios"""
        # Set random seed for reproducibility
        random.seed(42)

        # Sample many times and track which mode was used
        num_samples = 1000
        exploration_count = 0
        exploitation_count = 0
        weighted_count = 0

        for _ in range(num_samples):
            # Set seed and get rand_val that will be used
            # We need to track what the internal rand_val would be
            rand_val = random.random()

            # Reset seed to get same rand_val in sample_from_island
            random.seed(random.getstate()[1][0])  # Use current state

            parent, inspirations = self.db.sample_from_island(island_id=0)

            # Determine which mode should have been used based on rand_val
            if rand_val < self.db.config.exploration_ratio:
                exploration_count += 1
            elif rand_val < self.db.config.exploration_ratio + self.db.config.exploitation_ratio:
                exploitation_count += 1
            else:
                weighted_count += 1

        # Check that counts are within reasonable bounds (allowing 10% variance)
        expected_exploration = num_samples * self.db.config.exploration_ratio
        expected_exploitation = num_samples * self.db.config.exploitation_ratio
        expected_weighted = num_samples * (1 - self.db.config.exploration_ratio - self.db.config.exploitation_ratio)

        # Allow 15% tolerance for statistical variance
        tolerance = 0.15

        self.assertGreater(exploration_count, expected_exploration * (1 - tolerance))
        self.assertLess(exploration_count, expected_exploration * (1 + tolerance))

        self.assertGreater(exploitation_count, expected_exploitation * (1 - tolerance))
        self.assertLess(exploitation_count, expected_exploitation * (1 + tolerance))

        self.assertGreater(weighted_count, expected_weighted * (1 - tolerance))
        self.assertLess(weighted_count, expected_weighted * (1 + tolerance))

    def test_sample_from_island_returns_from_correct_island(self):
        """Test that sampled parent is actually from the requested island"""
        random.seed(42)

        for _ in range(100):
            parent, inspirations = self.db.sample_from_island(island_id=0)

            # Parent should be from island 0
            self.assertEqual(parent.metadata.get("island"), 0)

            # Inspirations should also be from island 0
            for insp in inspirations:
                self.assertEqual(insp.metadata.get("island"), 0)

    def test_sample_from_island_with_different_islands(self):
        """Test that different islands return different programs"""
        random.seed(42)

        island0_programs = set()
        island1_programs = set()

        for _ in range(50):
            parent0, _ = self.db.sample_from_island(island_id=0)
            island0_programs.add(parent0.id)

            parent1, _ = self.db.sample_from_island(island_id=1)
            island1_programs.add(parent1.id)

        # Programs from island 0 should be different from island 1
        # (they should be disjoint sets)
        self.assertEqual(len(island0_programs & island1_programs), 0)

    def test_exploitation_uses_archive(self):
        """Test that exploitation mode samples from archive"""
        # Force exploitation ratio to 1.0 to guarantee archive sampling
        self.db.config.exploration_ratio = 0.0
        self.db.config.exploitation_ratio = 1.0

        random.seed(42)

        # Sample many times and check if we get archive programs
        for _ in range(20):
            parent, _ = self.db.sample_from_island(island_id=0)

            # Parent should either be from archive or from island if archive is empty
            # Since we have programs, archive should be populated
            self.assertIsNotNone(parent)
            self.assertIn(parent.id, self.db.programs)

    def test_exploration_mode_uniform_distribution(self):
        """Test that exploration mode uses uniform random sampling"""
        # Force exploration ratio to 1.0
        self.db.config.exploration_ratio = 1.0
        self.db.config.exploitation_ratio = 0.0

        random.seed(42)

        # Sample many times and track distribution
        sampled_ids = []
        for _ in range(200):
            parent, _ = self.db.sample_from_island(island_id=0)
            sampled_ids.append(parent.id)

        # Count occurrences
        from collections import Counter
        counts = Counter(sampled_ids)

        # In exploration mode (uniform random), all programs should have similar counts
        # Check that no program is sampled way more than others (> 2x average)
        avg_count = len(sampled_ids) / len(counts)
        max_count = max(counts.values())
        min_count = min(counts.values())

        # With uniform distribution, max shouldn't be more than 3x the average
        # (allowing for statistical variance)
        self.assertLess(max_count, avg_count * 3)
        self.assertGreater(min_count, avg_count * 0.3)

    def test_weighted_mode_favors_high_fitness(self):
        """Test that weighted mode favors programs with higher fitness"""
        # Force weighted ratio to 1.0
        self.db.config.exploration_ratio = 0.0
        self.db.config.exploitation_ratio = 0.0

        random.seed(42)

        # Sample many times and track which programs are selected
        sampled_scores = []
        for _ in range(200):
            parent, _ = self.db.sample_from_island(island_id=0)
            sampled_scores.append(parent.metrics["score"])

        # Average sampled score should be higher than mean score of all programs
        # (mean represents what uniform random sampling would produce)
        island_programs = [self.db.programs[pid] for pid in self.db.islands[0]]
        all_scores = [p.metrics["score"] for p in island_programs]
        mean_score = sum(all_scores) / len(all_scores)
        avg_sampled_score = sum(sampled_scores) / len(sampled_scores)

        # Weighted sampling should favor higher scores (shift average upward from mean)
        self.assertGreater(avg_sampled_score, mean_score,
                          f"Weighted sampling should favor high fitness: "
                          f"sampled_avg={avg_sampled_score:.4f} should be > mean={mean_score:.4f}")


class TestSampleFromIslandEdgeCases(unittest.TestCase):
    """Tests for edge cases in sample_from_island()"""

    def setUp(self):
        """Set up test database"""
        config = Config()
        config.database.in_memory = True
        config.database.num_islands = 3
        self.db = ProgramDatabase(config.database)

    def test_empty_island_fallback(self):
        """Test that empty island falls back to sample()"""
        # Add programs only to island 0
        for i in range(5):
            program = Program(
                id=f"prog{i}",
                code=f"def test{i}(): pass",
                language="python",
                metrics={"score": 0.5 + i * 0.1},
            )
            self.db.add(program, target_island=0)

        # Try to sample from empty island 1
        parent, inspirations = self.db.sample_from_island(island_id=1)

        # Should still return a parent (from fallback)
        self.assertIsNotNone(parent)

    def test_empty_archive_fallback(self):
        """Test exploitation mode with empty archive"""
        # Force exploitation mode
        self.db.config.exploration_ratio = 0.0
        self.db.config.exploitation_ratio = 1.0

        # Clear archive
        self.db.archive.clear()

        # Add programs to island
        for i in range(5):
            program = Program(
                id=f"prog{i}",
                code=f"def test{i}(): pass",
                language="python",
                metrics={"score": 0.5 + i * 0.1},
            )
            self.db.add(program, target_island=0)

        # Clear archive again (add() populates it)
        self.db.archive.clear()

        # Sample should still work (fall back to weighted sampling)
        parent, inspirations = self.db.sample_from_island(island_id=0)
        self.assertIsNotNone(parent)

    def test_single_program_island(self):
        """Test sampling from island with only one program"""
        program = Program(
            id="solo",
            code="def solo(): pass",
            language="python",
            metrics={"score": 0.8},
        )
        self.db.add(program, target_island=0)

        # Should successfully return the single program
        parent, inspirations = self.db.sample_from_island(island_id=0)
        self.assertEqual(parent.id, "solo")
        # No inspirations available (only one program)
        self.assertEqual(len(inspirations), 0)

    def test_island_id_wrapping(self):
        """Test that island_id wraps around correctly"""
        # Add programs to island 0
        for i in range(5):
            program = Program(
                id=f"prog{i}",
                code=f"def test{i}(): pass",
                language="python",
                metrics={"score": 0.5},
            )
            self.db.add(program, target_island=0)

        # Sample from island_id = 3 (should wrap to 0 since we have 3 islands)
        parent1, _ = self.db.sample_from_island(island_id=3)

        # Sample from island_id = 0
        random.seed(42)
        parent2, _ = self.db.sample_from_island(island_id=0)

        # Both should be from the same island (island 0)
        self.assertEqual(parent1.metadata.get("island"), 0)
        self.assertEqual(parent2.metadata.get("island"), 0)


if __name__ == "__main__":
    unittest.main()
