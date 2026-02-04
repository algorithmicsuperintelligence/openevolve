"""
Tests for verifying child programs are placed in the correct target island.

This test specifically catches the bug where children inherit their parent's island
instead of being placed in the target island that was requested for the iteration.
"""

import unittest

from openevolve.config import Config, DatabaseConfig
from openevolve.database import ProgramDatabase, Program


class TestIslandChildPlacement(unittest.TestCase):
    """Test that child programs are placed in the correct island"""

    def setUp(self):
        """Set up test database with multiple islands"""
        config = Config()
        config.database.num_islands = 3
        config.database.population_size = 100
        self.db = ProgramDatabase(config.database)

    def test_child_inherits_parent_island_when_no_target_specified(self):
        """Test that child inherits parent's island when no target_island is given"""
        # Add parent to island 0
        parent = Program(
            id="parent_0",
            code="def parent(): pass",
            generation=0,
            metrics={"combined_score": 0.5},
        )
        self.db.add(parent, target_island=0)

        # Add child without specifying target_island
        child = Program(
            id="child_0",
            code="def child(): pass",
            generation=1,
            parent_id="parent_0",
            metrics={"combined_score": 0.6},
        )
        self.db.add(child)  # No target_island specified

        # Child should inherit parent's island (island 0)
        self.assertEqual(child.metadata.get("island"), 0)
        self.assertIn("child_0", self.db.islands[0])

    def test_child_placed_in_target_island_when_specified(self):
        """Test that child is placed in target_island when explicitly specified"""
        # Add parent to island 0
        parent = Program(
            id="parent_1",
            code="def parent(): pass",
            generation=0,
            metrics={"combined_score": 0.5},
        )
        self.db.add(parent, target_island=0)

        # Add child with explicit target_island=2
        child = Program(
            id="child_1",
            code="def child(): pass",
            generation=1,
            parent_id="parent_1",
            metrics={"combined_score": 0.6},
        )
        self.db.add(child, target_island=2)

        # Child should be in island 2, NOT island 0
        self.assertEqual(child.metadata.get("island"), 2)
        self.assertIn("child_1", self.db.islands[2])
        self.assertNotIn("child_1", self.db.islands[0])


class TestEmptyIslandChildPlacement(unittest.TestCase):
    """
    Test the critical bug: when sampling from an empty island falls back to
    another island's parent, the child should still go to the TARGET island.
    """

    def setUp(self):
        """Set up test database with programs only in island 0"""
        config = Config()
        config.database.num_islands = 3
        config.database.population_size = 100
        self.db = ProgramDatabase(config.database)

        # Add programs ONLY to island 0
        for i in range(5):
            program = Program(
                id=f"island0_prog_{i}",
                code=f"def func_{i}(): pass",
                generation=0,
                metrics={"combined_score": 0.5 + i * 0.1},
            )
            self.db.add(program, target_island=0)

        # Verify setup: island 0 has programs, islands 1 and 2 are empty
        self.assertGreater(len(self.db.islands[0]), 0)
        self.assertEqual(len(self.db.islands[1]), 0)
        self.assertEqual(len(self.db.islands[2]), 0)

    def test_sample_from_empty_island_returns_fallback_parent(self):
        """Test that sampling from empty island falls back to available programs"""
        # Sample from empty island 1
        parent, inspirations = self.db.sample_from_island(island_id=1)

        # Should return a parent (from island 0 via fallback)
        self.assertIsNotNone(parent)
        # Parent is from island 0
        self.assertEqual(parent.metadata.get("island"), 0)

    def test_child_should_go_to_target_island_not_parent_island(self):
        """
        CRITICAL TEST: This tests the bug reported in issue #391.

        When we want to add a child to island 1 (empty), but the parent came
        from island 0 (via fallback sampling), the child should still be
        placed in island 1 (the TARGET), not island 0 (the parent's island).
        """
        # Sample from empty island 1 - will fall back to island 0
        parent, inspirations = self.db.sample_from_island(island_id=1)

        # Parent is from island 0 (the only island with programs)
        self.assertEqual(parent.metadata.get("island"), 0)

        # Create a child program
        child = Program(
            id="child_for_island_1",
            code="def evolved(): pass",
            generation=1,
            parent_id=parent.id,
            metrics={"combined_score": 0.8},
        )

        # THIS IS THE BUG: If we don't pass target_island, child goes to parent's island
        # Simulating current behavior (without fix):
        self.db.add(child)  # No target_island - inherits parent's island

        # BUG: Child ends up in island 0 (parent's island) instead of island 1 (target)
        # This assertion will FAIL with current code, demonstrating the bug
        self.assertEqual(
            child.metadata.get("island"), 1,
            "Child should be in target island 1, not parent's island 0. "
            "This is the bug from issue #391!"
        )

    def test_explicit_target_island_overrides_parent_inheritance(self):
        """Test that explicit target_island works even with fallback parent"""
        # Sample from empty island 2 - will fall back to island 0
        parent, inspirations = self.db.sample_from_island(island_id=2)

        # Parent is from island 0
        self.assertEqual(parent.metadata.get("island"), 0)

        # Create child and explicitly specify target island
        child = Program(
            id="child_for_island_2",
            code="def evolved(): pass",
            generation=1,
            parent_id=parent.id,
            metrics={"combined_score": 0.8},
        )

        # With explicit target_island, child should go to island 2
        self.db.add(child, target_island=2)

        # This should work - explicit target_island is respected
        self.assertEqual(child.metadata.get("island"), 2)
        self.assertIn("child_for_island_2", self.db.islands[2])


class TestIslandPopulationGrowth(unittest.TestCase):
    """
    Test that simulates multiple evolution iterations and checks
    that all islands eventually get populated.
    """

    def setUp(self):
        """Set up test database"""
        config = Config()
        config.database.num_islands = 3
        config.database.population_size = 100
        self.db = ProgramDatabase(config.database)

    def test_islands_should_all_get_populated(self):
        """
        Simulate evolution and verify all islands get programs.

        This test demonstrates the feedback loop bug: if children always
        inherit parent's island, and we start with only island 0 populated,
        all new programs will go to island 0.
        """
        # Start with initial program in island 0 only
        initial = Program(
            id="initial",
            code="def initial(): pass",
            generation=0,
            metrics={"combined_score": 0.5},
        )
        self.db.add(initial, target_island=0)

        # Simulate 9 iterations, targeting islands in round-robin fashion
        for i in range(9):
            target_island = i % 3  # 0, 1, 2, 0, 1, 2, 0, 1, 2

            # Sample from target island (may fall back if empty)
            parent, _ = self.db.sample_from_island(island_id=target_island)

            # Create child
            child = Program(
                id=f"child_{i}",
                code=f"def child_{i}(): pass",
                generation=1,
                parent_id=parent.id,
                metrics={"combined_score": 0.5 + i * 0.05},
            )

            # BUG: Current code doesn't pass target_island, so child inherits parent's island
            # This simulates the current buggy behavior:
            self.db.add(child)  # No target_island

        # Check island populations
        island_sizes = [len(self.db.islands[i]) for i in range(3)]

        # With the bug, ALL programs end up in island 0
        # This assertion will FAIL, demonstrating the bug
        self.assertGreater(
            island_sizes[1], 0,
            f"Island 1 should have programs but has {island_sizes[1]}. "
            f"All islands: {island_sizes}. Bug: all programs went to island 0!"
        )
        self.assertGreater(
            island_sizes[2], 0,
            f"Island 2 should have programs but has {island_sizes[2]}. "
            f"All islands: {island_sizes}. Bug: all programs went to island 0!"
        )


if __name__ == "__main__":
    unittest.main()
