"""
Tests for island best program tracking functionality in openevolve.database
"""

import unittest
from openevolve.config import Config
from openevolve.database import Program
from openevolve.database_memory import InMemoryProgramDatabase as ProgramDatabase


class TestIslandTracking(unittest.TestCase):
    """Tests for island best program tracking in program database"""

    def setUp(self):
        """Set up test database with multiple islands"""
        config = Config()
        config.database.in_memory = True
        config.database.num_islands = 3
        self.db = ProgramDatabase(config.database)

    def _create_test_program(self, program_id: str, score: float, island: int) -> Program:
        """Helper to create a test program"""
        program = Program(
            id=program_id,
            code=f"def func_{program_id}(): return {score}",
            language="python",
            metrics={"score": score, "combined_score": score},
            metadata={"island": island},
        )
        return program

    def test_initial_island_best_tracking(self):
        """Test initial state of island best program tracking"""
        # Initially all island best programs should be None
        self.assertEqual(len(self.db.island_best_programs), 3)
        for best_id in self.db.island_best_programs:
            self.assertIsNone(best_id)

    def test_first_program_becomes_island_best(self):
        """Test that the first program added to an island becomes the best"""
        program = self._create_test_program("first", 0.5, 0)
        self.db.add(program, target_island=0)

        # Should become the best program for island 0
        self.assertEqual(self.db.island_best_programs[0], "first")

        # Other islands should still have None
        self.assertIsNone(self.db.island_best_programs[1])
        self.assertIsNone(self.db.island_best_programs[2])

    def test_better_program_updates_island_best(self):
        """Test that a better program replaces the island best"""
        # Add initial program
        program1 = self._create_test_program("mediocre", 0.5, 0)
        self.db.add(program1, target_island=0)
        self.assertEqual(self.db.island_best_programs[0], "mediocre")

        # Add better program
        program2 = self._create_test_program("better", 0.8, 0)
        self.db.add(program2, target_island=0)
        self.assertEqual(self.db.island_best_programs[0], "better")

    def test_worse_program_does_not_update_island_best(self):
        """Test that a worse program does not replace the island best"""
        # Add good program
        program1 = self._create_test_program("good", 0.8, 0)
        self.db.add(program1, target_island=0)
        self.assertEqual(self.db.island_best_programs[0], "good")

        # Add worse program
        program2 = self._create_test_program("worse", 0.3, 0)
        self.db.add(program2, target_island=0)

        # Should still be the good program
        self.assertEqual(self.db.island_best_programs[0], "good")

    def test_island_isolation_in_best_tracking(self):
        """Test that island best tracking is isolated between islands"""
        # Add programs to different islands
        program1 = self._create_test_program("island0_best", 0.9, 0)
        program2 = self._create_test_program("island1_best", 0.7, 1)
        program3 = self._create_test_program("island2_best", 0.5, 2)

        self.db.add(program1, target_island=0)
        self.db.add(program2, target_island=1)
        self.db.add(program3, target_island=2)

        # Each island should track its own best
        self.assertEqual(self.db.island_best_programs[0], "island0_best")
        self.assertEqual(self.db.island_best_programs[1], "island1_best")
        self.assertEqual(self.db.island_best_programs[2], "island2_best")

    def test_migration_updates_island_best(self):
        """Test that migration can update island best programs"""
        # Add program to island 0
        original = self._create_test_program("original", 0.6, 0)
        self.db.add(original, target_island=0)

        # Island 1 starts empty
        self.assertIsNone(self.db.island_best_programs[1])

        # Manually create a migrant to island 1 (simulating migration)
        migrant = Program(
            id="original_migrant_1",
            code=original.code,
            language=original.language,
            parent_id=original.id,
            generation=original.generation,
            metrics=original.metrics.copy(),
            metadata={"island": 1, "migrant": True},
        )

        # Add migrant to island 1
        self.db.add(migrant, target_island=1)

        # Should become best for island 1
        self.assertEqual(self.db.island_best_programs[1], "original_migrant_1")

    def test_get_top_programs_island_specific(self):
        """Test getting top programs from a specific island"""
        # Add programs to island 0
        program1 = self._create_test_program("prog1", 0.9, 0)
        program2 = self._create_test_program("prog2", 0.7, 0)
        program3 = self._create_test_program("prog3", 0.5, 0)

        # Add programs to island 1
        program4 = self._create_test_program("prog4", 0.8, 1)
        program5 = self._create_test_program("prog5", 0.6, 1)

        self.db.add(program1, target_island=0)
        self.db.add(program2, target_island=0)
        self.db.add(program3, target_island=0)
        self.db.add(program4, target_island=1)
        self.db.add(program5, target_island=1)

        # Get top programs from island 0
        island0_top = self.db.get_top_programs(n=2, island_idx=0)
        self.assertEqual(len(island0_top), 2)
        self.assertEqual(island0_top[0].id, "prog1")  # Highest score
        self.assertEqual(island0_top[1].id, "prog2")  # Second highest

        # Get top programs from island 1
        island1_top = self.db.get_top_programs(n=2, island_idx=1)
        self.assertEqual(len(island1_top), 2)
        self.assertEqual(island1_top[0].id, "prog4")  # Highest score in island 1
        self.assertEqual(island1_top[1].id, "prog5")  # Second highest in island 1

    def test_island_best_with_combined_score(self):
        """Test island best tracking with combined_score metric"""
        # Add programs with combined_score
        program1 = Program(
            id="test1",
            code="def test1(): pass",
            language="python",
            metrics={"score": 0.5, "other": 0.3, "combined_score": 0.4},
            metadata={"island": 0},
        )

        program2 = Program(
            id="test2",
            code="def test2(): pass",
            language="python",
            metrics={"score": 0.3, "other": 0.7, "combined_score": 0.5},
            metadata={"island": 0},
        )

        self.db.add(program1, target_island=0)
        self.assertEqual(self.db.island_best_programs[0], "test1")

        # program2 has higher combined_score, should become best
        self.db.add(program2, target_island=0)
        self.assertEqual(self.db.island_best_programs[0], "test2")

    def test_island_best_with_missing_program(self):
        """Test island best tracking when best program is removed"""
        program = self._create_test_program("to_remove", 0.8, 0)
        self.db.add(program, target_island=0)
        self.assertEqual(self.db.island_best_programs[0], "to_remove")

        # Manually remove the program (simulating cleanup)
        del self.db.programs["to_remove"]
        self.db.islands[0].remove("to_remove")

        # Add a new program - should detect stale reference and update
        new_program = self._create_test_program("new", 0.6, 0)
        self.db.add(new_program, target_island=0)

        # Should update the best program (the old one is gone)
        self.assertEqual(self.db.island_best_programs[0], "new")

    def test_sample_inspirations_from_island(self):
        """Test that inspiration sampling respects island boundaries"""
        # Add programs to island 0
        program1 = self._create_test_program("island0_prog1", 0.9, 0)
        program2 = self._create_test_program("island0_prog2", 0.7, 0)

        # Add programs to island 1
        program3 = self._create_test_program("island1_prog1", 0.8, 1)
        program4 = self._create_test_program("island1_prog2", 0.6, 1)

        self.db.add(program1, target_island=0)
        self.db.add(program2, target_island=0)
        self.db.add(program3, target_island=1)
        self.db.add(program4, target_island=1)

        # Sample from island 0 program
        inspirations = self.db._sample_inspirations(program1, n=5)

        # All inspirations should be from island 0
        for inspiration in inspirations:
            island = inspiration.metadata.get("island")
            self.assertEqual(
                island, 0, f"Program {inspiration.id} should be from island 0, got {island}"
            )

    def test_island_status_logging(self):
        """Test island status logging functionality"""
        # Add programs to different islands
        program1 = self._create_test_program("p1", 0.9, 0)
        program2 = self._create_test_program("p2", 0.7, 1)

        self.db.add(program1, target_island=0)
        self.db.add(program2, target_island=1)

        # Should not crash when logging status
        try:
            self.db.log_island_status()
        except Exception as e:
            self.fail(f"Island status logging should not crash: {e}")

    def test_island_best_persistence(self):
        """Test that island best programs are maintained across operations"""
        # Add programs to islands
        program1 = self._create_test_program("best0", 0.9, 0)
        program2 = self._create_test_program("best1", 0.8, 1)

        self.db.add(program1, target_island=0)
        self.db.add(program2, target_island=1)

        # Verify initial state
        self.assertEqual(self.db.island_best_programs[0], "best0")
        self.assertEqual(self.db.island_best_programs[1], "best1")

        # Add more programs that are not better
        program3 = self._create_test_program("worse0", 0.5, 0)
        program4 = self._create_test_program("worse1", 0.4, 1)

        self.db.add(program3, target_island=0)
        self.db.add(program4, target_island=1)

        # Best should remain unchanged
        self.assertEqual(self.db.island_best_programs[0], "best0")
        self.assertEqual(self.db.island_best_programs[1], "best1")

    def test_invalid_island_index_handling(self):
        """Test handling of invalid island indices"""
        # Test with island index out of bounds
        with self.assertRaises(IndexError):
            self.db.get_top_programs(n=5, island_idx=10)

    def test_empty_island_top_programs(self):
        """Test getting top programs from empty island"""
        # Island 0 is empty initially
        top_programs = self.db.get_top_programs(n=5, island_idx=0)
        self.assertEqual(len(top_programs), 0)


if __name__ == "__main__":
    unittest.main()
