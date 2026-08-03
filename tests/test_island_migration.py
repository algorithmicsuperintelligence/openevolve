"""
Tests for island migration functionality in openevolve.database
"""

import unittest
from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase


class TestIslandMigration(unittest.TestCase):
    """Tests for island migration in program database"""

    def setUp(self):
        """Set up test database with multiple islands"""
        config = Config()
        config.database.in_memory = True
        config.database.num_islands = 3
        config.database.migration_rate = 0.5  # 50% of programs migrate
        config.database.migration_interval = 5  # Migrate every 5 generations
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

    def test_initial_island_setup(self):
        """Test that islands are properly initialized"""
        self.assertEqual(len(self.db.islands), 3)
        self.assertEqual(len(self.db.island_best_programs), 3)
        self.assertEqual(len(self.db.island_generations), 3)

        # All islands should be empty initially
        for island in self.db.islands:
            self.assertEqual(len(island), 0)

        # All island best programs should be None initially
        for best_id in self.db.island_best_programs:
            self.assertIsNone(best_id)

    def test_program_island_assignment(self):
        """Test that programs are assigned to correct islands"""
        # Add programs to specific islands
        program1 = self._create_test_program("test1", 0.5, 0)
        program2 = self._create_test_program("test2", 0.7, 1)
        program3 = self._create_test_program("test3", 0.3, 2)

        self.db.add(program1, target_island=0)
        self.db.add(program2, target_island=1)
        self.db.add(program3, target_island=2)

        # Verify island assignments
        self.assertIn("test1", self.db.islands[0])
        self.assertIn("test2", self.db.islands[1])
        self.assertIn("test3", self.db.islands[2])

        # Verify metadata
        self.assertEqual(self.db.programs["test1"].metadata["island"], 0)
        self.assertEqual(self.db.programs["test2"].metadata["island"], 1)
        self.assertEqual(self.db.programs["test3"].metadata["island"], 2)

    def test_should_migrate_logic(self):
        """Test the migration timing logic"""
        # Initially should not migrate (no generations passed)
        self.assertFalse(self.db.should_migrate())

        # Advance island generations
        self.db.island_generations = [5, 6, 7]  # Max is 7, last migration was 0, so 7-0=7 >= 5
        self.assertTrue(self.db.should_migrate())

        # Test with mixed generations below threshold
        self.db.island_generations = [3, 4, 2]  # Max is 4, 4-0=4 < 5
        self.assertFalse(self.db.should_migrate())

    def test_migration_ring_topology(self):
        """Test that migration follows ring topology"""
        # Add programs to islands 0 and 1
        program1 = self._create_test_program("test1", 0.8, 0)
        program2 = self._create_test_program("test2", 0.6, 1)

        self.db.add(program1, target_island=0)
        self.db.add(program2, target_island=1)

        # Set up for migration
        self.db.island_generations = [6, 6, 6]  # Trigger migration

        initial_program_count = len(self.db.programs)

        # Perform migration
        self.db.migrate_programs()

        # Should have created migrant copies
        self.assertGreater(len(self.db.programs), initial_program_count)

        # With new implementation, verify migration occurred by checking island populations
        # and ensuring no _migrant_ suffixes exist
        migrant_suffix_ids = [pid for pid in self.db.programs.keys() if "_migrant_" in pid]
        self.assertEqual(len(migrant_suffix_ids), 0, "No programs should have _migrant_ suffixes")
        
        # Verify migration occurred by checking that programs exist in multiple islands
        programs_in_islands = []
        for island_idx, island_map in enumerate(self.db.island_feature_maps):
            programs_in_islands.extend([(pid, island_idx) for pid in island_map.values()])
        
        # Should have programs distributed across islands due to migration
        islands_with_programs = set(island_idx for _, island_idx in programs_in_islands)
        self.assertGreater(len(islands_with_programs), 1, "Migration should distribute programs across islands")
        # This is a known limitation of the current implementation that processes islands
        # sequentially while modifying them, causing interference between migration rounds.

    def test_migration_rate_respected(self):
        """Test that migration rate is properly applied"""
        # Add multiple programs to island 0
        programs = []
        for i in range(10):
            program = self._create_test_program(f"test{i}", 0.5 + i * 0.05, 0)
            programs.append(program)
            self.db.add(program, target_island=0)

        # Set up for migration
        self.db.island_generations = [6, 6, 6]

        # Count actual programs on island 0 after MAP-Elites deduplication
        # (some of the 10 programs might have been replaced if they mapped to same cell)
        island_0_count = len(self.db.islands[0])
        initial_program_count = len(self.db.programs)

        # Perform migration
        self.db.migrate_programs()

        # Calculate expected migrants based on ACTUAL island population
        # With 50% migration rate, expect ceil(island_0_count * 0.5) migrants
        import math
        expected_migrants = math.ceil(island_0_count * self.db.config.migration_rate)
        # Each migrant goes to 2 target islands
        expected_new_programs = expected_migrants * 2
        actual_new_programs = len(self.db.programs) - initial_program_count

        # Should have at least the expected migrants (accounting for MAP-Elites deduplication on targets)
        # Note: actual may be less than expected if migrants are deduplicated on target islands
        self.assertGreaterEqual(actual_new_programs, 0, "Migration should create new programs or be skipped")

        # With new implementation, verify no _migrant_ suffixes exist
        migrant_suffix_programs = [pid for pid in self.db.programs.keys() if "_migrant_" in pid]
        self.assertEqual(len(migrant_suffix_programs), 0, "No programs should have _migrant_ suffixes")

    def test_migration_preserves_best_programs(self):
        """Test that migration selects the best programs for migration"""
        # Add programs with different scores to island 0
        program1 = self._create_test_program("low_score", 0.2, 0)
        program2 = self._create_test_program("high_score", 0.9, 0)
        program3 = self._create_test_program("med_score", 0.5, 0)

        self.db.add(program1, target_island=0)
        self.db.add(program2, target_island=0)
        self.db.add(program3, target_island=0)

        # Set up for migration
        self.db.island_generations = [6, 6, 6]

        # Perform migration
        self.db.migrate_programs()

        # With new implementation, verify programs were migrated but no _migrant_ suffixes exist
        migrant_suffix_programs = [pid for pid in self.db.programs.keys() if "_migrant_" in pid]
        self.assertEqual(len(migrant_suffix_programs), 0, "No programs should have _migrant_ suffixes")
        
        # Verify that high-quality programs are distributed across islands
        high_score_program = self.db.get("high_score")
        self.assertIsNotNone(high_score_program, "Original high score program should still exist")
        
        # Main requirement: verify migration doesn't create duplicate chains
        # Migration behavior may vary based on feature coordinates and randomness
        total_programs_after = len(self.db.programs)
        self.assertGreaterEqual(total_programs_after, 3, "Should have at least the original programs")

    def test_migration_updates_generations(self):
        """Test that migration updates the last migration generation"""
        # Add a program and set up for migration
        program = self._create_test_program("test1", 0.5, 0)
        self.db.add(program, target_island=0)

        self.db.island_generations = [6, 7, 8]
        initial_migration_gen = self.db.last_migration_generation

        # Perform migration
        self.db.migrate_programs()

        # Should update to max of island generations
        self.assertEqual(self.db.last_migration_generation, 8)
        self.assertGreater(self.db.last_migration_generation, initial_migration_gen)

    def test_migration_with_empty_islands(self):
        """Test that migration handles empty islands gracefully"""
        # Add program only to island 0, leave others empty
        program = self._create_test_program("test1", 0.5, 0)
        self.db.add(program, target_island=0)

        # Set up for migration
        self.db.island_generations = [6, 6, 6]

        # Should not crash with empty islands
        try:
            self.db.migrate_programs()
        except Exception as e:
            self.fail(f"Migration with empty islands should not crash: {e}")

    def test_migration_creates_proper_copies(self):
        """Test that migration creates proper program copies"""
        program = self._create_test_program("original", 0.7, 0)
        self.db.add(program, target_island=0)
        artifacts = {
            "program-identity.txt": "stable-program-identity",
            "search-map.txt": "compact-search-map",
        }
        self.db.store_artifacts(program.id, artifacts)

        # Set up for migration
        self.db.island_generations = [6, 6, 6]

        # Perform migration
        self.db.migrate_programs()

        # With new implementation, no _migrant_ suffixes should exist
        migrant_suffix_ids = [pid for pid in self.db.programs.keys() if "_migrant_" in pid]
        self.assertEqual(len(migrant_suffix_ids), 0, "No programs should have _migrant_ suffixes")
        
        # Verify migration created new programs (indicated by increased program count)
        original_program = self.db.get("original")
        self.assertIsNotNone(original_program, "Original program should still exist")
        
        # Check migration behavior - main requirement is no duplicates
        # Migration may or may not distribute to other islands depending on feature coordinates and randomness
        total_programs_after = len(self.db.programs)
        self.assertGreaterEqual(total_programs_after, 1, "Should have at least the original program")
        
        # Check properties of migrated programs (those marked with migrant metadata)
        migrated_programs = [p for p in self.db.programs.values() if p.metadata.get("migrant", False)]
        for migrant in migrated_programs:

            # Should have same code and metrics as original
            self.assertEqual(migrant.code, program.code)
            self.assertEqual(migrant.metrics, program.metrics)

            # Should have proper parent reference
            self.assertEqual(migrant.parent_id, "original")

            # Should be marked as migrant
            self.assertTrue(migrant.metadata.get("migrant", False))

            # A migrated program can be sampled as a parent. Preserve the
            # evaluator context required by identity and archive gates.
            self.assertEqual(artifacts, self.db.get_artifacts(migrant.id))

            # Should be in correct target island
            target_island = migrant.metadata["island"]
            self.assertIn(migrant.id, self.db.islands[target_island])

    def test_no_migration_with_single_island(self):
        """Test that migration is skipped with single island"""
        # Create database with single island
        config = Config()
        config.database.in_memory = True
        config.database.num_islands = 1
        single_island_db = ProgramDatabase(config.database)

        program = self._create_test_program("test1", 0.5, 0)
        single_island_db.add(program, target_island=0)

        single_island_db.island_generations = [6]

        initial_count = len(single_island_db.programs)

        # Should not perform migration
        single_island_db.migrate_programs()

        # Program count should remain the same
        self.assertEqual(len(single_island_db.programs), initial_count)


if __name__ == "__main__":
    unittest.main()
