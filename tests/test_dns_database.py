"""
Tests for DNSProgramDatabase in openevolve.database
"""

import unittest
import uuid
from openevolve.config import Config
from openevolve.database import Program, DNSProgramDatabase
from tests.test_database import TestProgramDatabase


class TestDNSProgramDatabase(TestProgramDatabase):
    """Tests for program database"""

    def setUp(self):
        """Set up test database"""
        config = Config()
        config.database.in_memory = True
        self.db = DNSProgramDatabase(config.database)

    def test_multi_island_setup(self):
        """Test database with multiple islands"""
        # Create new database with multiple islands
        config = Config()
        config.database.in_memory = True
        config.database.num_islands = 3
        multi_db = DNSProgramDatabase(config.database)

        self.assertEqual(len(multi_db.islands), 3)
        self.assertEqual(len(multi_db.island_best_programs), 3)

        # Add programs to specific islands
        for i in range(3):
            program = Program(
                id=f"test_island_{i}",
                code=f"def test_{i}(): pass",
                language="python",
                metrics={"score": 0.5 + i * 0.1},
            )
            multi_db.add(program, target_island=i)

            # Verify assignment
            self.assertIn(f"test_island_{i}", multi_db.islands[i])
            self.assertEqual(program.metadata.get("island"), i)

    def test_feature_map_operations(self):
        pass

    def test_migration_prevents_re_migration(self):
        """Test that programs marked as migrants don't migrate again"""
        # Create database with multiple islands
        config = Config()
        config.database.in_memory = True
        config.database.num_islands = 3
        config.database.migration_interval = 1  # Migrate every generation
        multi_db = DNSProgramDatabase(config.database)

        # Add programs to each island (avoid "migrant" in original IDs)
        for i in range(3):
            program = Program(
                id=f"test_prog_{i}",
                code=f"def test_{i}(): return {i}",
                language="python",
                metrics={"score": 0.5 + i * 0.1},
            )
            multi_db.add(program, target_island=i)

        # Manually mark one as a migrant
        migrant_program = multi_db.get("test_prog_0")
        migrant_program.metadata["migrant"] = True

        # Store original ID
        original_id = migrant_program.id

        # Count initial programs (no _migrant suffixes should exist)
        initial_programs = set(multi_db.programs.keys())
        initial_migrant_count = sum(1 for pid in initial_programs if "_migrant_" in pid)
        self.assertEqual(initial_migrant_count, 0)  # Should be none with new implementation

        # Run migration
        multi_db.island_generations[0] = config.database.migration_interval
        multi_db.island_generations[1] = config.database.migration_interval
        multi_db.island_generations[2] = config.database.migration_interval
        multi_db.migrate_programs()

        # Check that the migrant program wasn't re-migrated
        # It should still exist with the same ID
        still_exists = multi_db.get(original_id)
        self.assertIsNotNone(still_exists)

        # With new implementation, no programs should have _migrant_ suffixes
        new_programs = set(multi_db.programs.keys())
        new_migrant_ids = [pid for pid in new_programs if "_migrant_" in pid]
        self.assertEqual(len(new_migrant_ids), 0, "New implementation should not create _migrant suffix programs")

    def test_empty_island_initialization_creates_copies(self):
        """Test that empty islands are initialized with copies, not shared references"""
        # Create database with multiple islands
        config = Config()
        config.database.in_memory = True
        config.database.num_islands = 3
        # Force exploration mode to test empty island handling
        config.database.exploration_ratio = 1.0
        config.database.exploitation_ratio = 0.0
        multi_db = DNSProgramDatabase(config.database)

        # Add a single program to island 1
        program = Program(
            id="original_program",
            code="def original(): return 42",
            language="python",
            metrics={"score": 0.9, "combined_score": 0.9},
        )
        multi_db.add(program, target_island=1)

        # Make it the best program
        multi_db.best_program_id = "original_program"

        # Switch to empty island 0 and sample
        multi_db.set_current_island(0)
        sampled_parent, _ = multi_db.sample()

        # The sampled program should be a copy, not the original
        self.assertNotEqual(sampled_parent.id, "original_program")
        self.assertEqual(sampled_parent.code, program.code)  # Same code
        self.assertEqual(sampled_parent.parent_id, "original_program")  # Parent is the original

        # Check island membership
        self.assertIn("original_program", multi_db.islands[1])
        self.assertNotIn("original_program", multi_db.islands[0])
        self.assertIn(sampled_parent.id, multi_db.islands[0])

        # Run validation - should not raise any errors
        multi_db._validate_migration_results()

    def test_no_program_assigned_to_multiple_islands(self):
        """Test that programs are never assigned to multiple islands"""
        # Create database with multiple islands
        config = Config()
        config.database.in_memory = True
        config.database.num_islands = 4
        multi_db = DNSProgramDatabase(config.database)

        # Add programs to different islands
        program_ids = []
        for i in range(4):
            program = Program(
                id=f"island_test_{i}",
                code=f"def test_{i}(): return {i}",
                language="python",
                metrics={"score": 0.5 + i * 0.1, "combined_score": 0.5 + i * 0.1},
            )
            multi_db.add(program, target_island=i)
            program_ids.append(program.id)

        # Make the best program from island 3
        multi_db.best_program_id = "island_test_3"

        # Sample from empty islands - this should create copies
        for empty_island in range(4):
            if len(multi_db.islands[empty_island]) == 0:
                multi_db.set_current_island(empty_island)
                parent, _ = multi_db.sample()

        # Check that no program ID appears in multiple islands
        all_island_programs = {}
        for island_idx, island_programs in enumerate(multi_db.islands):
            for program_id in island_programs:
                if program_id in all_island_programs:
                    self.fail(
                        f"Program {program_id} found in both island {all_island_programs[program_id]} "
                        f"and island {island_idx}"
                    )
                all_island_programs[program_id] = island_idx

        # Run validation - should not raise any errors
        multi_db._validate_migration_results()

    def test_migration_validation_passes(self):
        """Test that migration validation passes after our fixes"""
        # Create database with multiple islands
        config = Config()
        config.database.in_memory = True
        config.database.num_islands = 3
        config.database.migration_interval = 1
        multi_db = DNSProgramDatabase(config.database)

        # Add programs and run several migration cycles
        for i in range(6):
            program = Program(
                id=f"test_program_{i}",
                code=f"def test_{i}(): return {i * 2}",
                language="python",
                metrics={"score": 0.4 + i * 0.1, "combined_score": 0.4 + i * 0.1},
            )
            multi_db.add(program, target_island=i % 3)

        # Run multiple migration cycles
        for cycle in range(3):
            # Increment generations to trigger migration
            for island in range(3):
                multi_db.island_generations[island] += 1

            # Migrate programs
            multi_db.migrate_programs()

            # Validation should pass without warnings
            multi_db._validate_migration_results()

            # Verify no program has exponential ID growth
            for program_id in multi_db.programs:
                # Count occurrences of "migrant" in ID
                migrant_count = program_id.count("migrant")
                self.assertLessEqual(
                    migrant_count, 1, f"Program ID {program_id} has been migrated multiple times"
                )


if __name__ == "__main__":
    unittest.main()
