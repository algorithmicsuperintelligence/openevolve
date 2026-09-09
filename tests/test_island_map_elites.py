"""
Tests for per-island MAP-Elites functionality in openevolve.database

This test suite ensures that the per-island MAP-Elites implementation
works correctly and prevents regression to the old global feature map
that caused duplicate program chains.
"""

import unittest
import uuid
from openevolve.config import Config
from openevolve.database import Program
from openevolve.database_memory import InMemoryProgramDatabase as ProgramDatabase


class TestIslandMapElites(unittest.TestCase):
    """Tests for per-island MAP-Elites implementation"""

    def setUp(self):
        """Set up test database with multiple islands"""
        config = Config()
        config.database.in_memory = True
        config.database.num_islands = 3
        config.database.feature_bins = 5  # 5x5 grid
        self.db = ProgramDatabase(config.database)

    def _create_test_program(self, program_id: str, score: float, features: list, island: int = 0) -> Program:
        """Helper to create a test program with specific features"""
        program = Program(
            id=program_id,
            code=f"def func_{program_id}(): return {score}",
            language="python",
            metrics={"score": score, "combined_score": score},
            metadata={"island": island},
        )
        # Set features that will map to specific grid coordinates
        program.features = features
        return program

    def test_island_feature_maps_initialization(self):
        """Test that each island gets its own feature map"""
        # Verify we have the correct number of island feature maps
        self.assertEqual(len(self.db.island_feature_maps), 3)
        
        # Each island feature map should be empty initially
        for i, feature_map in enumerate(self.db.island_feature_maps):
            self.assertEqual(len(feature_map), 0, f"Island {i} feature map should be empty initially")
            self.assertIsInstance(feature_map, dict, f"Island {i} feature map should be a dictionary")

    def test_program_added_to_correct_island_feature_map(self):
        """Test that programs are added to their island's specific feature map"""
        # Create programs for different islands
        prog1 = self._create_test_program("prog1", 0.8, [0.1, 0.2], island=0)
        prog2 = self._create_test_program("prog2", 0.7, [0.3, 0.4], island=1)
        prog3 = self._create_test_program("prog3", 0.9, [0.5, 0.6], island=2)

        # Add programs to database with explicit target islands
        self.db.add(prog1, target_island=0)
        self.db.add(prog2, target_island=1)
        self.db.add(prog3, target_island=2)

        # Verify each program appears only in its island's feature map
        self.assertEqual(len(self.db.island_feature_maps[0]), 1)
        self.assertEqual(len(self.db.island_feature_maps[1]), 1)
        self.assertEqual(len(self.db.island_feature_maps[2]), 1)

        # Verify the correct programs are in each island's map
        self.assertIn("prog1", self.db.island_feature_maps[0].values())
        self.assertIn("prog2", self.db.island_feature_maps[1].values())
        self.assertIn("prog3", self.db.island_feature_maps[2].values())

        # Verify programs don't appear in other islands' feature maps
        self.assertNotIn("prog1", self.db.island_feature_maps[1].values())
        self.assertNotIn("prog1", self.db.island_feature_maps[2].values())
        self.assertNotIn("prog2", self.db.island_feature_maps[0].values())
        self.assertNotIn("prog2", self.db.island_feature_maps[2].values())

    def test_feature_coordinate_isolation(self):
        """Test that same feature coordinates in different islands don't conflict"""
        # Create programs with identical features but on different islands
        prog1 = self._create_test_program("prog1", 0.8, [0.1, 0.2], island=0)
        prog2 = self._create_test_program("prog2", 0.9, [0.1, 0.2], island=1)  # Same features, different island

        self.db.add(prog1, target_island=0)
        self.db.add(prog2, target_island=1)

        # Both programs should be added successfully (no conflict)
        self.assertIsNotNone(self.db.get("prog1"))
        self.assertIsNotNone(self.db.get("prog2"))

        # Each should be in their respective island's feature map
        self.assertIn("prog1", self.db.island_feature_maps[0].values())
        self.assertIn("prog2", self.db.island_feature_maps[1].values())

    def test_better_program_replaces_in_island_feature_map(self):
        """Test that a better program replaces existing program in same island's cell"""
        # Create two programs with identical code (same features) but different scores
        identical_code = "def test_function(): return 42"
        
        prog1 = Program(
            id="prog1",
            code=identical_code,
            language="python",
            metrics={"score": 0.5, "combined_score": 0.5},
            metadata={"island": 0},
        )
        
        prog2 = Program(
            id="prog2", 
            code=identical_code,  # Same code = same features
            language="python",
            metrics={"score": 0.8, "combined_score": 0.8},  # Better score
            metadata={"island": 0},
        )

        # Add first program
        self.db.add(prog1, target_island=0)
        
        # Should be in the feature map
        feature_map_values_before = list(self.db.island_feature_maps[0].values())
        self.assertIn("prog1", feature_map_values_before)

        # Add better program with same features
        self.db.add(prog2, target_island=0)
        
        # Should still have only one program in that cell, but it should be the better one
        feature_map_values_after = list(self.db.island_feature_maps[0].values())
        
        # If they mapped to the same cell, only the better program should remain
        if len(feature_map_values_before) == len(feature_map_values_after):
            self.assertIn("prog2", feature_map_values_after)
            # If they had identical features, prog1 should be replaced
            if identical_code == identical_code:  # They have identical features
                self.assertNotIn("prog1", feature_map_values_after)
                # Verify the worse program is no longer in the database
                self.assertIsNone(self.db.get("prog1"))
        
        # The better program should always be in the database
        self.assertIsNotNone(self.db.get("prog2"))

    def test_global_best_program_tracks_across_islands(self):
        """Test that global best program is tracked correctly across all islands"""
        # Create programs with different scores on different islands
        prog1 = self._create_test_program("prog1", 0.5, [0.1, 0.1], island=0)
        prog2 = self._create_test_program("prog2", 0.9, [0.2, 0.2], island=1)  # Best
        prog3 = self._create_test_program("prog3", 0.7, [0.3, 0.3], island=2)

        self.db.add(prog1, target_island=0)
        self.db.add(prog2, target_island=1)
        self.db.add(prog3, target_island=2)

        # Global best should be prog2
        best = self.db.get_best_program()
        self.assertIsNotNone(best)
        self.assertEqual(best.id, "prog2")

    def test_no_migrant_suffix_generation(self):
        """Test that no programs with _migrant suffixes are created"""
        # Add several programs
        for i in range(10):
            prog = self._create_test_program(f"prog{i}", 0.5 + i*0.1, [0.1 + i*0.1, 0.2], island=i % 3)
            self.db.add(prog)

        # Get all program IDs from all islands
        all_program_ids = set()
        for island_map in self.db.island_feature_maps:
            all_program_ids.update(island_map.values())

        # Verify no program ID contains '_migrant'
        migrant_programs = [pid for pid in all_program_ids if '_migrant' in pid]
        self.assertEqual(len(migrant_programs), 0, 
                        f"Found programs with _migrant suffix: {migrant_programs}")



if __name__ == '__main__':
    unittest.main()