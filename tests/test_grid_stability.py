"""
Integration tests for MAP-Elites grid stability across checkpoints
"""

import os
import shutil
import tempfile
import unittest

from openevolve.config import DatabaseConfig
from openevolve.database import Program, ProgramDatabase


class TestGridStability(unittest.TestCase):
    """Integration tests for MAP-Elites grid stability when resuming from checkpoints"""

    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)

    def test_feature_ranges_preserved_across_checkpoints(self):
        """Test that feature ranges are preserved across checkpoint save/load cycles"""
        config = DatabaseConfig(
            db_path=self.test_dir,
            feature_dimensions=["score", "prompt_length", "reasoning_sophistication"],
            feature_bins=5,  # Use smaller bins for easier testing
        )

        # Phase 1: Create initial population with specific range
        db1 = ProgramDatabase(config)

        # Create programs with known metrics to establish ranges
        test_cases = [
            {"combined_score": 0.2, "prompt_length": 100, "reasoning_sophistication": 0.1},
            {"combined_score": 0.5, "prompt_length": 300, "reasoning_sophistication": 0.5},
            {"combined_score": 0.8, "prompt_length": 500, "reasoning_sophistication": 0.9},
        ]

        for i, metrics in enumerate(test_cases):
            program = Program(
                id=f"range_test_{i}", code=f"# Range test program {i}", metrics=metrics
            )
            db1.add(program)

        # Record the established ranges
        original_ranges = {}
        for dim, stats in db1.feature_stats.items():
            original_ranges[dim] = {
                "min": stats["min"],
                "max": stats["max"],
                "value_count": len(stats["values"]),
            }

        # Save checkpoint
        db1.save(self.test_dir, iteration=25)

        # Phase 2: Resume from checkpoint
        db2 = ProgramDatabase(config)
        db2.load(self.test_dir)

        # Restore the active population; displaced cell owners stay evicted.
        self.assertEqual(set(db2.programs), set(db1.programs))
        self.assertEqual(db2.islands, db1.islands)

        # Verify feature ranges are preserved
        for dim, original_range in original_ranges.items():
            self.assertIn(dim, db2.feature_stats)
            loaded_stats = db2.feature_stats[dim]

            self.assertAlmostEqual(
                loaded_stats["min"],
                original_range["min"],
                places=5,
                msg=f"Min range changed for {dim}",
            )
            self.assertAlmostEqual(
                loaded_stats["max"],
                original_range["max"],
                places=5,
                msg=f"Max range changed for {dim}",
            )

        # Phase 3: Add new program within existing range - ranges should not contract
        new_program = Program(
            id="within_range_test",
            code="# New program within established range",
            metrics={
                "combined_score": 0.35,  # Between existing values
                "prompt_length": 200,  # Between existing values
                "reasoning_sophistication": 0.3,  # Between existing values
            },
        )

        # Add new program
        db2.add(new_program)
        new_coords = db2._calculate_feature_coords(new_program)

        # Verify ranges did not contract (should be same or expanded)
        for dim, original_range in original_ranges.items():
            current_stats = db2.feature_stats[dim]

            self.assertLessEqual(
                current_stats["min"], original_range["min"], f"Min range contracted for {dim}"
            )
            self.assertGreaterEqual(
                current_stats["max"], original_range["max"], f"Max range contracted for {dim}"
            )

    def test_grid_expansion_behavior(self):
        """Test that grid expands correctly when new programs exceed existing ranges"""
        config = DatabaseConfig(
            db_path=self.test_dir, feature_dimensions=["score", "execution_time"], feature_bins=5
        )

        # Phase 1: Establish initial range
        db1 = ProgramDatabase(config)

        # Initial programs with limited range
        for i in range(3):
            program = Program(
                id=f"initial_{i}",
                code=f"# Initial program {i}",
                metrics={
                    "combined_score": 0.4 + i * 0.1,  # 0.4 to 0.6
                    "execution_time": 10 + i * 5,  # 10 to 20
                },
            )
            db1.add(program)

        # Record feature ranges
        original_score_min = db1.feature_stats["score"]["min"]
        original_score_max = db1.feature_stats["score"]["max"]
        original_time_min = db1.feature_stats["execution_time"]["min"]
        original_time_max = db1.feature_stats["execution_time"]["max"]

        # Save checkpoint
        db1.save(self.test_dir, iteration=30)

        # Phase 2: Resume and add program outside range
        db2 = ProgramDatabase(config)
        db2.load(self.test_dir)

        # Verify ranges were preserved
        self.assertAlmostEqual(db2.feature_stats["score"]["min"], original_score_min)
        self.assertAlmostEqual(db2.feature_stats["score"]["max"], original_score_max)
        self.assertAlmostEqual(db2.feature_stats["execution_time"]["min"], original_time_min)
        self.assertAlmostEqual(db2.feature_stats["execution_time"]["max"], original_time_max)

        # Add program outside existing range
        expansion_program = Program(
            id="expansion_test",
            code="# Program to test range expansion",
            metrics={
                "combined_score": 0.9,  # Higher than existing max (0.6)
                "execution_time": 50,  # Higher than existing max (20)
            },
        )

        db2.add(expansion_program)

        # Verify ranges expanded appropriately
        self.assertLessEqual(db2.feature_stats["score"]["min"], original_score_min)
        self.assertGreaterEqual(db2.feature_stats["score"]["max"], 0.9)
        self.assertLessEqual(db2.feature_stats["execution_time"]["min"], original_time_min)
        self.assertGreaterEqual(db2.feature_stats["execution_time"]["max"], 50)

    def test_feature_stats_consistency_across_cycles(self):
        """Test that feature_stats remain consistent across multiple save/load cycles"""
        config = DatabaseConfig(
            db_path=self.test_dir, feature_dimensions=["score", "memory_usage"], feature_bins=4
        )

        # Initial program to establish baseline
        reference_program = Program(
            id="reference",
            code="# Reference program for consistency testing",
            metrics={"combined_score": 0.5, "memory_usage": 1024},
        )

        # Cycle 1: Establish initial feature stats
        db1 = ProgramDatabase(config)
        db1.add(reference_program)

        # Record initial feature stats
        cycle1_stats = {}
        for dim, stats in db1.feature_stats.items():
            cycle1_stats[dim] = {"min": stats["min"], "max": stats["max"]}

        db1.save(self.test_dir, iteration=10)

        # Cycle 2: Load and verify stats preservation
        db2 = ProgramDatabase(config)
        db2.load(self.test_dir)

        # Verify feature stats were preserved
        for dim, original_stats in cycle1_stats.items():
            self.assertIn(dim, db2.feature_stats)
            self.assertAlmostEqual(db2.feature_stats[dim]["min"], original_stats["min"])
            self.assertAlmostEqual(db2.feature_stats[dim]["max"], original_stats["max"])

        # Add another program and save again
        db2.add(
            Program(
                id="cycle2_program",
                code="# Cycle 2 program",
                metrics={"combined_score": 0.3, "memory_usage": 512},
            )
        )

        # Record expanded stats after adding new program
        cycle2_stats = {}
        for dim, stats in db2.feature_stats.items():
            cycle2_stats[dim] = {"min": stats["min"], "max": stats["max"]}

        db2.save(self.test_dir, iteration=20)

        # Cycle 3: Verify stats are still preserved
        db3 = ProgramDatabase(config)
        db3.load(self.test_dir)

        # Verify expanded feature stats were preserved
        for dim, cycle2_stats_dim in cycle2_stats.items():
            self.assertIn(dim, db3.feature_stats)
            self.assertAlmostEqual(
                db3.feature_stats[dim]["min"],
                cycle2_stats_dim["min"],
                msg=f"Min value changed for {dim} in cycle 3",
            )
            self.assertAlmostEqual(
                db3.feature_stats[dim]["max"],
                cycle2_stats_dim["max"],
                msg=f"Max value changed for {dim} in cycle 3",
            )

    def test_feature_stats_accumulation(self):
        """Test that feature_stats accumulate correctly across checkpoint cycles"""
        config = DatabaseConfig(
            db_path=self.test_dir, feature_dimensions=["score", "complexity"], feature_bins=10
        )

        # Cycle 1: Initial programs
        db1 = ProgramDatabase(config)

        for i in range(3):
            program = Program(
                id=f"phase1_{i}",
                code=f"# Phase 1 program {i}",
                metrics={"combined_score": 0.2 + i * 0.2, "complexity": 100 + i * 50},
            )
            db1.add(program)

        # Record phase 1 stats
        phase1_score_values = set(db1.feature_stats["score"]["values"])
        phase1_complexity_values = set(db1.feature_stats["complexity"]["values"])

        db1.save(self.test_dir, iteration=15)

        # Cycle 2: Load and add more programs
        db2 = ProgramDatabase(config)
        db2.load(self.test_dir)

        for i in range(2):
            program = Program(
                id=f"phase2_{i}",
                code=f"# Phase 2 program {i}",
                metrics={"combined_score": 0.1 + i * 0.3, "complexity": 75 + i * 75},
            )
            db2.add(program)

        # Verify that phase 1 values are still present
        phase2_score_values = set(db2.feature_stats["score"]["values"])
        phase2_complexity_values = set(db2.feature_stats["complexity"]["values"])

        # Phase 1 values should be preserved (subset relationship)
        self.assertTrue(
            phase1_score_values.issubset(phase2_score_values),
            "Phase 1 score values were lost after loading checkpoint",
        )
        self.assertTrue(
            phase1_complexity_values.issubset(phase2_complexity_values),
            "Phase 1 complexity values were lost after loading checkpoint",
        )


if __name__ == "__main__":
    unittest.main()
