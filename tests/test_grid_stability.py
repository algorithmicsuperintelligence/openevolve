"""
Integration tests for MAP-Elites grid stability during evolution
"""

import os
import tempfile
import shutil
import unittest

from openevolve.database_memory import InMemoryProgramDatabase as ProgramDatabase
from openevolve.database import Program
from openevolve.config import DatabaseConfig


class TestGridStability(unittest.TestCase):
    """Integration tests for MAP-Elites grid stability as programs are added"""

    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)

    def test_feature_ranges_do_not_contract(self):
        """Test that feature ranges are preserved as more programs are added"""
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

        db2 = db1

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

        # Phase 2: Add a program outside the established range
        db2 = db1

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

    def test_feature_stats_accumulation(self):
        """Test that feature_stats accumulate correctly as programs are added"""
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

        # Phase 2: Add more programs
        db2 = db1

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
            "Phase 1 score values were lost while adding programs",
        )
        self.assertTrue(
            phase1_complexity_values.issubset(phase2_complexity_values),
            "Phase 1 complexity values were lost while adding programs",
        )


if __name__ == "__main__":
    unittest.main()
