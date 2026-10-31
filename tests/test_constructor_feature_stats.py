"""Loading through db_path must retain the checkpoint's MAP-Elites scale."""

import tempfile
import unittest

from openevolve.config import DatabaseConfig
from openevolve.database import Program, ProgramDatabase


class TestConstructorFeatureStats(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.config = DatabaseConfig(
            in_memory=True,
            num_islands=1,
            feature_dimensions=["custom"],
            feature_bins=10,
            archive_size=1,
        )
        self.db = ProgramDatabase(self.config)
        self.db._update_feature_stats("custom", 0.0)
        self.db._update_feature_stats("custom", 100.0)
        self.program = Program(
            id="candidate", code="pass", metrics={"combined_score": 0.5, "custom": 25.0}
        )
        self.db.add(self.program)
        self.db.save(self.tempdir.name)
        self.config.db_path = self.tempdir.name

    def test_constructor_preserves_loaded_feature_ranges(self):
        restored = ProgramDatabase(self.config)
        self.assertEqual(restored.feature_stats, self.db.feature_stats)

    def test_constructor_preserves_feature_placement_without_second_load(self):
        expected = self.db._calculate_feature_coords(self.program)
        restored = ProgramDatabase(self.config)
        self.assertEqual(expected, [2])
        self.assertEqual(
            restored._calculate_feature_coords(restored.programs[self.program.id]), expected
        )


if __name__ == "__main__":
    unittest.main()
