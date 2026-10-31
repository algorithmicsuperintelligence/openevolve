"""Tests for resuming checkpoints with a different configured island count."""

import json
import os
import tempfile
import unittest

from openevolve.config import DatabaseConfig
from openevolve.database import Program, ProgramDatabase


class TestCheckpointIslands(unittest.TestCase):
    def setUp(self):
        checkpoint = tempfile.TemporaryDirectory()
        self.addCleanup(checkpoint.cleanup)
        self.checkpoint_path = checkpoint.name
        self.database = ProgramDatabase(DatabaseConfig(in_memory=True, num_islands=2))
        for island in range(2):
            self.database.add(self.make_program(island), target_island=island)
        self.database.save(self.checkpoint_path)

    @staticmethod
    def make_program(index):
        return Program(
            id=f"program_{index}",
            code=f"def solve(): return {index}",
            metrics={"combined_score": index + 1.0},
        )

    def load_database(self, num_islands):
        return ProgramDatabase(
            DatabaseConfig(
                in_memory=True,
                num_islands=num_islands,
                db_path=self.checkpoint_path,
            )
        )

    def test_resume_with_more_islands(self):
        resumed = self.load_database(num_islands=4)

        # Exercise both new islands; their feature maps must not share cells.
        for island in (2, 3):
            resumed.add(self.make_program(island), target_island=island)
            self.assertEqual(resumed.islands[island], {f"program_{island}"})
            self.assertEqual(
                set(resumed.island_feature_maps[island].values()), {f"program_{island}"}
            )

        self.assertEqual(resumed.island_feature_maps[:2], self.database.island_feature_maps)
        self.assertEqual(len(resumed.island_feature_maps), 4)
        self.assertIsNot(resumed.island_feature_maps[2], resumed.island_feature_maps[3])
        self.assertEqual(set(resumed.island_feature_maps[2].values()), {"program_2"})

        resumed.save(self.checkpoint_path)
        reloaded = ProgramDatabase(DatabaseConfig(in_memory=True, num_islands=4))
        reloaded.load(self.checkpoint_path)
        self.assertEqual(reloaded.island_feature_maps, resumed.island_feature_maps)
        self.assertEqual(reloaded.islands, resumed.islands)

    def test_resume_preserves_saved_islands(self):
        for num_islands in (1, 2):
            with self.subTest(num_islands=num_islands):
                resumed = self.load_database(num_islands)
                self.assertEqual(resumed.islands, self.database.islands)
                self.assertEqual(resumed.island_feature_maps, self.database.island_feature_maps)
                resumed.add(self.make_program(2), target_island=1)
                self.assertIn("program_2", resumed.islands[1])
                self.assertIn("program_2", resumed.island_feature_maps[1].values())

    def test_resume_without_saved_feature_maps(self):
        metadata_path = os.path.join(self.checkpoint_path, "metadata.json")
        with open(metadata_path) as metadata_file:
            metadata = json.load(metadata_file)
        del metadata["island_feature_maps"]
        with open(metadata_path, "w") as metadata_file:
            json.dump(metadata, metadata_file)

        resumed = self.load_database(num_islands=1)
        resumed.add(self.make_program(2), target_island=1)

        self.assertEqual(len(resumed.island_feature_maps), 2)
        self.assertEqual(resumed.island_feature_maps[0], {})
        self.assertEqual(set(resumed.island_feature_maps[1].values()), {"program_2"})


if __name__ == "__main__":
    unittest.main()
