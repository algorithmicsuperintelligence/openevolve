"""Regression tests for issue #437: island-aware inspiration sampling."""

import unittest
from unittest.mock import patch

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase


class TestSampleFromIslandInspirations(unittest.TestCase):
    def setUp(self):
        config = Config()
        config.database.in_memory = True
        config.database.num_islands = 2
        config.database.exploration_ratio = 1.0
        config.database.exploitation_ratio = 0.0
        self.db = ProgramDatabase(config.database)

        self.parent = Program(
            id="parent",
            code="def parent(): pass",
            metrics={"score": 0.5},
            metadata={"island": 1},
        )
        self.elite = Program(
            id="elite",
            code="def elite(): pass",
            metrics={"score": 1.0},
            metadata={"island": 1},
        )

        self.db.programs = {
            self.parent.id: self.parent,
            self.elite.id: self.elite,
        }
        self.db.islands = [set(), {self.parent.id, self.elite.id}]
        self.db.island_best_programs = [None, self.elite.id]

    def test_sample_from_island_reuses_strategy_aware_inspiration_sampler(self):
        with (
            patch.object(
                self.db,
                "_sample_from_island_random",
                return_value=self.parent,
            ),
            patch.object(
                self.db,
                "_sample_inspirations",
                return_value=[self.elite],
            ) as sample_inspirations,
        ):
            parent, inspirations = self.db.sample_from_island(
                island_id=1,
                num_inspirations=2,
            )

        self.assertIs(parent, self.parent)
        self.assertEqual(inspirations, [self.elite])
        sample_inspirations.assert_called_once_with(
            self.parent,
            n=2,
            island_id=1,
        )

    def test_explicit_island_overrides_parent_metadata_for_inspirations(self):
        island_zero_best = Program(
            id="island-zero-best",
            code="def island_zero(): pass",
            metrics={"score": 2.0},
            metadata={"island": 0},
        )
        parent_from_zero = Program(
            id="parent-from-zero",
            code="def parent_zero(): pass",
            metrics={"score": 0.1},
            metadata={"island": 0},
        )
        island_one_best = Program(
            id="island-one-best",
            code="def island_one(): pass",
            metrics={"score": 1.0},
            metadata={"island": 1},
        )

        self.db.programs = {
            parent_from_zero.id: parent_from_zero,
            island_zero_best.id: island_zero_best,
            island_one_best.id: island_one_best,
        }
        self.db.islands = [
            {parent_from_zero.id, island_zero_best.id},
            {island_one_best.id},
        ]
        self.db.island_best_programs = [
            island_zero_best.id,
            island_one_best.id,
        ]

        inspirations = self.db._sample_inspirations(
            parent_from_zero,
            n=1,
            island_id=1,
        )

        self.assertEqual([program.id for program in inspirations], [island_one_best.id])
        self.assertTrue(all(program.metadata.get("island") == 1 for program in inspirations))

    def test_zero_inspirations_is_forwarded_without_random_sampling(self):
        with (
            patch.object(
                self.db,
                "_sample_from_island_random",
                return_value=self.parent,
            ),
            patch.object(
                self.db,
                "_sample_inspirations",
                return_value=[],
            ) as sample_inspirations,
        ):
            _, inspirations = self.db.sample_from_island(
                island_id=1,
                num_inspirations=0,
            )

        self.assertEqual(inspirations, [])
        sample_inspirations.assert_called_once_with(
            self.parent,
            n=0,
            island_id=1,
        )


if __name__ == "__main__":
    unittest.main()
