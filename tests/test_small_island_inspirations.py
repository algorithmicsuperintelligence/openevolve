"""Small islands should still contribute their available inspiration programs."""

import unittest

from openevolve.config import DatabaseConfig
from openevolve.database import Program, ProgramDatabase


class TestSmallIslandInspirations(unittest.TestCase):
    def setUp(self):
        self.db = ProgramDatabase(
            DatabaseConfig(in_memory=True, num_islands=2, elite_selection_ratio=0.2)
        )
        self.parent = Program(id="parent", code="parent", metrics={"score": 1.0})
        self.best = Program(id="best", code="best", metrics={"score": 3.0})
        self.other = Program(id="other", code="other", metrics={"score": 2.0})
        outsider = Program(id="outsider", code="outsider", metrics={"score": 4.0})
        self.db.programs = {p.id: p for p in [self.parent, self.best, self.other, outsider]}
        self.db.islands = [{"parent", "best", "other"}, {"outsider"}]
        self.db.island_best_programs = ["best", "outsider"]

    def test_fills_available_slots_when_island_is_no_larger_than_request(self):
        for requested in (3, 5):
            with self.subTest(requested=requested):
                inspirations = self.db._sample_inspirations(self.parent, n=requested, island_id=0)
                self.assertEqual([p.id for p in inspirations][0], "best")
                self.assertCountEqual([p.id for p in inspirations], ["best", "other"])

    def test_respects_smaller_request(self):
        inspirations = self.db._sample_inspirations(self.parent, n=1, island_id=0)
        self.assertEqual([p.id for p in inspirations], ["best"])

    def test_parent_only_island_has_no_inspirations(self):
        self.db.islands[0] = {"parent"}
        self.db.island_best_programs[0] = "parent"
        self.assertEqual(self.db._sample_inspirations(self.parent, n=5, island_id=0), [])


if __name__ == "__main__":
    unittest.main()
