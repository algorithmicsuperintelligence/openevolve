"""
Tests for removal of orphaned ("zombie") programs displaced from MAP-Elites cells.

Regression tests for the bug where replacing a cell owner in an island removed the
old program from the island set but NOT from `self.programs`. The displaced program
then lingered forever: it owned no cell and belonged to no island (so it could never
be sampled for prompts) yet still consumed a population slot.

See: https://github.com/algorithmicsuperintelligence/openevolve/issues/454
"""

import unittest

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase


def _make_program(pid, fitness, island=0):
    return Program(
        id=pid,
        code=f"# program {pid} score {fitness}",
        language="python",
        metrics={"combined_score": fitness},
        metadata={"island": island},
    )


class TestRemoveProgramIfOrphaned(unittest.TestCase):
    """Directly exercise the _remove_program_if_orphaned helper."""

    def setUp(self):
        config = Config()
        config.database.in_memory = True
        config.database.num_islands = 2
        config.database.feature_bins = 5
        config.database.population_size = 100
        self.db = ProgramDatabase(config.database)

    def _register(self, pid, fitness, island=None, cell_key=None):
        prog = _make_program(pid, fitness, island=island if island is not None else 0)
        self.db.programs[pid] = prog
        if island is not None:
            self.db.islands[island].add(pid)
        if cell_key is not None:
            self.db.island_feature_maps[island][cell_key] = pid
        return prog

    def test_true_orphan_is_removed(self):
        # In programs, but in no island and owning no cell.
        self._register("orphan", 0.5)
        self.db.archive.add("orphan")

        self.db._remove_program_if_orphaned("orphan")

        self.assertNotIn("orphan", self.db.programs)
        self.assertNotIn("orphan", self.db.archive)

    def test_cell_owner_is_kept(self):
        self._register("owner", 0.5, island=0, cell_key="0")

        self.db._remove_program_if_orphaned("owner")

        self.assertIn("owner", self.db.programs, "A program owning a cell must not be removed")

    def test_island_member_is_kept(self):
        # In an island set but owning no cell (shouldn't happen often, but must be safe).
        self._register("member", 0.5, island=1)

        self.db._remove_program_if_orphaned("member")

        self.assertIn("member", self.db.programs, "A program in an island must not be removed")

    def test_cell_owner_in_other_island_is_kept(self):
        # Owns a cell in island 1; must be kept even though absent from island 0.
        self._register("multi", 0.5, island=1, cell_key="3")

        self.db._remove_program_if_orphaned("multi")

        self.assertIn("multi", self.db.programs)

    def test_missing_program_is_noop(self):
        # Should not raise if the id is unknown.
        self.db._remove_program_if_orphaned("does_not_exist")
        self.assertNotIn("does_not_exist", self.db.programs)

    def test_orphan_removal_clears_stale_island_best(self):
        self._register("orphan", 0.5)
        self.db.island_best_programs[0] = "orphan"

        self.db._remove_program_if_orphaned("orphan")

        self.assertNotIn("orphan", self.db.programs)
        self.assertIsNone(
            self.db.island_best_programs[0],
            "Stale island-best reference to a removed program must be cleared",
        )


def _cell_program(pid, fitness):
    """A program with a CONSTANT custom feature value, so every such program maps
    to the same MAP-Elites cell regardless of the configured bin count. Fitness is
    carried by combined_score (feature dimensions are excluded from fitness)."""
    return Program(
        id=pid,
        code=f"# program {pid} score {fitness}",
        language="python",
        metrics={"combined_score": fitness, "cell": 1.0},
        metadata={"island": 0},
    )


class TestNoZombiesThroughAdd(unittest.TestCase):
    """End-to-end: displaced programs must not accumulate through the public add() path."""

    def _make_db(self):
        config = Config()
        config.database.in_memory = True
        config.database.num_islands = 1
        # A single custom feature dimension with a constant value places every
        # program in the same cell, so each better program deterministically
        # replaces the previous owner.
        config.database.feature_dimensions = ["cell"]
        config.database.population_size = 100  # high, so eviction never interferes
        return ProgramDatabase(config.database)

    def test_displaced_program_removed_on_replacement(self):
        db = self._make_db()
        db.add(_cell_program("A", 0.5), target_island=0)
        self.assertIn("A", db.programs)

        # B is better and lands in the same cell, displacing A.
        db.add(_cell_program("B", 0.9), target_island=0)

        self.assertIn("B", db.programs)
        self.assertNotIn("A", db.programs, "Displaced program A must not linger as a zombie")
        self.assertEqual(len(db.programs), 1)

    def test_no_zombie_accumulation_over_many_replacements(self):
        db = self._make_db()
        # Each program strictly improves and replaces the previous owner of the cell.
        for i in range(20):
            db.add(_cell_program(f"p{i}", fitness=0.1 + i * 0.01), target_island=0)

        # Only the final (best) program should remain; all others were displaced.
        self.assertEqual(
            len(db.programs), 1, "Displaced programs must not accumulate across replacements"
        )
        self.assertIn("p19", db.programs)
        self.assertEqual(db.best_program_id, "p19")

    def test_displaced_program_removed_from_archive(self):
        db = self._make_db()
        db.add(_cell_program("A", 0.5), target_island=0)
        db.archive.add("A")
        db.add(_cell_program("B", 0.9), target_island=0)

        self.assertNotIn("A", db.programs)
        self.assertNotIn("A", db.archive, "Displaced program must also leave the archive")

    def test_worse_program_does_not_displace_owner(self):
        db = self._make_db()
        db.add(_cell_program("A", 0.9), target_island=0)
        # B is worse, so it should NOT replace A; A stays the cell owner.
        db.add(_cell_program("B", 0.1), target_island=0)

        self.assertIn("A", db.programs, "Better incumbent must remain the cell owner")
        self.assertIn("A", db.island_feature_maps[0].values(), "A must still own the cell")
        # B loses the cell but is still a legitimate (samplable) homeless population
        # member - it is NOT a zombie and must not be removed here.
        self.assertIn("B", db.programs)
        self.assertIn("B", db.islands[0])
        self.assertNotIn("B", db.island_feature_maps[0].values())
        self.assertEqual(len(db.programs), 2)


if __name__ == "__main__":
    unittest.main()
