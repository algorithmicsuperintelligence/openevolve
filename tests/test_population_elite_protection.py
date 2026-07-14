"""
Tests for MAP-Elites cell-owner protection during population eviction.

Regression tests for the bug where `_enforce_population_limit()` sorted all
programs globally by fitness and evicted the worst, ignoring whether a program
owned a MAP-Elites cell. This allowed a low-scoring but diverse cell owner to be
removed before a higher-scoring homeless (non-cell-owning) program, defeating the
diversity-preservation purpose of MAP-Elites.

See: https://github.com/algorithmicsuperintelligence/openevolve/issues/454
"""

import unittest

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase


class TestPopulationEliteProtection(unittest.TestCase):
    def setUp(self):
        config = Config()
        config.database.in_memory = True
        config.database.num_islands = 1
        config.database.feature_bins = 5
        self.config = config

    def _make_db(self, population_size):
        self.config.database.population_size = population_size
        return ProgramDatabase(self.config.database)

    def _add_raw(self, db, pid, fitness, island=0, cell_key=None):
        """Insert a program directly into the DB structures for controlled tests.

        If cell_key is provided the program is registered as the owner of that
        MAP-Elites cell (i.e. it is an "elite"); otherwise it is homeless.
        """
        prog = Program(
            id=pid,
            code=f"# program {pid}",
            language="python",
            metrics={"combined_score": fitness},
            metadata={"island": island},
        )
        db.programs[pid] = prog
        db.islands[island].add(pid)
        if cell_key is not None:
            db.island_feature_maps[island][cell_key] = pid
        return prog

    def test_homeless_removed_before_low_scoring_elite(self):
        """A low-scoring cell owner must survive over higher-scoring homeless programs."""
        db = self._make_db(population_size=2)
        # Elite owns a cell but has the WORST fitness.
        self._add_raw(db, "elite_low", fitness=0.1, cell_key="0")
        # Homeless programs have higher fitness but own no cell.
        self._add_raw(db, "homeless_high", fitness=0.9)
        self._add_raw(db, "homeless_mid", fitness=0.5)

        db._enforce_population_limit()

        # One removal needed (3 -> 2): the worst homeless should go, elite stays.
        self.assertEqual(len(db.programs), 2)
        self.assertIn("elite_low", db.programs, "Low-scoring cell owner must be protected")
        self.assertNotIn("homeless_mid", db.programs, "Worst homeless should be evicted first")
        self.assertIn("homeless_high", db.programs)

    def test_all_homeless_removed_before_any_elite(self):
        """Every homeless program is evicted before any elite, regardless of score."""
        db = self._make_db(population_size=2)
        self._add_raw(db, "elite_a", fitness=0.2, cell_key="0")
        self._add_raw(db, "elite_b", fitness=0.3, cell_key="1")
        self._add_raw(db, "homeless_a", fitness=0.99)
        self._add_raw(db, "homeless_b", fitness=0.98)

        db._enforce_population_limit()  # 4 -> 2, remove 2

        self.assertEqual(len(db.programs), 2)
        self.assertIn("elite_a", db.programs)
        self.assertIn("elite_b", db.programs)
        self.assertNotIn("homeless_a", db.programs)
        self.assertNotIn("homeless_b", db.programs)

    def test_falls_back_to_evicting_worst_elite(self):
        """When homeless removals are insufficient, the worst elite is evicted."""
        db = self._make_db(population_size=1)
        self._add_raw(db, "elite_worst", fitness=0.2, cell_key="0")
        self._add_raw(db, "elite_best", fitness=0.8, cell_key="1")

        db._enforce_population_limit()  # 2 -> 1, no homeless, must drop one elite

        self.assertEqual(len(db.programs), 1)
        self.assertIn("elite_best", db.programs, "Better elite must survive")
        self.assertNotIn("elite_worst", db.programs)

    def test_best_program_never_evicted(self):
        """The globally-tracked best program is protected even if homeless."""
        db = self._make_db(population_size=1)
        # Homeless program marked as the global best, with the worst score.
        self._add_raw(db, "the_best", fitness=0.05)
        db.best_program_id = "the_best"
        self._add_raw(db, "homeless_high", fitness=0.9)

        db._enforce_population_limit()  # 2 -> 1

        self.assertEqual(len(db.programs), 1)
        self.assertIn("the_best", db.programs, "best_program_id must never be evicted")

    def test_no_op_when_under_limit(self):
        """Nothing is removed when population is within the limit."""
        db = self._make_db(population_size=5)
        self._add_raw(db, "a", fitness=0.1, cell_key="0")
        self._add_raw(db, "b", fitness=0.2)

        db._enforce_population_limit()

        self.assertEqual(len(db.programs), 2)

    def test_removed_program_purged_from_all_structures(self):
        """An evicted program is removed from islands, feature maps and archive."""
        db = self._make_db(population_size=1)
        self._add_raw(db, "elite", fitness=0.9, cell_key="0")
        homeless = self._add_raw(db, "homeless", fitness=0.1)
        db.archive.add("homeless")

        db._enforce_population_limit()  # removes homeless

        self.assertNotIn("homeless", db.programs)
        self.assertNotIn("homeless", db.islands[0])
        self.assertNotIn("homeless", db.archive)
        self.assertNotIn("homeless", db.island_feature_maps[0].values())


if __name__ == "__main__":
    unittest.main()
