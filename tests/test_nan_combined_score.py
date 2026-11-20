"""
A NaN `combined_score` must rank as the worst score.

NaN is neither greater nor smaller than any number, so a program with a NaN fitness
could never be beaten: if it was added first it stayed the best program. It must not
outrank valid programs either, so `get_fitness_score` ranks it below every score.
"""

import os
import unittest

import numpy as np

os.environ.setdefault("OPENAI_API_KEY", "test")

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.utils.metrics_utils import get_fitness_score

NAN = float("nan")


def make_database() -> ProgramDatabase:
    config = Config()
    config.database.in_memory = True
    config.database.num_islands = 1
    return ProgramDatabase(config.database)


class TestNanCombinedScore(unittest.TestCase):
    def test_nan_combined_score_ranks_below_every_score(self):
        for nan in (NAN, np.nan, np.float32("nan")):
            with self.subTest(nan=repr(nan)):
                fitness = get_fitness_score({"combined_score": nan, "accuracy": 0.9})

                self.assertEqual(fitness, float("-inf"))
                self.assertLess(fitness, get_fitness_score({"combined_score": -1e9}))

    def test_valid_combined_score_is_unchanged(self):
        self.assertEqual(get_fitness_score({"combined_score": 0.8, "accuracy": 0.1}), 0.8)
        self.assertEqual(get_fitness_score({"combined_score": float("inf")}), float("inf"))

    def test_better_program_replaces_one_with_a_nan_combined_score(self):
        database = make_database()

        database.add(Program(id="nan", code="x = 1", metrics={"combined_score": NAN}))
        database.add(Program(id="good", code="x = 2", metrics={"combined_score": 0.9}))

        self.assertEqual(database.best_program_id, "good")

    def test_nan_combined_score_does_not_replace_a_valid_program(self):
        database = make_database()

        database.add(
            Program(id="good", code="x = 1", metrics={"combined_score": 0.9, "eval_time": 0.4})
        )
        database.add(
            Program(
                id="nan",
                code="x = 2",
                metrics={"combined_score": NAN, "sum_radii": 2.6, "eval_time": 3.0},
            )
        )

        self.assertEqual(database.best_program_id, "good")


if __name__ == "__main__":
    unittest.main()
