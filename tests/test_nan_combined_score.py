"""
A NaN `combined_score` is not a usable fitness.

`get_fitness_score` already skips NaN when it averages the other metrics, but it
returned a NaN `combined_score` as it was. NaN compares false against everything,
so a program with that fitness could never be beaten and stayed the best program.
"""

import math
import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test")

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.utils.metrics_utils import get_fitness_score

NAN = float("nan")


class TestNanCombinedScore(unittest.TestCase):
    def test_nan_combined_score_falls_back_to_the_other_metrics(self):
        fitness = get_fitness_score({"combined_score": NAN, "accuracy": 0.5, "speed": 0.25})

        self.assertEqual(fitness, 0.375)

    def test_nan_combined_score_alone_scores_zero(self):
        fitness = get_fitness_score({"combined_score": NAN})

        self.assertEqual(fitness, 0.0)

    def test_valid_combined_score_is_still_used(self):
        fitness = get_fitness_score({"combined_score": 0.8, "accuracy": 0.1})

        self.assertFalse(math.isnan(fitness))
        self.assertEqual(fitness, 0.8)

    def test_better_program_replaces_one_with_a_nan_combined_score(self):
        config = Config()
        config.database.in_memory = True
        config.database.num_islands = 1
        database = ProgramDatabase(config.database)

        database.add(Program(id="nan", code="x = 1", metrics={"combined_score": NAN}))
        database.add(Program(id="good", code="x = 2", metrics={"combined_score": 0.9}))

        self.assertEqual(database.best_program_id, "good")


if __name__ == "__main__":
    unittest.main()
