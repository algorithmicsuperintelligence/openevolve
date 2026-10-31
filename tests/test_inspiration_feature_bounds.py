"""Neighborhood sampling must use the bounds of each MAP-Elites dimension."""

import unittest
from unittest.mock import patch

from openevolve.config import DatabaseConfig
from openevolve.database import Program, ProgramDatabase


class TestInspirationFeatureBounds(unittest.TestCase):
    def test_neighbor_in_wide_dimension_is_selected_before_random_fallback(self):
        db = ProgramDatabase(
            DatabaseConfig(
                in_memory=True,
                num_islands=1,
                feature_dimensions=["x", "y"],
                feature_bins={"x": 20, "y": 2},
                elite_selection_ratio=0.2,
            )
        )
        db.feature_stats = {
            dim: {"min": 0.0, "max": 1.0, "values": [0.0, 1.0]} for dim in ("x", "y")
        }
        parent = Program(id="parent", code="parent", metrics={"score": 0.1, "x": 0.9, "y": 0.75})
        neighbor = Program(
            id="neighbor", code="neighbor", metrics={"score": 0.2, "x": 0.99, "y": 0.75}
        )
        best = Program(id="best", code="best", metrics={"score": 1.0, "x": 0.0, "y": 0.0})
        far = Program(id="far", code="far", metrics={"score": 0.3, "x": 0.0, "y": 0.75})
        db.programs = {p.id: p for p in (parent, neighbor, best, far)}
        db.islands = [set(db.programs)]
        db.island_best_programs = [best.id]
        self.assertEqual(db._calculate_feature_coords(parent), [18, 1])
        self.assertEqual(db._calculate_feature_coords(neighbor), [19, 1])

        with (
            patch("openevolve.database.random.randint", return_value=2),
            patch("openevolve.database.random.sample", return_value=[far.id]) as fallback,
        ):
            inspirations = db._sample_inspirations(parent, n=2, island_id=0)

        self.assertEqual([p.id for p in inspirations], [best.id, neighbor.id])
        fallback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
