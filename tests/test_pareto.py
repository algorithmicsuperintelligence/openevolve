"""Pareto selection (config.database.objectives): dominance, fronts, crowding, and the database's use of them."""
import math
import tempfile
import unittest

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.pareto import crowding_distance, dominates, non_dominated_sort, objective_values, pareto_keys


class TestParetoModule(unittest.TestCase):
    def test_dominance(self):
        self.assertTrue(dominates((2, 2), (1, 2)))
        self.assertFalse(dominates((2, 1), (1, 2)))       # trade-off: neither dominates
        self.assertFalse(dominates((1, 2), (2, 1)))
        self.assertFalse(dominates((1, 1), (1, 1)))       # equal: not strictly better anywhere

    def test_fronts(self):
        pts = [(1, 1), (2, 2), (3, 1), (1, 3), (0, 0)]
        fronts = non_dominated_sort(pts)
        self.assertEqual(sorted(fronts[0]), [1, 2, 3])     # (2,2), (3,1), (1,3) are mutually non-dominated
        self.assertEqual(fronts[1], [0])                   # (1,1) is dominated by (2,2) only
        self.assertEqual(fronts[2], [4])

    def test_crowding_boundaries_and_interior(self):
        cd = crowding_distance([(0, 3), (1, 2), (2, 1), (3, 0)])
        self.assertTrue(math.isinf(cd[0]) and math.isinf(cd[3]))
        self.assertAlmostEqual(cd[1], (2 - 0) / 3 + (3 - 1) / 3)
        self.assertAlmostEqual(cd[1], cd[2])
        self.assertEqual(crowding_distance([(1, 1)]), [math.inf])

    def test_directions_and_missing(self):
        self.assertEqual(objective_values({"a": 1.0, "b": 2.0}, ["a", "b"], ["max", "min"]), (1.0, -2.0))
        self.assertIsNone(objective_values({"a": 1.0}, ["a", "b"]))
        self.assertIsNone(objective_values({"a": True, "b": 1.0}, ["a", "b"]))

    def test_keys_order_rank_before_crowding(self):
        keys = pareto_keys([("p", (2, 2)), ("q", (3, 1)), ("r", (1, 1)), ("s", None)])
        self.assertEqual(keys["p"][0], 0); self.assertEqual(keys["q"][0], 0); self.assertEqual(keys["r"][0], 1)
        self.assertGreater(min(keys["p"][2], keys["q"][2]), keys["r"][2])   # any rank-0 key beats any rank-1 key
        self.assertGreater(keys["r"][2], keys["s"][2])                       # missing objectives rank last


def _db(objectives=None, **kw):
    config = Config()
    config.database.in_memory = True
    config.database.num_islands = 1
    config.database.feature_dimensions = ["complexity"]
    if objectives:
        config.database.objectives = objectives
    for k, v in kw.items():
        setattr(config.database, k, v)
    return ProgramDatabase(config.database)


def _prog(pid, **metrics):
    return Program(id=pid, code=f"def f_{pid}(): pass  # " + "x" * len(pid), language="python", metrics=metrics)


class TestParetoDatabase(unittest.TestCase):
    def test_single_fitness_unchanged_without_objectives(self):
        db = _db()
        a, b = _prog("a", combined_score=1.0), _prog("b", combined_score=2.0)
        db.add(a); db.add(b)
        self.assertTrue(db._is_better(b, a))
        self.assertEqual(db.get_best_program().id, "b")
        self.assertEqual(db.pareto_front(), [])

    def test_dominance_decides(self):
        db = _db(["primary", "robust"])
        a, b = _prog("a", primary=1.0, robust=1.0), _prog("b", primary=2.0, robust=1.0)
        db.add(a); db.add(b)
        self.assertTrue(db._is_better(b, a)); self.assertFalse(db._is_better(a, b))

    def test_tradeoff_uses_rank_then_crowding(self):
        db = _db(["primary", "robust"])
        for pid, p, r in [("a", 3.0, 0.0), ("b", 2.0, 2.0), ("c", 2.1, 1.9), ("d", 0.0, 3.0), ("e", 1.0, 1.0)]:
            db.add(_prog(pid, primary=p, robust=r))
        front = {p.id for p in db.pareto_front()}
        self.assertEqual(front, {"a", "b", "c", "d"})
        self.assertEqual(db.pareto_rank(db.programs["e"])[0], 1)
        # boundary members (best on one objective) are the least crowded: they outrank the interior pair
        self.assertGreater(db.fitness(db.programs["a"]), db.fitness(db.programs["b"]))
        self.assertTrue(db._is_better(db.programs["a"], db.programs["e"]))    # rank 0 beats rank 1 without dominance
        self.assertFalse(db._is_better(db.programs["e"], db.programs["a"]))

    def test_best_selection_rules_and_front_file(self):
        db = _db(["primary", "robust"])
        db.add(_prog("a", primary=3.0, robust=0.0)); db.add(_prog("b", primary=0.0, robust=3.0)); db.add(_prog("c", primary=1.0, robust=1.0))
        db.add(_prog("d", primary=0.5, robust=0.5))                        # dominated by c
        self.assertEqual(db.get_best_program().id, "a")                     # first_objective (default): best primary on the front
        db.config.best_selection = "knee"
        self.assertEqual(db.get_best_program().id, "c")                     # knee: the balanced member (1,1), closest to the ideal (3,3) after normalization
        db.config.best_selection = "crowding"
        self.assertIn(db.get_best_program().id, {"a", "b"})                  # both boundary points have infinite crowding
        with tempfile.TemporaryDirectory() as d:
            db.save(d)
            import json, os
            front = json.load(open(os.path.join(d, "pareto_front.json")))
            self.assertEqual({f["id"] for f in front["front"]}, {"a", "b", "c"})   # (1,1) is dominated by neither boundary point
            self.assertEqual(front["objectives"], ["primary", "robust"])

    def test_missing_objective_ranks_last(self):
        db = _db(["primary", "robust"])
        db.add(_prog("a", primary=1.0, robust=1.0)); db.add(_prog("b", primary=5.0))
        self.assertTrue(db._is_better(db.programs["a"], db.programs["b"]))
        self.assertEqual(db.get_best_program().id, "a")


if __name__ == "__main__":
    unittest.main()
