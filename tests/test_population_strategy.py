"""Decision hooks choose outcomes while ProgramDatabase owns mutations."""

import asyncio
import tempfile
import unittest
from concurrent.futures import Future
from enum import IntEnum
from unittest.mock import patch

import numpy as np

from openevolve import run_evolution
from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.population import ArchiveDecision, MigrationMove, PopulationStrategy
from openevolve.process_parallel import ProcessParallelController, SerializableResult


def program(pid, score, code=None):
    return Program(
        id=pid,
        code=code or f"def solve(): return '{pid}'",
        metrics={"combined_score": score},
    )


class TestPopulationStrategy(unittest.TestCase):
    def setUp(self):
        self.config = Config().database
        self.config.num_islands = 2
        self.config.in_memory = True

    def test_admission_can_reject_without_mutating_database(self):
        seen = []

        def admit(snapshot, candidate, island):
            seen.append((candidate.id, island, len(snapshot.programs)))
            with self.assertRaises(TypeError):
                snapshot.programs["changed"] = candidate
            return candidate.id != "rejected"

        db = ProgramDatabase(self.config, PopulationStrategy(admit=admit))
        db.add(program("accepted", 0.5))
        self.assertIsNone(db.add(program("rejected", 0.9)))

        self.assertEqual(seen, [("accepted", 0, 0), ("rejected", 0, 1)])
        self.assertEqual(set(db.programs), {"accepted"})
        self.assertNotIn("rejected", db.islands[0])

    def test_rejected_child_does_not_advance_generation_or_store_artifacts(self):
        config = Config()
        config.database.num_islands = 2
        config.evaluator.parallel_evaluations = 1
        config.checkpoint_interval = 1
        db = ProgramDatabase(
            config.database,
            PopulationStrategy(
                admit=lambda snapshot, candidate, island: candidate.id != "rejected"
            ),
        )
        db.add(program("seed", 0.5))
        controller = ProcessParallelController(config, "unused.py", db)
        controller.executor = object()

        def submit(iteration, island):
            future = Future()
            future.set_result(
                SerializableResult(
                    child_program_dict=program("rejected", 0.9).to_dict(),
                    target_island=island,
                    artifacts={"output": "should not be stored"},
                )
            )
            return future

        checkpoints = []
        with (
            patch.object(controller, "_submit_iteration", side_effect=submit),
            patch.object(db, "store_artifacts") as store_artifacts,
        ):
            asyncio.run(controller.run_evolution(1, 1, checkpoint_callback=checkpoints.append))

        self.assertEqual(set(db.programs), {"seed"})
        self.assertEqual(db.island_generations, [0, 0])
        self.assertEqual(db.last_iteration, 1)
        self.assertEqual(checkpoints, [1])
        store_artifacts.assert_not_called()

        def fail_checkpoint(iteration):
            raise OSError("checkpoint failed")

        with (
            patch.object(controller, "_submit_iteration", side_effect=submit),
            self.assertLogs("openevolve.process_parallel", level="ERROR") as logs,
        ):
            asyncio.run(controller.run_evolution(2, 1, checkpoint_callback=fail_checkpoint))
        self.assertTrue(any("checkpoint failed" in message for message in logs.output))

    def test_initial_program_cannot_be_rejected(self):
        db = ProgramDatabase(self.config, PopulationStrategy(admit=lambda *args: False))
        with self.assertRaisesRegex(ValueError, "initial program"):
            db.add(program("seed", 0.5))
        self.assertFalse(db.programs)

    def test_only_strategy_errors_escape_evolution_loop(self):
        config = Config()
        config.database.num_islands = 2
        config.evaluator.parallel_evaluations = 1

        def admit(snapshot, candidate, island):
            if candidate.id == "child":
                raise RuntimeError("strategy failed") from OSError("original cause")
            return True

        db = ProgramDatabase(config.database, PopulationStrategy(admit=admit))
        db.add(program("seed", 0.5))
        controller = ProcessParallelController(config, "unused.py", db)
        controller.executor = object()

        def submit(iteration, island):
            future = Future()
            future.set_result(
                SerializableResult(
                    child_program_dict=program("child", 0.9).to_dict(),
                    artifacts={"output": "artifact"},
                )
            )
            return future

        with patch.object(controller, "_submit_iteration", side_effect=submit):
            with self.assertRaisesRegex(RuntimeError, "strategy failed") as failure:
                asyncio.run(controller.run_evolution(1, 1))
        self.assertIsInstance(failure.exception.__cause__, OSError)
        self.assertEqual(set(db.programs), {"seed"})

        db.population_strategy = PopulationStrategy(admit=lambda *args: True)
        controller = ProcessParallelController(config, "unused.py", db)
        controller.executor = object()
        with (
            patch.object(controller, "_submit_iteration", side_effect=submit),
            patch.object(db, "store_artifacts", side_effect=OSError("artifact failed")),
            self.assertLogs("openevolve.process_parallel", level="ERROR") as logs,
        ):
            asyncio.run(controller.run_evolution(1, 1))
        self.assertTrue(any("artifact failed" in message for message in logs.output))

    def test_cell_replacement_decision_controls_map(self):
        calls = []

        def replace(snapshot, candidate, incumbent, island):
            calls.append((candidate.id, incumbent.id, island))
            return True

        db = ProgramDatabase(self.config, PopulationStrategy(replace_cell=replace))
        with (
            patch.object(db, "_calculate_feature_coords", return_value=[0, 0]),
            self.assertLogs("openevolve.database", level="INFO") as logs,
        ):
            db.add(program("first", 0.8))
            db.add(program("second", 0.2))

        self.assertEqual(calls, [("second", "first", 0)])
        self.assertIn("second", db.island_feature_maps[0].values())
        self.assertNotIn("first", db.islands[0])
        self.assertTrue(any("cell replaced" in message for message in logs.output))

    def test_island_best_recomputed_when_replaced_program_remains_global_best(self):
        db = ProgramDatabase(self.config, PopulationStrategy(replace_cell=lambda *args: True))
        with patch.object(db, "_calculate_feature_coords", side_effect=[[0, 0], [1, 0], [0, 0]]):
            db.add(program("first", 0.9), target_island=0)
            db.add(program("other", 0.7), target_island=0)
            db.add(program("replacement", 0.1), target_island=0)

        self.assertEqual(db.best_program_id, "first")
        self.assertEqual(db.get_best_program().id, "first")
        self.assertIn("first", db.programs)
        self.assertNotIn("first", db.islands[0])
        self.assertEqual(db.island_best_programs[0], "other")

    def test_displaced_historical_best_does_not_evict_valid_cell(self):
        self.config.population_size = 2
        db = ProgramDatabase(self.config, PopulationStrategy(replace_cell=lambda *args: True))
        with patch.object(db, "_calculate_feature_coords", side_effect=[[0, 0], [1, 0], [0, 0]]):
            db.add(program("best", 0.9), target_island=0)
            db.add(program("other_cell", 0.7), target_island=0)
            db.add(program("replacement", 0.1), target_island=0)

        self.assertEqual(set(db.programs), {"best", "other_cell", "replacement"})
        self.assertEqual(set(db.island_feature_maps[0].values()), {"other_cell", "replacement"})
        self.assertEqual(db.get_best_program().id, "best")
        self.assertEqual(db.get_best_program("combined_score").id, "best")
        snapshot = db._population_snapshot()
        self.assertEqual(snapshot.best_program_id, "best")
        self.assertIn("best", snapshot.programs)

        with tempfile.TemporaryDirectory() as checkpoint:
            db.save(checkpoint)
            restored = ProgramDatabase(self.config)
            restored.load(checkpoint)
            self.assertEqual(set(restored.programs), set(db.programs))
            self.assertEqual(restored.get_best_program().id, "best")
            self.assertNotIn("best", restored.islands[0])

    def test_old_historical_best_loses_protection_when_new_best_appears(self):
        self.config.population_size = 2
        db = ProgramDatabase(self.config, PopulationStrategy(replace_cell=lambda *args: True))
        with patch.object(
            db, "_calculate_feature_coords", side_effect=[[0, 0], [1, 0], [0, 0], [2, 0]]
        ):
            db.add(program("best", 0.9), target_island=0)
            db.add(program("other_cell", 0.7), target_island=0)
            db.add(program("replacement", 0.1), target_island=0)
            self.assertIn("best", db.programs)
            db.add(program("new_best", 1.0), target_island=0)

        self.assertEqual(db.best_program_id, "new_best")
        self.assertNotIn("best", db.programs)
        self.assertEqual(len(db.programs), self.config.population_size)

    def test_metric_query_does_not_change_historical_best(self):
        self.config.population_size = 2
        db = ProgramDatabase(self.config, PopulationStrategy(replace_cell=lambda *args: True))
        best = program("best", 0.9)
        best.metrics["aux"] = 0
        other = program("other", 0.7)
        other.metrics["aux"] = 10
        with patch.object(
            db, "_calculate_feature_coords", side_effect=[[0, 0], [1, 0], [0, 0], [2, 0]]
        ):
            db.add(best, target_island=0)
            db.add(other, target_island=0)
            db.add(program("replacement", 0.1), target_island=0)
            self.assertEqual(db.get_best_program("aux").id, "other")
            self.assertEqual(db.best_program_id, "best")
            db.add(program("new_cell", 0.2), target_island=0)

        self.assertIn("best", db.programs)
        self.assertEqual(db.get_best_program().id, "best")

    def test_archived_historical_best_does_not_displace_valid_cell(self):
        self.config.population_size = 2
        strategy = PopulationStrategy(
            replace_cell=lambda *args: True,
            archive=lambda snapshot, candidate: ArchiveDecision(add=candidate.id == "best"),
        )
        db = ProgramDatabase(self.config, strategy)
        with patch.object(db, "_calculate_feature_coords", side_effect=[[0, 0], [1, 0], [0, 0]]):
            db.add(program("best", 0.9), target_island=0)
            db.add(program("other_cell", 0.7), target_island=0)
            db.add(program("replacement", 0.1), target_island=0)

        self.assertEqual(set(db.programs), {"best", "other_cell", "replacement"})
        self.assertEqual(set(db.island_feature_maps[0].values()), {"other_cell", "replacement"})
        self.assertEqual(db.get_best_program().id, "best")
        self.assertIn("best", db.archive)

    def test_invalid_replacement_rolls_back_provisional_writes(self):
        db = ProgramDatabase(
            self.config, PopulationStrategy(replace_cell=lambda *args: "not a bool")
        )
        with patch.object(db, "_calculate_feature_coords", return_value=[0, 0]):
            db.add(program("first", 0.8))
            before = db._population_snapshot()
            candidate = program("second", 0.9)
            with self.assertRaisesRegex(ValueError, "replace_cell must return bool"):
                db.add(candidate, iteration=7)

        self.assertEqual(db._population_snapshot(), before)
        self.assertNotIn("second", db.programs)
        self.assertEqual(candidate.iteration_found, 0)

    def test_archive_hook_sees_displaced_member_before_deciding(self):
        seen = []

        def archive(snapshot, candidate):
            seen.append(snapshot.archive)
            return ArchiveDecision(add=candidate.id == "first")

        db = ProgramDatabase(
            self.config,
            PopulationStrategy(replace_cell=lambda *args: True, archive=archive),
        )
        with patch.object(db, "_calculate_feature_coords", return_value=[0, 0]):
            db.add(program("first", 0.8))
            db.add(program("second", 0.9))

        self.assertEqual(seen[-1], frozenset({"first"}))
        self.assertEqual(db.archive, {"first"})

    def test_invalid_archive_decision_rolls_back_population(self):
        db = ProgramDatabase(
            self.config,
            PopulationStrategy(
                archive=lambda snapshot, candidate: (
                    ArchiveDecision(add=True) if candidate.id == "first" else "invalid"
                )
            ),
        )
        with patch.object(db, "_calculate_feature_coords", side_effect=[[0, 0], [1, 0]]):
            db.add(program("first", 0.8))
            before = db._population_snapshot()
            with self.assertRaisesRegex(ValueError, "archive must return ArchiveDecision"):
                db.add(program("second", 0.9), iteration=7)

        self.assertEqual(db._population_snapshot(), before)

    def test_invalid_eviction_decision_rolls_back_population(self):
        self.config.population_size = 2
        db = ProgramDatabase(
            self.config,
            PopulationStrategy(evict=lambda snapshot, required, protected: ("first",)),
        )
        with patch.object(db, "_calculate_feature_coords", side_effect=[[0, 0], [1, 0], [2, 0]]):
            db.add(program("first", 0.9), target_island=0)
            db.add(program("second", 0.8), target_island=1)
            before = db._population_snapshot()
            with self.assertRaisesRegex(ValueError, "distinct eligible"):
                db.add(program("third", 0.7), iteration=7, target_island=0)

        self.assertEqual(db._population_snapshot(), before)

    def test_archive_decision_and_eviction_keep_database_consistent(self):
        self.config.archive_size = 1
        self.config.population_size = 2

        def archive(snapshot, candidate):
            old = next(iter(snapshot.archive), None)
            return ArchiveDecision(add=True, evict_id=old)

        def evict(snapshot, required, protected):
            self.assertEqual(required, 1)
            self.assertNotIn("second", protected)
            return ("second",)

        db = ProgramDatabase(self.config, PopulationStrategy(archive=archive, evict=evict))
        db.add(program("first", 0.9), target_island=0)
        db.add(program("second", 0.5), target_island=1)
        self.assertEqual(db.archive, {"second"})
        db.add(program("third", 0.4), target_island=0)

        self.assertNotIn("second", db.programs)
        self.assertNotIn("second", db.islands[1])
        self.assertNotIn("second", db.island_feature_maps[1].values())
        self.assertEqual(db.archive, {"third"})

    def test_archive_policy_can_replace_member_before_capacity(self):
        self.config.archive_size = 2

        def archive(snapshot, candidate):
            return ArchiveDecision(add=True, evict_id=next(iter(snapshot.archive), None))

        db = ProgramDatabase(self.config, PopulationStrategy(archive=archive))
        db.add(program("first", 0.5))
        db.add(program("second", 0.6))
        self.assertEqual(db.archive, {"second"})

    def test_migration_plan_is_applied_by_database(self):
        strategy = PopulationStrategy(
            migration_due=lambda snapshot: True,
            migrate=lambda snapshot: (MigrationMove("seed", 1),),
        )
        db = ProgramDatabase(self.config, strategy)
        db.add(program("seed", 0.8), target_island=0)
        self.assertTrue(db.should_migrate())

        db.migrate_programs()

        migrants = [p for p in db.programs.values() if p.metadata.get("migrant")]
        self.assertEqual(len(migrants), 1)
        self.assertEqual(migrants[0].parent_id, "seed")
        self.assertIn(migrants[0].id, db.islands[1])
        self.assertIn("seed", db.islands[0])

    def test_migration_accepts_integral_targets_but_not_bool(self):
        class Island(IntEnum):
            SECOND = 1

        for target in (np.int64(1), Island.SECOND):
            with self.subTest(target=target):
                db = ProgramDatabase(
                    self.config,
                    PopulationStrategy(migrate=lambda snapshot: (MigrationMove("seed", target),)),
                )
                db.add(program("seed", 0.8), target_island=0)
                db.migrate_programs()
                migrant = next(p for p in db.programs.values() if p.metadata.get("migrant"))
                self.assertIs(type(migrant.metadata["island"]), int)
                self.assertEqual(migrant.metadata["island"], 1)

        db = ProgramDatabase(
            self.config,
            PopulationStrategy(migrate=lambda snapshot: (MigrationMove("seed", True),)),
        )
        db.add(program("seed", 0.8), target_island=0)
        with self.assertRaisesRegex(ValueError, "invalid migration target island"):
            db.migrate_programs()

    def test_invalid_eviction_cannot_remove_protected_best(self):
        self.config.population_size = 1
        db = ProgramDatabase(
            self.config, PopulationStrategy(evict=lambda snapshot, required, protected: ("best",))
        )
        db.programs["best"] = program("best", 0.9)
        db.programs["other"] = program("other", 0.1)
        db.best_program_id = "best"
        db.islands[0].add("best")
        db.island_feature_maps[0]["0-0"] = "best"

        with self.assertRaisesRegex(ValueError, "distinct eligible"):
            db._enforce_population_limit()
        self.assertEqual(set(db.programs), {"best", "other"})

    def test_public_api_forwards_strategy(self):
        async def fake_run(*args, **kwargs):
            return "forwarded"

        strategy = PopulationStrategy(admit=lambda snapshot, candidate, island: True)
        with patch("openevolve.api._run_evolution_async", side_effect=fake_run) as mock_run:
            result = run_evolution("program", "evaluator", population_strategy=strategy)

        self.assertEqual(result, "forwarded")
        self.assertIs(mock_run.call_args.kwargs["population_strategy"], strategy)


if __name__ == "__main__":
    unittest.main()
