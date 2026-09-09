"""Backend contract tests; future database implementations can reuse this mixin."""

import tempfile
import unittest
from pathlib import Path

from openevolve.config import DatabaseConfig
from openevolve.database import Program, ProgramDatabase
from openevolve.database_memory import InMemoryProgramDatabase


class ProgramDatabaseContract:
    """Use only the public interface when checking backend behavior."""

    def make_database(self, config: DatabaseConfig) -> ProgramDatabase:
        raise NotImplementedError

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.config = DatabaseConfig(
            num_islands=2,
            feature_dimensions=["slot"],
            exploration_ratio=1.0,
            exploitation_ratio=0.0,
            artifacts_base_path=self.temp_dir.name,
        )
        self.db = self.make_database(self.config)

    def seed(self):
        for island, score in enumerate([0.4, 0.8]):
            self.db.add(
                Program(
                    id=f"p{island}",
                    code=f"return {island}",
                    metrics={"combined_score": score, "slot": 0.5, "other": 1 - score},
                ),
                target_island=island,
            )

    def test_empty_database(self):
        self.assertIsInstance(self.db, ProgramDatabase)
        self.assertEqual(self.db.get_state().program_count, 0)
        self.assertIsNone(self.db.get("missing"))
        self.assertIsNone(self.db.get_best_program())
        self.assertEqual(self.db.get_top_programs(3), [])
        self.assertEqual(self.db.get_artifacts("missing"), {})
        self.assertEqual(self.db.get_prompt_history("missing"), {})
        with self.assertRaises(ValueError):
            self.db.sample_from_island(0)

    def test_writes_and_reads_are_detached(self):
        program = Program(
            id="value",
            code="pass",
            metrics={"combined_score": 1.0, "slot": 0.5},
            metadata={"nested": ["original"]},
        )
        self.db.add(program, iteration=7, target_island=1)
        program.code = "caller changed this"
        program.metrics["combined_score"] = -100
        program.metadata["nested"].append("changed")
        stored = self.db.get("value")
        self.assertEqual(stored.code, "pass")
        self.assertEqual(stored.metadata, {"nested": ["original"], "island": 1})
        self.assertEqual(stored.iteration_found, 7)
        stored.metrics["combined_score"] = -100
        self.assertEqual(self.db.get("value").metrics["combined_score"], 1.0)

    def test_bounded_ranked_queries_and_detached_results(self):
        self.seed()
        self.assertEqual([p.id for p in self.db.get_top_programs(1)], ["p1"])
        self.assertEqual(self.db.get_top_programs(0), [])
        self.assertEqual([p.id for p in self.db.get_top_programs(4, island_idx=0)], ["p0"])
        with self.assertRaises(ValueError):
            self.db.get_top_programs(-1)
        with self.assertRaises(IndexError):
            self.db.get_top_programs(1, island_idx=2)
        best = self.db.get_best_program()
        top = self.db.get_top_programs(1)[0]
        best.metrics["combined_score"] = -100
        top.code = "changed"
        self.assertEqual(self.db.get("p1").metrics["combined_score"], 0.8)
        self.assertEqual(self.db.get("p1").code, "return 1")

    def test_custom_metric_query_does_not_change_default_best(self):
        self.seed()
        self.assertEqual(self.db.get_best_program(metric="other").id, "p0")
        self.assertEqual(self.db.get_best_program().id, "p1")
        self.assertEqual(self.db.get_top_programs(10, metric="missing"), [])

    def test_sampling_is_bounded_and_does_not_expose_stored_records(self):
        self.seed()
        for limit in [0, 1, 5]:
            parent, inspirations = self.db.sample_from_island(1, num_inspirations=limit)
            self.assertEqual(parent.id, "p1")
            self.assertLessEqual(len(inspirations), limit)
            parent.code = "changed"
            for program in inspirations:
                program.code = "changed"
        self.assertEqual(self.db.get("p1").code, "return 1")
        self.assertEqual(self.db.get_state().current_island, 0)

    def test_explicit_target_island_overrides_parent(self):
        self.seed()
        self.db.add(
            Program(
                id="child",
                code="return 3",
                parent_id="p0",
                metrics={"combined_score": 0.9, "slot": 0.5},
            ),
            iteration=2,
            target_island=1,
        )
        self.assertEqual(self.db.get("child").metadata["island"], 1)
        self.assertEqual(self.db.get_top_programs(1, island_idx=1)[0].id, "child")

    def test_progress_is_monotonic_even_without_new_programs(self):
        self.seed()
        for iteration in [4, 2, 7, 7]:
            self.db.record_iteration(iteration)
        state = self.db.get_state()
        self.assertEqual(state.last_iteration, 7)
        self.assertEqual(state.program_count, 2)
        self.assertEqual(state.num_islands, 2)
        self.assertEqual(state.feature_dimensions, ("slot",))
        with self.assertRaises(ValueError):
            self.db.record_iteration(-1)

    def test_artifacts_and_prompt_history_are_explicit_writes(self):
        self.seed()
        artifacts = {"stderr": "evidence", "large": "x" * (40 * 1024)}
        self.db.store_artifacts("p0", artifacts)
        self.assertEqual(self.db.get_artifacts("p0"), artifacts)
        retrieved = self.db.get_artifacts("p0")
        retrieved["stderr"] = "changed"
        self.assertEqual(self.db.get_artifacts("p0")["stderr"], "evidence")
        prompt = {"system": "system", "user": "user"}
        responses = ["answer"]
        self.db.log_prompt("p0", "rewrite", prompt, responses)
        prompt["user"] = "changed"
        responses.append("changed")
        history = self.db.get_prompt_history("p0")
        self.assertEqual(
            history["rewrite"], {"system": "system", "user": "user", "responses": ["answer"]}
        )
        history["rewrite"]["responses"].append("changed")
        self.assertEqual(self.db.get_prompt_history("p0")["rewrite"]["responses"], ["answer"])

    def test_generation_and_migration_queries(self):
        self.seed()
        self.assertFalse(self.db.should_migrate())
        self.db.increment_island_generation(1)
        stats = self.db.get_island_stats()
        self.assertEqual(stats[1]["generation"], 1)
        stats[1]["generation"] = 100
        self.assertEqual(self.db.get_island_stats()[1]["generation"], 1)


class TestInMemoryDatabase(ProgramDatabaseContract, unittest.TestCase):
    def make_database(self, config):
        return InMemoryProgramDatabase(config)

    def test_no_population_files_or_restore(self):
        config = DatabaseConfig(db_path=self.temp_dir.name, in_memory=False)
        db = InMemoryProgramDatabase(config)
        db.add(Program(id="p", code="pass"))
        self.assertEqual(list(Path(self.temp_dir.name).iterdir()), [])
        # A new instance is a new empty session, even with the same legacy path.
        self.assertEqual(InMemoryProgramDatabase(config).get_state().program_count, 0)
        self.assertFalse(hasattr(db, "save"))
        self.assertFalse(hasattr(db, "load"))
