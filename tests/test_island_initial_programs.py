"""Tests for seeding islands from per-island initial programs (issue #55)."""

import asyncio
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import yaml

# Dummy API key so LLM ensembles initialize without a real key
os.environ.setdefault("OPENAI_API_KEY", "test")

from openevolve.config import Config
from openevolve.controller import OpenEvolve
from openevolve.population import PopulationStrategy


class MockEvaluator:
    """Mock evaluator that records the code it was asked to evaluate"""

    def __init__(self):
        self.evaluated = []

    async def evaluate_program(self, code, program_id):
        self.evaluated.append(code)
        return {"combined_score": 0.1 * len(self.evaluated)}

    def get_pending_artifacts(self, program_id):
        return None


class TestIslandInitialProgramsConfig(unittest.TestCase):
    def test_defaults_to_empty(self):
        self.assertEqual(Config().database.island_initial_programs, [])
        self.assertEqual(Config.from_dict({}).database.island_initial_programs, [])

    def test_accepts_null_entries(self):
        config = Config.from_dict(
            {"database": {"num_islands": 3, "island_initial_programs": [None, "b.py", None]}}
        )
        self.assertEqual(config.database.island_initial_programs, [None, "b.py", None])

    def test_rejects_more_entries_than_islands(self):
        with self.assertRaisesRegex(ValueError, "island_initial_programs.*num_islands"):
            Config.from_dict(
                {"database": {"num_islands": 2, "island_initial_programs": [None, "b.py", "c.py"]}}
            )

    def test_rejects_non_null_entry_zero(self):
        for entry in ("a.py", ""):
            with self.subTest(entry=entry):
                with self.assertRaisesRegex(ValueError, r"island_initial_programs\[0\]"):
                    Config.from_dict(
                        {"database": {"num_islands": 2, "island_initial_programs": [entry]}}
                    )

    def test_relative_paths_resolve_against_config_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config_subdir"
            config_dir.mkdir()
            config_file = config_dir / "config.yaml"
            absolute_seed = str(Path(tmpdir) / "absolute.py")
            with open(config_file, "w") as f:
                yaml.dump(
                    {
                        "database": {
                            "num_islands": 4,
                            "island_initial_programs": [
                                None,
                                "seeds/one.py",
                                absolute_seed,
                                None,
                            ],
                        }
                    },
                    f,
                )

            config = Config.from_yaml(config_file)

            self.assertEqual(
                config.database.island_initial_programs,
                [None, str((config_dir / "seeds/one.py").resolve()), absolute_seed, None],
            )


class TestIslandInitialProgramsController(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.initial_program_path = self._write("initial_program.py", "def solve(): return 0\n")
        self.seed_one_path = self._write("seed_one.py", "def solve(): return 1\n")
        self.seed_three_path = self._write("seed_three.py", "def solve(): return 3\n")
        self.evaluator_path = self._write(
            "evaluator.py", "def evaluate(program_path):\n    return {'combined_score': 0.5}\n"
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _write(self, name, content):
        path = os.path.join(self.test_dir, name)
        with open(path, "w") as f:
            f.write(content)
        return path

    def _make_config(self, island_initial_programs=None, num_islands=4):
        config = Config()
        config.database.in_memory = True
        config.database.num_islands = num_islands
        if island_initial_programs is not None:
            config.database.island_initial_programs = island_initial_programs
        return config

    def _make_controller(self, config, **kwargs):
        return OpenEvolve(
            initial_program_path=self.initial_program_path,
            evaluation_file=self.evaluator_path,
            config=config,
            output_dir=os.path.join(self.test_dir, "out"),
            **kwargs,
        )

    def _run(self, config, checkpoint_path=None, **kwargs):
        """Run the controller with mocked evaluation and no evolution iterations"""

        async def run():
            with (
                patch("openevolve.controller.Evaluator", return_value=MockEvaluator()),
                patch("openevolve.controller.ProcessParallelController") as parallel_class,
            ):
                parallel = Mock()
                parallel.run_evolution = AsyncMock(return_value=None)
                parallel.shutdown_event.is_set.return_value = False
                parallel.produced_no_candidates = False
                parallel_class.return_value = parallel

                controller = self._make_controller(config, **kwargs)
                await controller.run(iterations=0, checkpoint_path=checkpoint_path)
                return controller

        return asyncio.run(run())

    def _codes_by_island(self, database):
        return [
            sorted(database.programs[pid].code for pid in island) for island in database.islands
        ]

    def _read(self, path):
        with open(path) as f:
            return f.read()

    def test_seeds_land_on_their_islands(self):
        config = self._make_config([None, self.seed_one_path, None, self.seed_three_path])
        controller = self._run(config)
        database = controller.database

        self.assertEqual(
            self._codes_by_island(database),
            [
                [self._read(self.initial_program_path)],
                [self._read(self.seed_one_path)],
                [],
                [self._read(self.seed_three_path)],
            ],
        )
        for island_idx, island in enumerate(database.islands):
            for program_id in island:
                self.assertEqual(database.programs[program_id].metadata["island"], island_idx)
        self.assertEqual(len(controller.evaluator.evaluated), 3)

    def test_null_island_is_filled_lazily_as_before(self):
        config = self._make_config([None, self.seed_one_path])
        database = self._run(config).database
        self.assertEqual(database.islands[2], set())

        database.current_island = 2
        parent = database._sample_exploration_parent()

        self.assertEqual(parent.code, database.get_best_program().code)
        self.assertEqual(parent.metadata["island"], 2)

    def test_missing_seed_file_fails_at_construction(self):
        missing = os.path.join(self.test_dir, "missing.py")
        config = self._make_config([None, missing])
        with self.assertRaisesRegex(FileNotFoundError, r"island_initial_programs\[1\].*missing"):
            self._make_controller(config)

    def test_config_built_in_code_is_validated_at_construction(self):
        # Config objects built in code (e.g. for run_evolution) skip Config.from_dict
        cases = {
            "more entries": ([None, self.seed_one_path, self.seed_three_path], "num_islands"),
            "entry zero": ([self.seed_one_path], r"island_initial_programs\[0\]"),
        }
        for name, (seeds, message) in cases.items():
            with self.subTest(name):
                config = self._make_config(seeds, num_islands=2)
                with self.assertRaisesRegex(ValueError, message):
                    self._make_controller(config)

    def test_different_extension_warns(self):
        seed_path = self._write("seed.rs", "fn solve() {}\n")
        config = self._make_config([None, seed_path])
        with self.assertLogs("openevolve.controller", level="WARNING") as logs:
            controller = self._make_controller(config)
        self.assertTrue(any("different extension" in line for line in logs.output))
        self.assertEqual(controller.island_initial_program_codes, {1: "fn solve() {}\n"})

    def test_rejected_seed_warns_and_leaves_island_empty(self):
        strategy = PopulationStrategy(admit=lambda snapshot, candidate, island: island == 0)
        config = self._make_config([None, self.seed_one_path])
        with self.assertLogs("openevolve.controller", level="WARNING") as logs:
            database = self._run(config, population_strategy=strategy).database
        self.assertTrue(any("rejected the initial program" in line for line in logs.output))
        self.assertEqual(len(database.programs), 1)
        self.assertEqual(database.islands[1], set())

    def test_seed_artifacts_are_stored(self):
        self.evaluator_path = self._write(
            "evaluator.py",
            "from openevolve.evaluation_result import EvaluationResult\n"
            "\n"
            "def evaluate(program_path):\n"
            "    with open(program_path) as f:\n"
            "        source = f.read()\n"
            "    return EvaluationResult(\n"
            "        metrics={'combined_score': 0.5}, artifacts={'source': source}\n"
            "    )\n",
        )
        config = self._make_config([None, self.seed_one_path])
        controller = self._make_controller(config)
        with patch.object(
            controller, "_run_evolution_with_checkpoints", new=AsyncMock(return_value=None)
        ):
            asyncio.run(controller.run(iterations=0))

        database = controller.database
        (seed_id,) = database.islands[1]
        self.assertEqual(database.get_artifacts(seed_id)["source"], self._read(self.seed_one_path))

    def test_resume_does_not_duplicate_seeds(self):
        config = self._make_config([None, self.seed_one_path, None, self.seed_three_path])
        controller = self._run(config)
        controller._save_checkpoint(0)
        checkpoint_path = os.path.join(controller.output_dir, "checkpoints", "checkpoint_0")
        saved_codes = self._codes_by_island(controller.database)

        resumed_config = self._make_config([None, self.seed_one_path, None, self.seed_three_path])
        resumed = self._run(resumed_config, checkpoint_path=checkpoint_path)

        self.assertEqual(self._codes_by_island(resumed.database), saved_codes)
        self.assertEqual(len(resumed.database.programs), 3)
        self.assertEqual(resumed.evaluator.evaluated, [])

    def test_config_without_field_matches_previous_behaviour(self):
        for island_initial_programs in (None, [], [None, None]):
            with self.subTest(island_initial_programs=island_initial_programs):
                controller = self._run(self._make_config(island_initial_programs))
                database = controller.database

                self.assertEqual(
                    self._codes_by_island(database),
                    [[self._read(self.initial_program_path)], [], [], []],
                )
                (program,) = database.programs.values()
                self.assertEqual(program.metadata["island"], 0)
                self.assertEqual(database.best_program_id, program.id)
                self.assertEqual(controller.evaluator.evaluated, [program.code])


if __name__ == "__main__":
    unittest.main()
