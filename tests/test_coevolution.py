"""Competitive phase boundaries exercised with the real database and evaluator."""

import copy
import json
import logging
import runpy
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from openevolve import Opponent, Population, run_coevolution
from openevolve.config import Config, LLMModelConfig
from openevolve.database import Program


def configuration():
    config = Config()
    config.language = "python"
    config.llm.models = [LLMModelConfig(name="local-test", api_key="test")]
    config.llm.evaluator_models = []
    config.database.num_islands = 2
    config.database.population_size = 20
    config.database.archive_size = 10
    config.evaluator.cascade_evaluation = False
    config.evaluator.max_retries = 0
    return config


class TestCoevolution(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.specs = {}
        for name in ("blue", "red"):
            source = self.root / f"{name}.py"
            source.write_text("0")
            self.specs[name] = Population(source, configuration())
        self.output = self.root / "output"

    def run_competition(self, **kwargs):
        options = dict(
            output_dir=str(self.output),
            evaluation_id="test-game-v1",
            rounds=2,
            iterations_per_phase=1,
            opponent_count=2,
        )
        options.update(kwargs)

        def evaluate(path, name, opponents):
            code = Path(path).read_text()
            target = int(opponents[-1].code) + 1
            return {"combined_score": 1.0 / (1 + abs(int(code) - target))}

        return run_coevolution(self.specs, evaluate, **options)

    @staticmethod
    async def evolve(engine, iterations):
        # Only generation is replaced; all re-evaluation, archive insertion and
        # coordinator persistence use production code.
        next_value = max(int(p.code) for p in engine.database.programs.values()) + 1
        program = Program(id=f"candidate-{next_value}", code=str(next_value))
        program.metrics = await engine.evaluator.evaluate_program(program.code, program.id)
        engine.database.add(program, iteration=engine.database.last_iteration, target_island=1)
        return engine.database.get_best_program()

    def test_refreshes_all_retained_scores_and_selection_state(self):
        observed = []

        async def inspect(engine, iterations):
            previous = engine.database.get("candidate-1")
            if previous:
                observed.append(previous.metrics["combined_score"])
            return await TestCoevolution.evolve(engine, iterations)

        with patch("openevolve.coevolution.OpenEvolve.run", inspect):
            result = self.run_competition()
        state = json.loads(Path(result.checkpoint_path).read_text())
        self.assertEqual(result.completed_phases, 4)
        self.assertEqual([p["population"] for p in state["history"]], ["blue", "red"] * 2)
        # Blue's old champion no longer has its old perfect score after red moves.
        blue = state["populations"]["blue"]
        self.assertIn(0.5, observed)
        self.assertEqual(result.champions["blue"].code, "2")
        self.assertEqual(result.champions["blue"].metrics["combined_score"], 1.0)
        for name, population in state["populations"].items():
            self.assertTrue(all(p["metadata"]["island"] in (0, 1) for p in population["programs"]))
            context = next(
                h["context"] for h in reversed(state["history"]) if h["population"] == name
            )
            self.assertEqual(
                {p["metadata"]["coevolution_context"] for p in population["programs"]}, {context}
            )

    def test_populations_do_not_share_candidates_or_configs(self):
        before = {name: copy.deepcopy(spec.config.to_dict()) for name, spec in self.specs.items()}

        async def evolve(engine, iterations):
            team = "blue" if "blue-seed" in engine.database.programs else "red"
            self.assertTrue(
                all(not p.id.startswith("other-") for p in engine.database.programs.values())
            )
            program = Program(id=f"{team}-only", code="1", metrics={"combined_score": 1.0})
            engine.database.add(program)
            return program

        with patch("openevolve.coevolution.OpenEvolve.run", evolve):
            result = self.run_competition(rounds=1)
        state = json.loads(Path(result.checkpoint_path).read_text())
        self.assertNotIn("red-only", [p["id"] for p in state["populations"]["blue"]["programs"]])
        self.assertNotIn("blue-only", [p["id"] for p in state["populations"]["red"]["programs"]])
        for name, spec in self.specs.items():
            self.assertEqual(spec.config.to_dict(), before[name])

    def test_resume_skips_completed_phases(self):
        with patch("openevolve.coevolution.OpenEvolve.run", self.evolve):
            first = self.run_competition(rounds=1)
            checkpoint = Path(first.checkpoint_path)
            history = json.loads(checkpoint.read_text())["history"]
            resumed = self.run_competition(resume=True)
        final = json.loads(checkpoint.read_text())
        self.assertEqual(final["history"][:2], history)
        self.assertEqual(resumed.completed_phases, 4)
        with patch(
            "openevolve.coevolution.OpenEvolve.run", side_effect=AssertionError("must not run")
        ):
            self.assertEqual(self.run_competition(resume=True).completed_phases, 4)

    def test_failed_phase_retains_checkpoint_and_frozen_opponents(self):
        with patch("openevolve.coevolution.OpenEvolve.run", self.evolve):
            self.run_competition(rounds=1)
        checkpoint = self.output / "coevolution.json"
        before = checkpoint.read_bytes()

        async def fail(engine, iterations):
            engine.database.add(Program(id="partial", code="99", metrics={"combined_score": 99.0}))
            raise RuntimeError("worker failure")

        with (
            patch("openevolve.coevolution.OpenEvolve.run", fail),
            self.assertRaisesRegex(RuntimeError, "worker failure"),
        ):
            self.run_competition(resume=True)
        self.assertEqual(checkpoint.read_bytes(), before)
        with patch("openevolve.coevolution.OpenEvolve.run", self.evolve):
            result = self.run_competition(resume=True)
        self.assertNotIn(
            "99",
            [
                p["code"]
                for p in json.loads(checkpoint.read_text())["populations"]["blue"]["programs"]
            ],
        )
        self.assertEqual(result.completed_phases, 4)

    def test_resume_rejects_changed_experiment(self):
        with patch("openevolve.coevolution.OpenEvolve.run", self.evolve):
            self.run_competition(rounds=1)
        for options in (
            {"evaluation_id": "v2"},
            {"opponent_count": 3},
            {"iterations_per_phase": 2},
        ):
            with (
                self.subTest(options=options),
                self.assertRaisesRegex(ValueError, "context differs"),
            ):
                self.run_competition(resume=True, **options)
        self.specs["blue"].initial_program.write_text("changed seed")
        with self.assertRaisesRegex(ValueError, "context differs"):
            self.run_competition(resume=True)

    def test_credential_rotation_allowed_and_credentials_not_saved(self):
        with patch("openevolve.coevolution.OpenEvolve.run", self.evolve):
            result = self.run_competition(rounds=1)
            self.specs["blue"].config.llm.models[0].api_key = "rotated-secret"
            self.run_competition(rounds=1, resume=True)
        self.assertNotIn("rotated-secret", Path(result.checkpoint_path).read_text())

    def test_failed_refresh_does_not_commit_or_run_generation(self):
        with patch("openevolve.coevolution.OpenEvolve.run", self.evolve):
            self.run_competition(rounds=1)
        before = (self.output / "coevolution.json").read_bytes()
        with (
            patch("openevolve.evaluator.Evaluator.evaluate_program", return_value={"error": 0.0}),
            patch("openevolve.coevolution.OpenEvolve.run") as generate,
        ):
            with self.assertRaisesRegex(ValueError, "Re-evaluation failed"):
                self.run_competition(resume=True)
            generate.assert_not_called()
        self.assertEqual((self.output / "coevolution.json").read_bytes(), before)

    def test_drops_stale_artifacts_and_prompts(self):
        with patch("openevolve.coevolution.OpenEvolve.run", self.evolve):
            self.run_competition(rounds=1)
        checkpoint = self.output / "coevolution.json"
        state = json.loads(checkpoint.read_text())
        for p in state["populations"]["blue"]["programs"]:
            p.update(
                artifacts_json='{"old_score": 999}', artifact_dir="/old", prompts={"old": "score"}
            )
        checkpoint.write_text(json.dumps(state))

        async def inspect(engine, iterations):
            for p in engine.database.programs.values():
                self.assertIsNone(p.artifacts_json)
                self.assertIsNone(p.artifact_dir)
                self.assertIsNone(p.prompts)
            return await TestCoevolution.evolve(engine, iterations)

        with patch("openevolve.coevolution.OpenEvolve.run", inspect):
            self.run_competition(resume=True)

    def test_no_overwrite_and_missing_resume(self):
        with self.assertRaises(FileNotFoundError):
            self.run_competition(resume=True)
        with patch("openevolve.coevolution.OpenEvolve.run", self.evolve):
            self.run_competition(rounds=1)
        with self.assertRaises(FileExistsError):
            self.run_competition()

    def test_champion_history_is_bounded_and_deduplicated(self):
        with patch("openevolve.coevolution.OpenEvolve.run", self.evolve):
            result = self.run_competition(rounds=3)
        state = json.loads(Path(result.checkpoint_path).read_text())
        for history in state["opponents"].values():
            self.assertLessEqual(len(history), 2)
            self.assertEqual(len(history), len({p["code"] for p in history}))

    def test_invalid_boundary_inputs(self):
        for options in (
            {"rounds": 0},
            {"iterations_per_phase": 0},
            {"opponent_count": 0},
            {"evaluation_id": ""},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_competition(**options)
        self.specs["blue"].config.database.db_path = "shared"
        with self.assertRaisesRegex(ValueError, "db_path"):
            self.run_competition()

    def test_island_schedule_survives_phase_refresh(self):
        seen = []

        async def evolve(engine, iterations):
            seen.append(list(engine.database.island_generations))
            engine.database.island_generations[1] += 3
            engine.database.current_island = 1
            engine.database.last_migration_generation = 2
            return engine.database.get_best_program()

        with patch("openevolve.coevolution.OpenEvolve.run", evolve):
            result = self.run_competition()
        self.assertEqual(seen, [[0, 0], [0, 0], [0, 3], [0, 3]])
        for value in json.loads(Path(result.checkpoint_path).read_text())["populations"].values():
            self.assertEqual(value["schedule"]["island_generations"], [0, 6])
            self.assertEqual(value["schedule"]["current_island"], 1)

    def test_logging_handlers_are_not_accumulated(self):
        before = list(logging.getLogger().handlers)
        with patch("openevolve.coevolution.OpenEvolve.run", self.evolve):
            self.run_competition()
        self.assertEqual(logging.getLogger().handlers, before)

    def test_real_process_workers_and_resume(self):
        example = runpy.run_path(
            str(Path(__file__).parents[1] / "examples/competitive_resource_allocation/compare.py")
        )
        server = example["ProposalServer"](("127.0.0.1", 0), example["Handler"])
        server.mixed_attacker = False
        server.reset(0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            initial = self.root / "allocation.py"
            initial.write_text(example["policy"]((1.0, 0.0, 0.0)))
            populations = {
                name: Population(initial, example["config"](server.server_port, name, 0))
                for name in ("defender", "attacker")
            }
            evaluate = example["evaluator"](self.root / "matches.jsonl")
            options = dict(
                output_dir=str(self.output), evaluation_id="integration-v1", iterations_per_phase=3
            )
            first = run_coevolution(populations, evaluate, rounds=1, **options)
            first_state = json.loads(Path(first.checkpoint_path).read_text())
            final = run_coevolution(populations, evaluate, rounds=2, resume=True, **options)
            state = json.loads(Path(final.checkpoint_path).read_text())
            self.assertEqual(sum(server.counts.values()), 12)
            self.assertEqual(state["history"][:2], first_state["history"])
            self.assertEqual(final.completed_phases, 4)
            for entry in state["history"]:
                context = json.loads((Path(entry["output_dir"]) / "opponents.json").read_text())
                self.assertEqual(entry["opponent_ids"], [p["id"] for p in context["opponents"]])
            # Persisted fitness agrees with independently recomputing the final
            # programs against that population's actual frozen opponent cohort.
            for name, population in state["populations"].items():
                last = next(h for h in reversed(state["history"]) if h["population"] == name)
                context = json.loads((Path(last["output_dir"]) / "opponents.json").read_text())
                cohort = tuple(Opponent(**item) for item in context["opponents"])
                for program in population["programs"]:
                    source = self.root / "check.py"
                    source.write_text(program["code"])
                    self.assertEqual(program["metrics"], evaluate(str(source), name, cohort))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
