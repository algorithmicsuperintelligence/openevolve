"""Exercise the example CLI with local fixtures and an offline evolution worker."""

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openevolve.compat.alphaevolve_env import config_from_alphaevolve_env

RUNNER = Path(__file__).resolve().parents[1] / "examples/alphaevolve_interop/run.py"
spec = importlib.util.spec_from_file_location("_alphaevolve_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class OfflineModel:
    def __init__(self, config):
        self.model = config.name

    async def generate_with_context(self, system_message, messages, **kwargs):
        assert "Pack circles" in system_message
        return "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE"


def offline_model(config):
    return OfflineModel(config)


class TestAlphaEvolveRunner(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for name, function, metric in (
            ("circle_packing", "circle_packing_evaluation", "sum_of_radii"),
            ("tsp", "tsp_evaluation_function", "neg_tour_length"),
            ("signal_processing", "signal_processing_evaluation", "overall_score"),
        ):
            source = self.root / name / "src"
            source.mkdir(parents=True)
            (source / "program.py").write_text("# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END\n")
            (source / "evaluate.py").write_text(
                f"def {function}(candidate):\n"
                "    namespace = {}\n"
                "    exec(candidate['content']['files'][0]['content'], namespace)\n"
                f"    return {{'scores': {{'scores': [{{'metric': '{metric}', 'score': namespace['x']}}]}}}}\n"
            )
            # Loading an upstream cloud entrypoint would fail this test.
            (source / "run_evolution.py").write_text("raise AssertionError('cloud entrypoint')\n")
        self.example = self.root / "circle_packing"
        self.instructions = self.root / "instructions.md"
        self.instructions.write_text("Pack circles into a unit square.")
        self.env = self.root / "example.env"
        self.env.write_text("MODEL=offline-model\nMAX_PROGRAMS_GENERATED=3\n")
        self.payload = {
            "program_language": "python",
            "problem_description": self.instructions.read_text(),
            "run_settings": {"max_programs": 3},
            "generation_settings": {"models": [{"name": "offline-model"}]},
        }
        self.config_json = self.root / "experiment.json"
        self.config_json.write_text(json.dumps(self.payload))
        self.json_args = [
            "--alphaevolve-config",
            str(self.config_json),
            "--api-base",
            "http://localhost:8080/v1",
        ]
        self.env_args = [
            "--alphaevolve-env",
            str(self.env),
            "--instructions",
            str(self.instructions),
            "--api-base",
            "http://localhost:8080/v1",
        ]

    def invoke(self, args):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            runner.main([str(self.example), *args])
        return json.loads(stdout.getvalue())

    def assert_invalid(self, args):
        stderr = io.StringIO()
        with (
            contextlib.redirect_stderr(stderr),
            patch.object(
                runner,
                "load_evaluator",
                side_effect=AssertionError("must validate before loading evaluator"),
            ),
            patch.object(runner, "run_evolution") as evolve,
        ):
            with self.assertRaises(SystemExit) as caught:
                self.invoke(args)
        self.assertEqual(caught.exception.code, 2)
        evolve.assert_not_called()
        self.assertNotIn("Traceback", stderr.getvalue())
        return stderr.getvalue()

    def test_no_config_and_imported_configs_default_to_seed_only(self):
        for args in ([], self.json_args, self.env_args):
            with (
                self.subTest(args=args),
                patch.object(runner, "run_evolution") as evolve,
                patch(
                    "openevolve.llm.openai.openai.AsyncOpenAI", side_effect=AssertionError("client")
                ),
                patch("socket.socket.connect", side_effect=AssertionError("network")),
            ):
                self.assertEqual(self.invoke(args)["metrics"]["sum_of_radii"], 1)
                evolve.assert_not_called()

    def test_explicit_json_evolution_uses_imported_budget(self):
        with patch.object(
            runner,
            "run_evolution",
            return_value=SimpleNamespace(metrics={"combined_score": 2}, best_score=2),
        ) as evolve:
            result = self.invoke([*self.json_args, "--evolve"])
        config = evolve.call_args.kwargs["config"]
        self.assertEqual(config.max_iterations, 3)
        self.assertEqual(config.llm.models[0].name, "offline-model")
        self.assertIsNone(evolve.call_args.kwargs["iterations"])
        self.assertEqual(result["best_score"], 2)

    def test_env_evolution_can_reduce_budget_and_select_key_by_variable_name(self):
        with (
            patch.dict(os.environ, {"OE_INTEROP_TEST_KEY": "do-not-print-key"}),
            patch.object(
                runner,
                "run_evolution",
                return_value=SimpleNamespace(metrics={"combined_score": 2}, best_score=2),
            ) as evolve,
        ):
            result = self.invoke(
                [
                    *self.env_args,
                    "--evolve",
                    "--iterations",
                    "2",
                    "--api-key-env",
                    "OE_INTEROP_TEST_KEY",
                ]
            )
        kwargs = evolve.call_args.kwargs
        self.assertEqual(kwargs["iterations"], 2)
        self.assertEqual(kwargs["config"].max_iterations, 3)
        self.assertEqual(kwargs["config"].llm.models[0].api_key, "do-not-print-key")
        self.assertNotIn("do-not-print-key", json.dumps(result))

    def test_legacy_yaml_mode_keeps_one_iteration_default_and_explicit_override(self):
        config_path = self.root / "oe.yaml"
        for extra, expected in (([], 1), (["--iterations", "2"], 2)):
            with (
                self.subTest(extra=extra),
                patch.object(
                    runner, "run_evolution", return_value=SimpleNamespace(metrics={}, best_score=1)
                ) as evolve,
            ):
                self.invoke(["--config", str(config_path), *extra])
            self.assertEqual(evolve.call_args.kwargs["config"], config_path)
            self.assertEqual(evolve.call_args.kwargs["iterations"], expected)

    def test_invalid_flag_combinations_fail_before_loading_example_code(self):
        for args in (
            ["--evolve"],
            ["--iterations", "1"],
            [*self.json_args, "--iterations", "1"],
            ["--api-base", "http://localhost/v1"],
            ["--api-key-env", "ANY_KEY"],
            ["--instructions", str(self.instructions)],
            ["--alphaevolve-config", str(self.config_json)],
            ["--alphaevolve-env", str(self.env), "--api-base", "http://localhost/v1"],
            [*self.json_args, "--instructions", str(self.instructions)],
            [*self.json_args, "--config", "oe.yaml"],
            [*self.json_args, "--alphaevolve-env", str(self.env)],
            ["--config", "oe.yaml", "--api-base", "http://localhost/v1"],
        ):
            with self.subTest(args=args):
                self.assert_invalid(args)

    def test_iteration_override_must_be_positive_and_cannot_increase_imported_budget(self):
        for source in (self.json_args, self.env_args):
            for count in ("0", "-1", "4", "not-an-int"):
                with self.subTest(source=source, count=count):
                    self.assert_invalid([*source, "--evolve", "--iterations", count])

    def test_malformed_missing_or_invalid_config_files_are_usage_errors(self):
        for text in ("not JSON", "[]", "{}"):
            self.config_json.write_text(text)
            with self.subTest(text=text):
                self.assert_invalid(self.json_args)
        self.config_json.unlink()
        self.assert_invalid(self.json_args)
        self.env.write_text("MODEL=x\nMAX_PROGRAMS_GENERATED=2\nMAX_PROGRAMS_EVALUATED=1\n")
        self.assertIn("MAX_PROGRAMS_EVALUATED", self.assert_invalid(self.env_args))
        self.instructions.unlink()
        self.assert_invalid(self.env_args)

    def test_explicit_missing_or_blank_key_variable_is_not_replaced_by_ambient_key(self):
        for value in (None, "", "   "):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"OPENAI_API_KEY": "ambient-key"}),
            ):
                if value is None:
                    os.environ.pop("OE_INTEROP_TEST_KEY", None)
                else:
                    os.environ["OE_INTEROP_TEST_KEY"] = value
                message = self.assert_invalid(
                    [*self.json_args, "--api-key-env", "OE_INTEROP_TEST_KEY"]
                )
                self.assertIn("--api-key-env", message)
                self.assertNotIn("ambient-key", message)

    def test_missing_optional_parser_is_actionable_and_json_seed_still_works(self):
        with patch(
            "openevolve.compat.alphaevolve_env._read_env",
            side_effect=ImportError('pip install "openevolve[alphaevolve-env]"'),
        ):
            self.assertIn("openevolve[alphaevolve-env]", self.assert_invalid(self.env_args))
            self.assertEqual(self.invoke(self.json_args)["metrics"]["sum_of_radii"], 1)

    def test_local_model_types_flag_loads_legacy_imports_without_retaining_aliases(self):
        (self.example / "src/evaluate.py").write_text(
            "from alpha_evolve.models import (AlphaEvolveEvaluationScore, "
            "AlphaEvolveEvaluationScores, AlphaEvolveProgramEvaluation)\n"
            "def circle_packing_evaluation(candidate):\n"
            "    return AlphaEvolveProgramEvaluation(scores=AlphaEvolveEvaluationScores(\n"
            "        scores=[AlphaEvolveEvaluationScore(metric='sum_of_radii', score=3)]))\n"
        )
        before = {
            name: value for name, value in sys.modules.items() if name.startswith("alpha_evolve")
        }
        result = self.invoke(["--local-model-types"])
        self.assertEqual(result["metrics"]["sum_of_radii"], 3)
        self.assertEqual(
            {name: value for name, value in sys.modules.items() if name.startswith("alpha_evolve")},
            before,
        )

    def test_local_model_loading_is_not_enabled_implicitly(self):
        with patch(
            "openevolve.compat.alpha_evolve.load_evaluator",
            side_effect=AssertionError("must require explicit opt-in"),
        ):
            self.assertEqual(self.invoke([])["metrics"]["sum_of_radii"], 1)

    def test_local_model_types_flag_runs_real_evolution_after_alias_cleanup(self):
        source = self.example / "src/evaluate.py"
        source.write_text(
            "from alpha_evolve.models import AlphaEvolveProgramEvaluation\n" + source.read_text()
        )

        def offline_config(*args, **kwargs):
            config = config_from_alphaevolve_env(*args, **kwargs)
            config.database.num_islands = 1
            for model in config.llm.models + config.llm.evaluator_models:
                model.init_client = offline_model
            return config

        self.env.write_text("MODEL=offline-model\nMAX_PROGRAMS_GENERATED=1\n")
        with (
            patch(
                "openevolve.compat.alphaevolve_env.config_from_alphaevolve_env",
                side_effect=offline_config,
            ),
            patch("socket.socket.connect", side_effect=AssertionError("network")),
        ):
            result = self.invoke(
                [
                    *self.env_args,
                    "--local-model-types",
                    "--evolve",
                    "--output-dir",
                    str(self.root / "local-output"),
                ]
            )
        self.assertEqual(result["metrics"]["sum_of_radii"], 2)
        self.assertFalse(any(name.startswith("alpha_evolve") for name in sys.modules))

    def test_missing_seed_or_evaluator_is_reported_before_import(self):
        (self.example / "src/program.py").unlink()
        self.assertIn("Required example file", self.assert_invalid([]))

    def test_imported_env_runs_one_generation_through_real_controller_and_worker(self):
        self.env.write_text("MODEL=offline-model\nMAX_PROGRAMS_GENERATED=1\n")

        def offline_config(*args, **kwargs):
            config = config_from_alphaevolve_env(*args, **kwargs)
            config.database.num_islands = 1
            for model in config.llm.models + config.llm.evaluator_models:
                model.init_client = offline_model
            return config

        with (
            patch(
                "openevolve.compat.alphaevolve_env.config_from_alphaevolve_env",
                side_effect=offline_config,
            ),
            patch("socket.socket.connect", side_effect=AssertionError("network")),
        ):
            result = self.invoke(
                [*self.env_args, "--evolve", "--output-dir", str(self.root / "output")]
            )
        self.assertEqual(result["metrics"]["sum_of_radii"], 2)
        self.assertEqual(result["best_score"], 2)
        self.assertIn("x = 1", (self.example / "src/program.py").read_text())
        self.assertTrue((self.root / "output").is_dir())

    def test_real_cli_subprocess_evaluates_each_supported_fixture(self):
        for name, metric in (
            ("circle_packing", "sum_of_radii"),
            ("tsp", "neg_tour_length"),
            ("signal_processing", "overall_score"),
        ):
            with self.subTest(name=name):
                result = subprocess.run(
                    [sys.executable, str(RUNNER), str(self.root / name)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)["metrics"][metric], 1)


if __name__ == "__main__":
    unittest.main()
