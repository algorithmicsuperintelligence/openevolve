"""Entry-point configuration swaps, staged source isolation and offline evolution."""

import contextlib
import importlib.util
import io
import json
import os
import secrets
import shutil
import signal
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from openevolve.compat.alphaevolve import config_from_alphaevolve

RUNNER = Path(__file__).resolve().parents[1] / "examples/alphaevolve_interop/run_entrypoint.py"
spec = importlib.util.spec_from_file_location("_local_ae_entrypoint_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class OfflineModel:
    def __init__(self, config):
        self.model = config.name

    async def generate_with_context(self, system_message, messages, **kwargs):
        return "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE"


def offline_model(config):
    return OfflineModel(config)


class TestAlphaEvolveEntrypoint(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        self.addCleanup(lambda: [signal.signal(sig, handler) for sig, handler in handlers.items()])
        self.before = {
            name: module for name, module in sys.modules.items() if name.startswith("alpha_evolve")
        }
        self.options = {"api_base": "http://localhost:8000/v1", "model": "offline-model"}
        for name in ("circle_packing", "tsp"):
            metric = runner.EXAMPLES[name]
            directory = self.root / name
            source = directory / "src"
            helpers = source / "utils"
            helpers.mkdir(parents=True)
            (source / "__init__.py").write_text("")
            (helpers / "__init__.py").write_text("")
            (directory / "instructions.md").write_text("Maximize x using the original interface.")
            (directory / ".env").write_text("DO_NOT_LOAD_EXAMPLE_ENV=1\n")
            (source / ".env.private").write_text("DO_NOT_COPY=1\n")
            (source / "program.py").write_text("# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END\n")
            (source / "evaluate.py").write_text(
                "from pathlib import Path\n"
                "from alpha_evolve.models import (AlphaEvolveEvaluationScore, "
                "AlphaEvolveEvaluationScores, AlphaEvolveProgramEvaluation)\n"
                "INITIAL_PROGRAM_CODE = Path(__file__).with_name('program.py').read_text()\n"
                "def evaluate(candidate):\n"
                "    namespace = {}\n"
                "    exec(candidate['content']['files'][0]['content'], namespace)\n"
                "    return AlphaEvolveProgramEvaluation(scores=AlphaEvolveEvaluationScores(\n"
                f"        scores=[AlphaEvolveEvaluationScore(metric={metric!r}, score=namespace['x'])]))\n"
            )
            (helpers / "report.py").write_text(
                "from pathlib import Path\n"
                "from alpha_evolve.visualization import get_score\n"
                "def report(programs, config):\n"
                "    root = Path(__file__).resolve().parents[2]\n"
                "    assert get_score(programs[0], config['metric']) >= 1\n"
                "    import json\n"
                "    (root / 'report.json').write_text(json.dumps(config))\n"
            )
            (source / "run_evolution.py").write_text(
                "import asyncio\n"
                "from pathlib import Path\n"
                "from dotenv import load_dotenv\n"
                "load_dotenv()\n"
                "from alpha_evolve.client import AlphaEvolveClient\n"
                "from alpha_evolve.experiment import AlphaEvolveExperiment\n"
                "from alpha_evolve.controller import run_controller_loop\n"
                "from .evaluate import evaluate, INITIAL_PROGRAM_CODE\n"
                "from .utils.report import report\n"
                "MAX_PROGRAMS_GENERATED = MAX_PROGRAMS_EVALUATED = 100\n"
                "MODEL = MODEL_1 = MODEL_2 = 'must-be-replaced'\n"
                "MODEL_1_WEIGHT = MODEL_2_WEIGHT = 0.5\n"
                "PARALLEL_EVALUATION = True\n"
                "CONCURRENCY = WORKER_CONCURRENCY = 32\n"
                "def main():\n"
                "    client = AlphaEvolveClient(project_id='ignored-project')\n"
                "    experiment = AlphaEvolveExperiment(client, evaluate, "
                "MAX_PROGRAMS_EVALUATED, parallel_evaluation=PARALLEL_EVALUATION)\n"
                f"    if {name!r} == 'circle_packing':\n"
                "        models = [{'name': MODEL_1, 'weight': MODEL_1_WEIGHT}]\n"
                "        assert MODEL_2 == '' and MODEL_2_WEIGHT == 0 and WORKER_CONCURRENCY == 1\n"
                "    else:\n"
                "        models = [{'name': MODEL}]\n"
                "    experiment.create_experiment({\n"
                "        'program_language': 'python',\n"
                "        'problem_description': Path(__file__).resolve().parents[1].joinpath('instructions.md').read_text(),\n"
                "        'run_settings': {'max_programs': MAX_PROGRAMS_GENERATED, 'concurrency': CONCURRENCY},\n"
                "        'generation_settings': {'models': models},\n"
                "    })\n"
                "    experiment.create_initial_program({'content': {'files': [{'path': 'program.py', 'content': INITIAL_PROGRAM_CODE}]}})\n"
                "    experiment.start_experiment()\n"
                "    asyncio.run(run_controller_loop(experiment))\n"
                "    report(experiment.list_programs()['alphaEvolvePrograms'], {\n"
                f"        'metric': {metric!r}, 'model': models[0]['name'],\n"
                "        'parallel': PARALLEL_EVALUATION, 'concurrency': CONCURRENCY,\n"
                "        'budget': MAX_PROGRAMS_GENERATED, 'api_base': client.api_base,\n"
                "    })\n"
            )
        self.example = self.root / "circle_packing"

    def tearDown(self):
        self.assertEqual(
            {
                name: module
                for name, module in sys.modules.items()
                if name.startswith("alpha_evolve")
            },
            self.before,
        )
        self.assertFalse(any(name.startswith("_openevolve_ae_entrypoint_") for name in sys.modules))

    def offline_config(self, *args, **kwargs):
        config = config_from_alphaevolve(*args, **kwargs)
        config.database.num_islands = 1
        for model in config.llm.models + config.llm.evaluator_models:
            model.init_client = offline_model
        return config

    def run_offline(self, example, **options):
        with (
            patch(
                "openevolve.compat.alpha_evolve.experiment.config_from_alphaevolve",
                side_effect=self.offline_config,
            ),
            patch("socket.socket.connect", side_effect=AssertionError("network")),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", UserWarning)
            return runner.run_entrypoint(example, **(self.options | options))

    def signal_example(self):
        example = self.root / "signal_processing"
        shutil.copytree(self.example, example)
        source = example / "src"
        evaluator = source / "evaluate.py"
        evaluator.write_text(
            "import os\n"
            + evaluator.read_text()
            .replace("'sum_of_radii'", "'overall_score'")
            .replace("def evaluate(candidate):", "def signal_processing_evaluation(candidate):")
            .replace(
                "    return AlphaEvolveProgramEvaluation",
                "    archive = Path(os.environ['SAVE_PROGRAM_DIR'])\n"
                "    archive.mkdir(exist_ok=True)\n"
                "    (archive / 'candidate.py').write_text(candidate['content']['files'][0]['content'])\n"
                "    return AlphaEvolveProgramEvaluation",
            )
        )
        (source / "run_evolution.py").write_text(
            "import asyncio\n"
            "import json\n"
            "import os\n"
            "from pathlib import Path\n"
            "from dotenv import load_dotenv\n"
            "from alpha_evolve.client import AlphaEvolveClient\n"
            "from alpha_evolve.experiment import AlphaEvolveExperiment\n"
            "from alpha_evolve.controller import run_controller_loop\n"
            "from .evaluate import signal_processing_evaluation, INITIAL_PROGRAM_CODE\n"
            "load_dotenv()\n"
            "MAX_PROGRAMS_GENERATED = MAX_PROGRAMS_EVALUATED = 10\n"
            "CONCURRENCY = 2\n"
            "def main():\n"
            "    client = AlphaEvolveClient(project_id='ignored-project')\n"
            "    config = {\n"
            "        'program_language': 'python', 'problem_description': 'Maximize x',\n"
            "        'run_settings': {'max_programs': MAX_PROGRAMS_GENERATED, 'concurrency': CONCURRENCY},\n"
            "    }\n"
            "    try:\n"
            "        probe = AlphaEvolveExperiment(client, signal_processing_evaluation, MAX_PROGRAMS_EVALUATED)\n"
            "        probe.create_experiment(config)\n"
            "        metadata = client.get_alpha_evolve_experiment(experiment_name=probe.experiment_name)\n"
            "    except Exception:\n"
            "        return\n"
            "    experiment = AlphaEvolveExperiment(client, signal_processing_evaluation, MAX_PROGRAMS_EVALUATED)\n"
            "    experiment.create_experiment(config)\n"
            "    assert 'generation_settings' not in config\n"
            "    experiment.create_initial_program({'content': {'files': [{'path': 'program.py', 'content': INITIAL_PROGRAM_CODE}]}})\n"
            "    experiment.start_experiment()\n"
            "    asyncio.run(run_controller_loop(experiment, num_samplers=CONCURRENCY, num_evaluators=CONCURRENCY))\n"
            "    result = experiment.list_programs({'order_by': 'score desc'})\n"
            "    output = Path(__file__).resolve().parents[1]\n"
            "    report = {'probe': metadata, 'result': result, 'budget': MAX_PROGRAMS_GENERATED,\n"
            "              'concurrency': CONCURRENCY, 'save_dir': os.environ['SAVE_PROGRAM_DIR']}\n"
            "    (output / 'report.json').write_text(json.dumps(report))\n"
        )
        return example

    def test_signal_original_lifecycle_runs_seed_and_real_worker_with_selected_model(self):
        example = self.signal_example()
        original = {p.relative_to(example): p.read_bytes() for p in example.rglob("*.py")}
        for iterations in (0, 1):
            output = self.root / f"signal-output-{iterations}"
            with self.subTest(iterations=iterations):
                result = self.run_offline(example, iterations=iterations, output_dir=output)
                self.assertEqual(result["metrics"]["overall_score"], 1 + iterations)
                report = json.loads((output / "report.json").read_text())
                self.assertEqual(report["probe"]["state"], "CREATED")
                self.assertEqual(report["budget"], max(1, iterations))
                self.assertEqual(report["concurrency"], 1)
                self.assertEqual(report["save_dir"], str(output / "programs"))
                self.assertTrue((output / "programs/candidate.py").is_file())
                self.assertTrue((output / "oe-output/best/best_program.py").is_file())
                self.assertNotEqual(
                    report["probe"]["name"],
                    report["result"]["alphaEvolvePrograms"][0]["name"].rsplit("/programs/", 1)[0],
                )
                for path, content in original.items():
                    self.assertEqual((example / path).read_bytes(), content)
                    self.assertEqual((output / path).read_bytes(), content)

    def test_signal_explicit_model_overrides_supplied_models_without_mutating_config(self):
        example = self.signal_example()
        source = example / "src/run_evolution.py"
        source.write_text(
            source.read_text()
            .replace(
                "'program_language':",
                "'generation_settings': {'models': [{'name': 'upstream-model'}]}, 'program_language':",
            )
            .replace(
                "assert 'generation_settings' not in config",
                "assert config['generation_settings']['models'] == [{'name': 'upstream-model'}]",
            )
        )
        configurations = []
        configure = self.offline_config

        def record(*args, **kwargs):
            local = configure(*args, **kwargs)
            configurations.append([(m.name, m.weight) for m in local.llm.models])
            return local

        with patch.object(self, "offline_config", side_effect=record):
            self.run_offline(example, iterations=1)
        self.assertEqual(configurations, [[("offline-model", 1.0)]] * 2)

    def test_signal_probe_query_failure_or_early_return_is_not_success(self):
        example = self.signal_example()
        with patch.object(
            runner.AlphaEvolveClient,
            "get_alpha_evolve_experiment",
            side_effect=ValueError("lookup failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "without completing"):
                self.run_offline(example)
        source = example / "src/run_evolution.py"
        source.write_text(
            source.read_text().replace("    experiment.start_experiment()", "    return")
        )
        with self.assertRaisesRegex(RuntimeError, "without completing"):
            self.run_offline(example)

    def test_signal_archive_environment_is_restored_on_success_and_failure(self):
        example = self.signal_example()
        inherited = self.root / "must-not-create-archives"
        with patch.dict(os.environ, {"SAVE_PROGRAM_DIR": str(inherited)}):
            self.run_offline(example)
            self.assertEqual(os.environ["SAVE_PROGRAM_DIR"], str(inherited))
            source = example / "src/run_evolution.py"
            source.write_text(
                source.read_text().replace(
                    "    experiment.start_experiment()",
                    "    raise RuntimeError('entrypoint failed')",
                )
            )
            with self.assertRaisesRegex(RuntimeError, "entrypoint failed"):
                self.run_offline(example)
            self.assertEqual(os.environ["SAVE_PROGRAM_DIR"], str(inherited))
        self.assertFalse(inherited.exists())

    def test_signal_layout_drift_and_malformed_generation_settings_are_rejected(self):
        example = self.signal_example()
        source = example / "src/run_evolution.py"
        original = source.read_text()
        source.write_text(original.replace("CONCURRENCY = 2", "REMOVED_CONCURRENCY = 2"))
        with self.assertRaisesRegex(ValueError, "layout changed"):
            self.run_offline(example)
        source.write_text(
            original.replace(
                "'program_language':", "'generation_settings': None, 'program_language':"
            )
        )
        with self.assertRaisesRegex(RuntimeError, "without completing"):
            self.run_offline(example)

    def test_original_main_relative_helpers_and_real_workers_run_with_config_swaps(self):
        for name in ("circle_packing", "tsp"):
            metric = runner.EXAMPLES[name]
            example = self.root / name
            originals = {
                path.relative_to(example): path.read_bytes()
                for path in example.rglob("*")
                if path.is_file()
            }
            for iterations in (0, 1):
                output = self.root / f"output-{name}-{iterations}"
                with self.subTest(name=name, iterations=iterations):
                    result = self.run_offline(example, iterations=iterations, output_dir=output)
                    self.assertEqual(result["metrics"][metric], 1 + iterations)
                    self.assertEqual(result["generation_attempt_limit"], iterations)
                    report = json.loads((output / "report.json").read_text())
                    self.assertEqual(report["model"], "offline-model")
                    self.assertEqual(report["budget"], max(1, iterations))
                    self.assertFalse(report["parallel"])
                    self.assertEqual(report["concurrency"], 1)
                    for path, content in originals.items():
                        self.assertEqual((example / path).read_bytes(), content)
                        if path.suffix == ".py":
                            self.assertEqual((output / path).read_bytes(), content)
                    self.assertFalse((output / ".env").exists())
                    self.assertFalse((output / "src/.env.private").exists())
                    self.assertTrue((output / "oe-output/best/best_program.py").is_file())

    def test_temporary_run_cleans_up_sources_and_report(self):
        seen = []
        original_loader = runner._load_entrypoint

        def load(root, settings):
            seen.append(root)
            return original_loader(root, settings)

        with patch.object(runner, "_load_entrypoint", side_effect=load):
            result = self.run_offline(self.example)
        self.assertIsNone(result["output_dir"])
        self.assertFalse(seen[0].exists())

    def test_upstream_constructor_cannot_override_selected_endpoint_budget_or_output(self):
        source = self.example / "src/run_evolution.py"
        source.write_text(
            source.read_text().replace(
                "project_id='ignored-project'",
                "project_id='ignored-project', api_base='https://must-not-contact.invalid', "
                "iterations=100, output_dir='must-not-create'",
            )
        )
        output = self.root / "selected-output"
        result = self.run_offline(self.example, output_dir=output)
        self.assertEqual(result["generation_attempt_limit"], 0)
        report = json.loads((output / "report.json").read_text())
        self.assertEqual(report["api_base"], self.options["api_base"])
        self.assertTrue((output / "oe-output/best/best_program.py").is_file())

    def test_explicit_output_must_be_new_and_outside_source_tree(self):
        existing = self.root / "existing"
        existing.mkdir()
        marker = existing / "keep.txt"
        marker.write_text("user data")
        for destination, error in (
            (existing, FileExistsError),
            (self.example / "new-output", ValueError),
        ):
            with (
                self.subTest(destination=destination),
                patch.object(runner, "_load_entrypoint") as load,
            ):
                with self.assertRaises(error):
                    runner.run_entrypoint(self.example, **self.options, output_dir=destination)
                load.assert_not_called()
        self.assertEqual(marker.read_text(), "user data")
        link = self.root / "dangling-link"
        link.symlink_to(self.root / "missing-output")
        with self.assertRaises(FileExistsError):
            runner.run_entrypoint(self.example, **self.options, output_dir=link)
        self.assertTrue(link.is_symlink())
        self.assertFalse(link.exists())

    def test_bad_configuration_fails_before_creating_output_or_importing(self):
        output = self.root / "must-not-exist"
        for options in (
            {"model": ""},
            {"api_base": ""},
            {"iterations": -1},
            {"iterations": True},
            {"api_key": ""},
        ):
            with self.subTest(options=options), patch.object(runner, "_load_entrypoint") as load:
                with self.assertRaises(ValueError):
                    runner.run_entrypoint(
                        self.example, **(self.options | options), output_dir=output
                    )
                load.assert_not_called()
            self.assertFalse(output.exists())

    def test_unsupported_layout_missing_sources_and_symlinks_are_rejected(self):
        unsupported = self.root / "adaptive_sort"
        unsupported.mkdir()
        with self.assertRaisesRegex(ValueError, "Supported upstream"):
            runner.run_entrypoint(unsupported, **self.options)
        seed = self.example / "src/program.py"
        content = seed.read_text()
        seed.unlink()
        with self.assertRaisesRegex(ValueError, "Missing required"):
            runner.run_entrypoint(self.example, **self.options)
        elsewhere = self.root / "outside.py"
        elsewhere.write_text(content)
        seed.symlink_to(elsewhere)
        with self.assertRaisesRegex(ValueError, "symlink"):
            runner.run_entrypoint(self.example, **self.options)

    def test_import_disables_dotenv_discovery_and_restores_function_on_success_or_failure(self):
        import dotenv

        for fail in (False, True):
            with (
                self.subTest(fail=fail),
                patch.object(
                    dotenv, "load_dotenv", side_effect=AssertionError("must not search .env")
                ) as original,
            ):
                if fail:
                    (self.example / "src/run_evolution.py").write_text(
                        "raise RuntimeError('failed import')"
                    )
                    with self.assertRaisesRegex(RuntimeError, "failed import"):
                        self.run_offline(self.example)
                else:
                    self.run_offline(self.example)
                self.assertIs(dotenv.load_dotenv, original)
                original.assert_not_called()

    def test_import_configuration_wins_over_invalid_ambient_settings_and_restores_environment(self):
        source = self.example / "src/run_evolution.py"
        source.write_text(
            "import os\n"
            "assert int(os.environ['MAX_PROGRAMS_GENERATED']) == 1\n"
            "assert float(os.environ['MODEL_1_WEIGHT']) == 1.0\n"
            "assert os.environ['MODEL_1'] == 'offline-model'\n" + source.read_text()
        )
        settings = {"MAX_PROGRAMS_GENERATED": "bad", "MODEL_1_WEIGHT": "bad", "MODEL_1": "ambient"}
        with patch.dict(os.environ, settings):
            self.run_offline(self.example)
            self.assertEqual({key: os.environ[key] for key in settings}, settings)

    def test_existing_cloud_sdk_is_preserved_and_temporary_copy_is_cleaned(self):
        original = ModuleType("alpha_evolve")
        copies = []
        actual = runner._load_entrypoint

        def load(root, settings):
            copies.append(root)
            return actual(root, settings)

        with (
            patch.dict(sys.modules, {"alpha_evolve": original}),
            patch.object(runner, "_load_entrypoint", side_effect=load),
        ):
            with self.assertRaisesRegex(RuntimeError, "already loaded"):
                self.run_offline(self.example)
            self.assertIs(sys.modules["alpha_evolve"], original)
        self.assertFalse(copies[0].exists())

    def test_main_return_without_completed_experiment_is_an_error(self):
        source = self.example / "src/run_evolution.py"
        source.write_text(source.read_text().replace("def main():", "def main():\n    return"))
        with self.assertRaisesRegex(RuntimeError, "without completing"):
            self.run_offline(self.example)

    def test_layout_drift_and_import_interruptions_leave_no_aliases(self):
        source = self.example / "src/run_evolution.py"
        original = source.read_text()
        source.write_text(original.replace("WORKER_CONCURRENCY = 32", "REMOVED_WORKER = 32"))
        with self.assertRaisesRegex(ValueError, "layout changed"):
            self.run_offline(self.example)
        source.write_text("raise KeyboardInterrupt()")
        with self.assertRaises(KeyboardInterrupt):
            self.run_offline(self.example)

    def test_cli_seed_default_evolution_opt_in_and_key_selection(self):
        token = secrets.token_urlsafe(20)
        for options, expected in (
            ([], 0),
            (["--evolve"], 1),
            (["--evolve", "--iterations", "2"], 2),
        ):
            output = io.StringIO()
            with (
                self.subTest(options=options),
                patch.dict(os.environ, {"OE_ENTRYPOINT_TEST_KEY": token}),
                patch.object(
                    runner, "run_entrypoint", return_value={"metrics": {"score": 1}}
                ) as run,
                contextlib.redirect_stdout(output),
            ):
                runner.main(
                    [
                        str(self.example),
                        "--api-base",
                        "local",
                        "--model",
                        "model",
                        "--api-key-env",
                        "OE_ENTRYPOINT_TEST_KEY",
                        *options,
                    ]
                )
            self.assertEqual(run.call_args.kwargs["iterations"], expected)
            self.assertEqual(run.call_args.kwargs["api_key"], token)
            self.assertEqual(json.loads(output.getvalue()), {"metrics": {"score": 1}})
            self.assertNotIn(token, output.getvalue())

    def test_cli_rejects_invalid_budget_or_key_before_import(self):
        for options in (
            ["--iterations", "1"],
            ["--evolve", "--iterations", "0"],
            ["--api-key-env", "OE_ENTRYPOINT_MISSING"],
        ):
            with (
                self.subTest(options=options),
                patch.object(runner, "run_entrypoint") as run,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit):
                    runner.main(
                        [str(self.example), "--api-base", "local", "--model", "model", *options]
                    )
                run.assert_not_called()

    def test_named_key_is_not_replaced_by_ambient_key(self):
        with (
            patch.dict(os.environ, {"OE_ENTRYPOINT_BLANK": "", "OPENAI_API_KEY": "ambient"}),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit):
                runner.main(
                    [
                        str(self.example),
                        "--api-base",
                        "local",
                        "--model",
                        "model",
                        "--api-key-env",
                        "OE_ENTRYPOINT_BLANK",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
