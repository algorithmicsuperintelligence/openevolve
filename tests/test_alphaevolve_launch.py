"""Offline preflight, explicit cloud opt-in and launch failure-path tests."""

import contextlib
import copy
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

from openevolve.compat.alphaevolve_bundle import export_to_alphaevolve
from openevolve.compat.alphaevolve_launch import main
from openevolve.config import Config


class TestAlphaEvolveLaunch(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.program = self.inputs / "seed.py"
        self.program.write_bytes("# 中文\r\n7\r\n".encode("utf-8"))
        self.evaluator = self.inputs / "evaluator.py"
        self.evaluator.write_text(
            "from dataclasses import dataclass\nfrom pathlib import Path\n"
            "@dataclass\nclass Score:\n    value: int\n"
            "def evaluate(path):\n    print('evaluator diagnostic')\n"
            "    return {'score': Score(len(Path(path).read_bytes())).value}\n"
        )
        self.bundle = self.root / "bundle"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            export_to_alphaevolve(
                self.program,
                self.evaluator,
                Config.from_dict(
                    {
                        "language": "python",
                        "max_iterations": 3,
                        "llm": {"models": [{"name": "gemini-3.5-flash", "weight": 1}]},
                    }
                ),
                self.bundle,
                title="Example 中文",
                problem_description="Score the source.",
                objective_metric="score",
            )
        self.expected_score = len(self.program.read_bytes())
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.run_options = [
            "--run",
            "--project-id",
            "review-project",
            "--engine",
            "review-engine",
            "--max-programs-evaluated",
            "1",
        ]

    def launch(self, *options, bundle=None):
        with contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
            return main([str(bundle or self.bundle), *options])

    @contextlib.contextmanager
    def sdk(self):
        self.client = Mock()
        self.client.start_experiment.return_value = {}
        self.client_type = Mock(return_value=self.client)
        self.experiment = Mock()
        self.experiment.experiment_name = "offline-session/alphaEvolveExperiments/example"
        self.experiment.initial_program_name = "offline-seed"
        self.experiment.stats = {"num_programs_generated": 1, "num_programs_evaluated": 1}
        self.experiment_type = Mock(return_value=self.experiment)
        self.config_schema = Mock()
        self.evaluation_schema = Mock()

        async def controller(experiment, **options):
            self.controller_options = options
            evaluate = self.experiment_type.call_args.args[1]
            result = evaluate({"content": self.initial["content"]})
            self.assertEqual(result["scores"]["scores"][0]["score"], self.expected_score)

        self.controller = Mock(side_effect=controller)

        def capture_seed(candidate):
            self.initial = copy.deepcopy(candidate)

        self.experiment.create_initial_program.side_effect = capture_seed
        modules = {
            name: types.ModuleType(name)
            for name in (
                "alpha_evolve",
                "alpha_evolve.client",
                "alpha_evolve.controller",
                "alpha_evolve.experiment",
                "alpha_evolve.models",
            )
        }
        modules["alpha_evolve"].__path__ = []
        modules["alpha_evolve.client"].AlphaEvolveClient = self.client_type
        modules["alpha_evolve.controller"].run_controller_loop = self.controller
        modules["alpha_evolve.experiment"].AlphaEvolveExperiment = self.experiment_type
        modules["alpha_evolve.models"].AlphaEvolveExperimentConfig = self.config_schema
        modules["alpha_evolve.models"].AlphaEvolveProgramEvaluation = self.evaluation_schema
        with patch.dict(sys.modules, modules):
            yield

    def assert_no_bundle_modules(self):
        self.assertFalse(any(name.startswith("_openevolve_alphaevolve_") for name in sys.modules))

    def test_default_preflight_never_imports_sdk_evaluator_or_executes_source(self):
        marker = self.root / "must-not-exist"
        self.bundle.joinpath("src/openevolve_evaluator.py").write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
            "def evaluate(path): return {'score': 1}\n"
        )
        original_import = __import__

        def guarded_import(name, *args, **kwargs):
            if name.startswith(("alpha_evolve", "_openevolve_alphaevolve_")):
                raise AssertionError("preflight imported executable code")
            return original_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=guarded_import),
            patch(
                "socket.socket.connect", side_effect=AssertionError("network forbidden")
            ) as sockets,
        ):
            self.assertEqual(self.launch(), 0)
        sockets.assert_not_called()
        self.assertFalse(marker.exists())
        plan = json.loads(self.stdout.getvalue())
        self.assertEqual(plan["mode"], "dry-run")
        self.assertFalse(plan["sdk_and_evaluator_imported"])
        self.assertEqual(plan["max_generation_attempts"], 3)
        self.assertEqual(plan["title"], "Example 中文")
        self.assertNotIn("content", plan)
        self.assert_no_bundle_modules()

    def test_explicit_dry_run_ignores_cloud_environment_and_keeps_files_unchanged(self):
        before = {p: p.read_bytes() for p in self.bundle.rglob("*") if p.is_file()}
        with patch.dict(os.environ, {"PROJECT_ID": "not-selected", "GE_APP_ID": "not-selected"}):
            self.assertEqual(self.launch("--dry-run"), 0)
        self.assertIsNone(json.loads(self.stdout.getvalue())["project_id"])
        self.assertIsNone(json.loads(self.stdout.getvalue())["engine"])
        self.assertEqual(before, {p: p.read_bytes() for p in self.bundle.rglob("*") if p.is_file()})

    def test_relocated_generated_entry_runs_from_unrelated_directory(self):
        moved = self.root / "moved"
        self.bundle.rename(moved)
        shutil.rmtree(self.inputs)
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        completed = subprocess.run(
            [sys.executable, str(moved / "run_alphaevolve.py")],
            cwd=elsewhere,
            env=environment,
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["bundle_dir"], str(moved.resolve()))

    def test_invalid_run_options_fail_before_loading_sdk(self):
        cases = [
            ["--run"],
            ["--run", "--project-id", "review-project"],
            ["--run", "--dry-run"],
            ["--max-programs-evaluated", "0"],
            ["--max-programs-evaluated", "-1"],
            ["--max-programs-evaluated", "1.5"],
            ["--max-programs-evaluated", "4"],
            ["--idle-timeout-seconds", "0"],
            ["--project-id", "project/path"],
            ["--engine", "with space"],
            ["--location", "https://another-host"],
        ]
        with self.sdk():
            for options in cases:
                with self.subTest(options=options), self.assertRaises(SystemExit) as caught:
                    self.launch(*options)
                self.assertEqual(caught.exception.code, 2)
            self.client_type.assert_not_called()
            self.config_schema.model_validate.assert_not_called()

    def test_missing_bundle_files_fail_before_cloud_calls(self):
        files = (
            "experiment.json",
            "evaluation.json",
            "src/program.py",
            "src/openevolve_evaluator.py",
            "src/evaluate.py",
            "src/__init__.py",
        )
        with self.sdk():
            for relative in files:
                path = self.bundle / relative
                data = path.read_bytes()
                path.unlink()
                try:
                    with self.subTest(file=relative), self.assertRaises(SystemExit) as caught:
                        self.launch(*self.run_options)
                    self.assertEqual(caught.exception.code, 2)
                finally:
                    path.write_bytes(data)
            self.client_type.assert_not_called()

    def test_invalid_configuration_subset_is_rejected(self):
        path = self.bundle / "experiment.json"
        original = json.loads(path.read_text())
        changes = [
            {"program_language": "cpp"},
            {"title": " "},
            {"run_settings": {}},
            {"run_settings": {"max_programs": True}},
            {"run_settings": {"max_programs": 3, "concurrency": 0}},
            {"generation_settings": {"models": []}},
            {"generation_settings": {"models": [{"name": "local-model", "weight": 1}]}},
            {"generation_settings": {"models": [{"name": "gemini-x", "weight": True}]}},
            {"generation_settings": {"models": [{"name": "gemini-x", "weight": -1}]}},
            {"generation_settings": {"models": [{"name": "gemini-x", "weight": float("nan")}]}},
            {"unrecognized_field": "must not be silently dropped"},
        ]
        for change in changes:
            path.write_text(json.dumps({**original, **change}))
            with self.subTest(change=change), self.assertRaises(SystemExit) as caught:
                self.launch()
            self.assertEqual(caught.exception.code, 2)

    def test_invalid_evaluation_options_and_python_are_rejected(self):
        path = self.bundle / "evaluation.json"
        original = path.read_bytes()
        for data in (
            "[",
            "null",
            '{"objective_metric":"error","failure_score":-1}',
            '{"objective_metric":"score","failure_score":Infinity}',
        ):
            path.write_text(data)
            with self.subTest(data=data), self.assertRaises(SystemExit) as caught:
                self.launch()
            self.assertEqual(caught.exception.code, 2)
        path.write_bytes(original)
        for file, source in (
            ("src/program.py", "return 1\n"),
            ("src/openevolve_evaluator.py", "def other(): pass\n"),
            ("src/evaluate.py", "def invalid(:\n"),
            ("src/__init__.py", "return 1\n"),
        ):
            path = self.bundle / file
            original = path.read_bytes()
            path.write_text(source)
            try:
                with self.subTest(file=file), self.assertRaises(SystemExit) as caught:
                    self.launch()
                self.assertEqual(caught.exception.code, 2)
            finally:
                path.write_bytes(original)

    def test_explicit_run_passes_real_seed_scores_and_separate_controller_options(self):
        marker = self.root / "seed-evaluated"
        path = self.bundle / "src/openevolve_evaluator.py"
        path.write_text(
            path.read_text().replace(
                "    print('evaluator diagnostic')",
                f"    Path({str(marker)!r}).touch()\n    print('evaluator diagnostic')",
            )
        )
        with self.sdk():

            def require_seed_first(**options):
                self.assertTrue(marker.exists())
                self.assertEqual(
                    options,
                    {
                        "project_id": "review-project",
                        "engine": "review-engine",
                        "location": "us",
                    },
                )
                return self.client

            self.client_type.side_effect = require_seed_first
            self.assertEqual(
                self.launch(*self.run_options, "--location", "us", "--idle-timeout-seconds", "17"),
                0,
            )
            self.experiment_type.assert_called_once()
            self.assertEqual(
                self.experiment_type.call_args.kwargs,
                {
                    "max_programs_evaluated": 1,
                    "parallel_evaluation": False,
                },
            )
            self.experiment.create_experiment.assert_called_once_with(
                json.loads((self.bundle / "experiment.json").read_text())
            )
            self.assertEqual(
                self.initial["content"]["files"][0]["content"].encode("utf-8"),
                self.program.read_bytes(),
            )
            self.assertEqual(
                self.initial["evaluation"]["scores"]["scores"],
                [{"metric": "score", "score": self.expected_score}],
            )
            self.assertEqual(
                self.controller_options,
                {"num_samplers": 1, "num_evaluators": 1, "idle_timeout_s": 17},
            )
            self.client.start_experiment.assert_called_once_with(self.experiment.experiment_name)
            self.experiment.start_experiment.assert_not_called()
        self.assertTrue(json.loads(self.stdout.getvalue())["evaluation_target_reached"])
        self.assertIn("evaluator diagnostic", self.stderr.getvalue())
        self.assert_no_bundle_modules()

    def test_missing_sdk_fails_without_loading_evaluator(self):
        with patch.dict(sys.modules, {"alpha_evolve": None}):
            with self.assertRaises(SystemExit) as caught:
                self.launch(*self.run_options)
        self.assertEqual(caught.exception.code, 1)
        self.assertIn("Install the official AlphaEvolve SDK", self.stderr.getvalue())
        self.assert_no_bundle_modules()

    def test_sdk_schema_rejection_happens_before_evaluator_or_client(self):
        with self.sdk():
            self.config_schema.model_validate.side_effect = ValueError("SDK schema changed")
            with self.assertRaises(SystemExit) as caught:
                self.launch(*self.run_options)
            self.assertEqual(caught.exception.code, 1)
            self.client_type.assert_not_called()
        self.assertNotIn("evaluator diagnostic", self.stderr.getvalue())
        self.assert_no_bundle_modules()

    def test_failed_seed_never_creates_cloud_client_or_resources(self):
        for source in (
            "def evaluate(path): raise RuntimeError('private-detail')\n",
            "def evaluate(path): return {'wrong_objective': 1}\n",
        ):
            (self.bundle / "src/openevolve_evaluator.py").write_text(source)
            with self.sdk(), self.subTest(source=source):
                with self.assertRaises(SystemExit) as caught:
                    self.launch(*self.run_options)
                self.assertEqual(caught.exception.code, 1)
                self.client_type.assert_not_called()
            self.assert_no_bundle_modules()
        self.assertNotIn("private-detail", self.stderr.getvalue())

    def test_bundle_package_does_not_replace_src_or_reuse_an_earlier_bundle(self):
        sentinel = types.ModuleType("src")
        with patch.dict(sys.modules, {"src": sentinel}):
            for score in (7, 9):
                other = self.root / f"bundle-{score}"
                shutil.copytree(self.bundle, other)
                (other / "src/openevolve_evaluator.py").write_text(
                    f"def evaluate(path): return {{'score': {score}}}\n"
                )
                with self.sdk():

                    async def finish(*args, **kwargs):
                        return None

                    self.controller.side_effect = finish
                    self.launch(*self.run_options, bundle=other)
                    self.assertEqual(
                        self.initial["evaluation"]["scores"]["scores"][0]["score"], score
                    )
                self.assertIs(sys.modules["src"], sentinel)
                self.assert_no_bundle_modules()

    def test_session_or_experiment_failure_does_not_upload_seed_or_start_controller(self):
        with self.sdk():
            self.experiment.create_experiment.side_effect = RuntimeError("private-response")
            with self.assertRaises(SystemExit) as caught:
                self.launch(*self.run_options)
            self.assertEqual(caught.exception.code, 1)
            self.experiment.create_initial_program.assert_not_called()
            self.controller.assert_not_called()
        self.assertIn("session/experiment creation", self.stderr.getvalue())
        self.assertNotIn("private-response", self.stderr.getvalue())
        self.assert_no_bundle_modules()

    def test_unconfirmed_seed_upload_does_not_start_generation(self):
        with self.sdk():
            self.experiment.initial_program_name = None
            with self.assertRaises(SystemExit) as caught:
                self.launch(*self.run_options)
            self.assertEqual(caught.exception.code, 1)
            self.client.start_experiment.assert_not_called()
            self.controller.assert_not_called()
        self.assertIn("seed upload", self.stderr.getvalue())
        self.assert_no_bundle_modules()

    def test_unconfirmed_start_does_not_enter_controller(self):
        with self.sdk():
            self.client.start_experiment.return_value = None
            with self.assertRaises(SystemExit) as caught:
                self.launch(*self.run_options)
            self.assertEqual(caught.exception.code, 1)
            self.controller.assert_not_called()
        self.assertIn("experiment start", self.stderr.getvalue())
        self.assert_no_bundle_modules()

    def test_controller_failure_cleans_imports_and_returns_nonzero(self):
        with self.sdk():
            self.controller.side_effect = RuntimeError("controller problem")
            with self.assertRaises(SystemExit) as caught:
                self.launch(*self.run_options)
            self.assertEqual(caught.exception.code, 1)
        self.assertIn("controller execution", self.stderr.getvalue())
        self.assert_no_bundle_modules()

    def test_controller_early_exit_is_reported_as_unmet_target(self):
        with self.sdk():
            self.experiment.stats["num_programs_evaluated"] = 0
            self.assertEqual(self.launch(*self.run_options), 1)
        report = json.loads(self.stdout.getvalue())
        self.assertFalse(report["evaluation_target_reached"])
        self.assertEqual(report["stats"]["num_programs_evaluated"], 0)
        self.assert_no_bundle_modules()


if __name__ == "__main__":
    unittest.main()
