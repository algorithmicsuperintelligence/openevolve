"""Local facade boundaries and real offline OpenEvolve execution."""

import asyncio
import contextlib
import copy
import importlib.util
import io
import os
import secrets
import signal
import sys
import tempfile
import threading
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openevolve.compat.alpha_evolve import (
    AlphaEvolveClient,
    AlphaEvolveExperiment,
    run_controller_loop,
)
from openevolve.compat.alphaevolve import config_from_alphaevolve

EXPERIMENT_MODULE = "openevolve.compat.alpha_evolve.experiment"
SEED = "# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END\n"


def evaluate_candidate(candidate):
    namespace = {}
    exec(candidate["content"]["files"][0]["content"], namespace)
    return {
        "scores": {
            "scores": [
                {"metric": "quality", "score": namespace["x"]},
                {"metric": "cost", "score": -2.0},
            ]
        }
    }


class OfflineModel:
    def __init__(self, config):
        self.model = config.name

    async def generate_with_context(self, system_message, messages, **kwargs):
        return "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE"


def offline_model(config):
    return OfflineModel(config)


class TestLocalAlphaEvolveExperiment(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "title": "Local test",
            "program_language": "python",
            "problem_description": "Maximize x while preserving the interface.",
            "run_settings": {"max_programs": 2},
            "generation_settings": {"models": [{"name": "offline-model"}]},
        }
        self.candidate = {"content": {"files": [{"path": "source.py", "content": SEED}]}}

    def experiment(self, **client_options):
        client = AlphaEvolveClient(
            api_base="http://localhost:8000/v1",
            objective_metric="quality",
            **client_options,
        )
        return AlphaEvolveExperiment(client, evaluate_candidate, 2)

    def ready(self, **client_options):
        experiment = self.experiment(**client_options)
        with self.assertWarnsRegex(UserWarning, "generation-attempt ceiling"):
            experiment.create_experiment(self.payload)
        experiment.create_initial_program(self.candidate)
        return experiment

    def fake_result(self, **overrides):
        values = dict(
            best_program=SimpleNamespace(id="best-id"),
            best_code="x = 2\n",
            metrics={"quality": 2, "cost": -2, "combined_score": 2},
            best_score=2,
        )
        return SimpleNamespace(**(values | overrides))

    def complete(self, experiment=None, result=None):
        experiment = experiment or self.ready()
        with patch(EXPERIMENT_MODULE + ".run_evolution", return_value=result or self.fake_result()):
            experiment.start_experiment()
        return experiment

    def test_setup_has_no_execution_io_or_google_namespace_side_effects(self):
        candidate_before, config_before = copy.deepcopy(self.candidate), copy.deepcopy(self.payload)
        sdk_before = {
            name: value for name, value in sys.modules.items() if name.startswith("alpha_evolve")
        }
        with (
            patch("socket.socket.connect", side_effect=AssertionError("network")),
            patch("tempfile.TemporaryDirectory", side_effect=AssertionError("filesystem")),
            patch(EXPERIMENT_MODULE + ".run_evolution") as evolve,
        ):
            experiment = self.ready()
        evolve.assert_not_called()
        self.assertEqual(experiment.state, "READY")
        self.assertEqual(self.payload, config_before)
        self.assertEqual(self.candidate, candidate_before)
        self.assertEqual(
            {name: value for name, value in sys.modules.items() if name.startswith("alpha_evolve")},
            sdk_before,
        )

    def test_google_arguments_warn_without_leaking_or_becoming_endpoint(self):
        private_value = secrets.token_urlsafe(20)
        with self.assertWarns(UserWarning) as caught:
            experiment = self.experiment(
                project_id=private_value, engine=private_value, base_url=private_value
            )
        self.assertNotIn(private_value, str(caught.warning))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            experiment.create_experiment(self.payload)
        self.assertEqual(experiment._config.llm.api_base, "http://localhost:8000/v1")

    def test_invalid_local_options_are_rejected(self):
        for options in (
            {"api_base": ""},
            {"api_base": None},
            {"objective_metric": ""},
            {"objective_metric": "error"},
            {"iterations": -1},
            {"iterations": True},
            {"iterations": 1.5},
            {"output_dir": 123},
        ):
            with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                AlphaEvolveClient(
                    **({"api_base": "local", "objective_metric": "quality"} | options)
                )

    def test_invalid_experiment_options_fail_before_creation(self):
        client = self.experiment().client

        async def async_evaluator(candidate):
            pass

        class AsyncEvaluator:
            __call__ = staticmethod(async_evaluator)

        for evaluator in (None, async_evaluator, AsyncEvaluator()):
            with self.subTest(evaluator=evaluator), self.assertRaises(TypeError):
                AlphaEvolveExperiment(client, evaluator, 2)
        for count in (0, -1, True, 1.5, None):
            with self.subTest(count=count), self.assertRaises(ValueError):
                AlphaEvolveExperiment(client, evaluate_candidate, count)
        with self.assertRaises(TypeError):
            AlphaEvolveExperiment(object(), evaluate_candidate, 2)
        with self.assertRaisesRegex(ValueError, "parallel_evaluation"):
            AlphaEvolveExperiment(client, evaluate_candidate, 2, parallel_evaluation=True)

    def test_config_validation_can_be_corrected_without_consuming_experiment(self):
        experiment = self.experiment()
        for payload in ({}, dict(self.payload, program_language="cpp")):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                experiment.create_experiment(payload)
            self.assertEqual(experiment.state, "NEW")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            experiment.create_experiment(self.payload)
        self.assertEqual(experiment.state, "CREATED")

    def test_distinct_evaluation_budget_and_increased_override_are_rejected(self):
        experiment = self.experiment()
        self.payload["run_settings"]["max_programs"] = 3
        with self.assertRaisesRegex(ValueError, "distinct generation/evaluation"):
            experiment.create_experiment(self.payload)
        self.payload["run_settings"]["max_programs"] = 2
        with self.assertRaisesRegex(ValueError, "cannot increase"):
            self.experiment(iterations=3).create_experiment(self.payload)

    def test_cloud_concurrency_is_warned_and_not_mapped_to_oe_workers(self):
        self.payload["run_settings"]["concurrency"] = 100
        experiment = self.experiment(iterations=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            experiment.create_experiment(self.payload)
        self.assertTrue(any("run_settings.concurrency" in str(item.message) for item in caught))
        self.assertEqual(experiment._config.evaluator.parallel_evaluations, 1)
        self.assertEqual(experiment._config.max_iterations, 0)

    def test_supplied_seed_scores_are_ignored_and_sources_are_isolated(self):
        experiment = self.experiment()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            experiment.create_experiment(self.payload)
        self.candidate["evaluation"] = {"scores": {"scores": [{"metric": "quality", "score": 999}]}}
        with self.assertWarnsRegex(UserWarning, "re-evaluates"):
            experiment.create_initial_program(self.candidate)
        self.candidate["content"]["files"][0]["content"] = "x = 999\n"
        self.payload["generation_settings"]["models"][0]["name"] = "changed"
        with patch(EXPERIMENT_MODULE + ".run_evolution") as evolve:

            def inspect_seed(path, evaluator, **kwargs):
                self.assertEqual(path.read_bytes(), SEED.encode())
                self.assertEqual(evaluator(str(path)).metrics["quality"], 1)
                self.assertEqual(kwargs["config"].llm.models[0].name, "offline-model")
                return self.fake_result()

            evolve.side_effect = inspect_seed
            experiment.start_experiment()

    def test_malformed_multifile_and_non_python_seeds_are_rejected(self):
        invalid = (
            None,
            {},
            {"content": {"files": []}},
            {"content": {"files": [None]}},
            {"content": {"files": [{}, {}]}},
            {"content": {"files": [{"path": "main.cpp", "content": "x = 1"}]}},
            {"content": {"files": [{"path": "main.py", "content": ""}]}},
            {"content": {"files": [{"path": "main.py", "content": "x ="}]}},
            {
                "content": {
                    "files": [{"path": "main.py", "content": "x = 1", "program_language": "cpp"}]
                }
            },
        )
        experiment = self.experiment()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            experiment.create_experiment(self.payload)
        for candidate in invalid:
            with self.subTest(candidate=candidate), self.assertRaises((ValueError, SyntaxError)):
                experiment.create_initial_program(candidate)
            self.assertEqual(experiment.state, "CREATED")

    def test_seed_path_is_metadata_and_exact_bytes_are_preserved(self):
        source = "# café\r\nx = 1\r\n"
        self.candidate["content"]["files"][0].update(path="../../outside.py", content=source)
        experiment = self.ready()
        captured = []

        def inspect_seed(path, evaluator, **kwargs):
            captured.append(path)
            self.assertEqual(path.name, "program.py")
            self.assertEqual(path.read_bytes(), source.encode("utf-8"))
            return self.fake_result()

        with patch(EXPERIMENT_MODULE + ".run_evolution", side_effect=inspect_seed):
            experiment.start_experiment()
        self.assertFalse(captured[0].exists())
        self.assertEqual(
            experiment.list_programs()["alphaEvolvePrograms"][0]["content"]["files"][0]["path"],
            "../../outside.py",
        )

    def test_lifecycle_guards_prevent_early_or_duplicate_execution(self):
        experiment = self.experiment()
        for action in (
            experiment.start_experiment,
            experiment.list_programs,
            lambda: experiment.create_initial_program(self.candidate),
            lambda: asyncio.run(run_controller_loop(experiment)),
        ):
            with self.assertRaises(RuntimeError):
                action()
        experiment = self.complete()
        for action in (
            experiment.start_experiment,
            lambda: experiment.create_experiment(self.payload),
            lambda: experiment.create_initial_program(self.candidate),
        ):
            with self.assertRaises(RuntimeError):
                action()
        with patch(EXPERIMENT_MODULE + ".run_evolution") as evolve:
            asyncio.run(run_controller_loop(experiment))
            asyncio.run(run_controller_loop(experiment))
        evolve.assert_not_called()

    def test_async_and_worker_thread_start_fail_before_consuming_ready_state(self):
        experiment = self.ready()

        async def start():
            with self.assertRaisesRegex(RuntimeError, "running asyncio"):
                experiment.start_experiment()

        asyncio.run(start())
        errors = []

        def start_in_thread():
            try:
                experiment.start_experiment()
            except RuntimeError as error:
                errors.append(str(error))

        thread = threading.Thread(target=start_in_thread)
        thread.start()
        thread.join(timeout=5)
        self.assertEqual(len(errors), 1)
        self.assertIn("main thread", errors[0])
        self.assertEqual(experiment.state, "READY")
        self.complete(experiment)

    def test_failures_and_interruptions_are_terminal_and_clean_up_seed(self):
        for failure in (ValueError("evaluator failed"), KeyboardInterrupt()):
            experiment = self.ready()
            paths = []

            def fail(path, evaluator, **kwargs):
                paths.append(path)
                raise failure

            with (
                self.subTest(failure=type(failure)),
                patch(EXPERIMENT_MODULE + ".run_evolution", side_effect=fail),
            ):
                with self.assertRaises(type(failure)):
                    experiment.start_experiment()
            self.assertEqual(experiment.state, "FAILED")
            self.assertFalse(paths[0].exists())
            self.assertIsNone(experiment.result)
            with self.assertRaises(RuntimeError):
                experiment.start_experiment()
            with self.assertRaises(RuntimeError):
                asyncio.run(run_controller_loop(experiment))

    def test_missing_best_program_is_failure_not_completion(self):
        experiment = self.ready()
        with self.assertRaisesRegex(RuntimeError, "no best program"):
            self.complete(experiment, self.fake_result(best_program=None))
        self.assertEqual(experiment.state, "FAILED")

    def test_best_only_envelope_preserves_negative_scores_and_does_not_alias_metrics(self):
        experiment = self.complete(
            result=self.fake_result(metrics={"quality": -3, "cost": -2, "combined_score": -3})
        )
        response = experiment.list_programs({"order_by": "quality desc"})
        self.assertEqual(len(response["alphaEvolvePrograms"]), 1)
        program = response["alphaEvolvePrograms"][0]
        self.assertEqual(
            program["evaluation"]["scores"]["scores"],
            [{"metric": "quality", "score": -3.0}, {"metric": "cost", "score": -2.0}],
        )
        program["evaluation"]["scores"]["scores"][0]["score"] = 10
        self.assertEqual(experiment.result.metrics["quality"], -3)
        for params in (
            1,
            {"page_size": 3},
            {"order_by": "unknown desc"},
            {"order_by": "quality"},
            {"order_by": None},
            {"order_by": "combined_score desc"},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                experiment.list_programs(params)

    def test_failed_evaluation_result_retains_failure_insight(self):
        experiment = self.complete(
            result=self.fake_result(
                metrics={"quality": -1e12, "combined_score": -1e12, "error": 0.0}
            )
        )
        evaluation = experiment.list_programs()["alphaEvolvePrograms"][0]["evaluation"]
        self.assertEqual(evaluation["scores"]["scores"], [{"metric": "quality", "score": -1e12}])
        self.assertIn("Error", evaluation["insights"]["insights"][0]["label"])

    def test_controller_rejects_other_sdk_and_unimplemented_cloud_options(self):
        with self.assertRaises(TypeError):
            asyncio.run(run_controller_loop(object()))
        with self.assertRaises(TypeError):
            run_controller_loop(self.complete(), idle_timeout_s=10)

    def test_controller_accepts_counts_without_rerunning_or_adding_workers(self):
        experiment = self.complete()
        with (
            self.assertWarnsRegex(UserWarning, "ignores num_samplers/num_evaluators"),
            patch(EXPERIMENT_MODULE + ".run_evolution") as evolve,
        ):
            asyncio.run(run_controller_loop(experiment, num_samplers=2, num_evaluators=4))
        evolve.assert_not_called()
        self.assertEqual(experiment.state, "COMPLETED")
        for key in ("num_samplers", "num_evaluators"):
            for value in (0, -1, True, 1.5, "1"):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    asyncio.run(run_controller_loop(experiment, **{key: value}))
        with self.assertRaises(RuntimeError):
            asyncio.run(run_controller_loop(self.ready(), num_evaluators=1))

    def test_generic_score_ordering_uses_selected_objective_and_preserves_metrics(self):
        experiment = self.complete(result=self.fake_result(metrics={"quality": -3, "cost": -2}))
        expected = experiment.list_programs({"order_by": "quality desc"})
        for direction in ("asc", "desc"):
            self.assertEqual(experiment.list_programs({"order_by": "score " + direction}), expected)
        missing = self.complete(result=self.fake_result(metrics={"cost": -2}))
        with self.assertRaises(ValueError):
            missing.list_programs({"order_by": "score desc"})
        with self.assertRaises(ValueError):
            experiment.list_programs({"order_by": "score sideways"})

    def test_local_query_tracks_lifecycle_without_io_and_returns_independent_metadata(self):
        experiment = self.ready(api_key="private-selected-key")
        client, name = experiment.client, experiment.experiment_name
        with (
            patch("socket.socket.connect", side_effect=AssertionError("network")),
            patch("builtins.open", side_effect=AssertionError("filesystem")),
        ):
            self.assertEqual(
                client.get_alpha_evolve_experiment(experiment_name=name),
                {"name": name, "state": "READY"},
            )
            response = client.get_alpha_evolve_experiment(name)
            response["state"] = "WRONG"
            self.assertEqual(client.get_alpha_evolve_experiment(name)["state"], "READY")
            self.complete(experiment)
            self.assertEqual(client.get_alpha_evolve_experiment(name)["state"], "COMPLETED")
            for invalid in (None, [], "unknown", self.ready().experiment_name):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    client.get_alpha_evolve_experiment(invalid)

    def test_local_query_keeps_created_probes_separate_from_failed_runs(self):
        probe = self.experiment()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            probe.create_experiment(self.payload)
        client = probe.client
        running = AlphaEvolveExperiment(client, evaluate_candidate, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            running.create_experiment(self.payload)
        running.create_initial_program(self.candidate)
        with patch(EXPERIMENT_MODULE + ".run_evolution", side_effect=RuntimeError("run failed")):
            with self.assertRaisesRegex(RuntimeError, "run failed"):
                running.start_experiment()
        self.assertEqual(
            client.get_alpha_evolve_experiment(probe.experiment_name)["state"], "CREATED"
        )
        self.assertEqual(
            client.get_alpha_evolve_experiment(running.experiment_name)["state"], "FAILED"
        )

    def test_documented_seed_example_works_with_real_provider_client_and_no_ambient_key(self):
        readme = Path(__file__).resolve().parents[1] / "examples/alphaevolve_interop/README.md"
        source = readme.read_text().split("```python\n", 1)[1].split("```", 1)[0]
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            with (
                patch.dict(os.environ, {}, clear=True),
                patch("socket.socket.connect", side_effect=AssertionError("network")),
                contextlib.redirect_stdout(io.StringIO()),
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore", UserWarning)
                namespace = {}
                exec(compile(source, str(readme), "exec"), namespace)
            self.assertEqual(namespace["experiment"].result.metrics["quality"], 1)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)

    def test_real_seed_only_and_worker_generation_with_file_loaded_evaluator(self):
        # File-loaded evaluators must remain callable in OE's subprocess.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module_path = root / "local_candidate_evaluator.py"
            module_path.write_text(
                "def evaluate(candidate):\n"
                "    namespace = {}\n"
                "    exec(candidate['content']['files'][0]['content'], namespace)\n"
                "    return {'scores': {'scores': [{'metric': 'quality', 'score': namespace['x']}]}}\n"
            )
            spec = importlib.util.spec_from_file_location("local_candidate_evaluator", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # Cloudpickle's normal by-reference mode requires an importable module
            # on the worker path. Here, use its by-value mode as the example runner does.
            old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
            self.addCleanup(
                lambda: [signal.signal(sig, value) for sig, value in old_handlers.items()]
            )

            def offline_config(*args, **kwargs):
                config = config_from_alphaevolve(*args, **kwargs)
                config.database.num_islands = 1
                for model in config.llm.models + config.llm.evaluator_models:
                    model.init_client = offline_model
                return config

            for iterations, expected in ((0, 1), (1, 2)):
                client = AlphaEvolveClient(
                    api_base="http://localhost:8000/v1",
                    objective_metric="quality",
                    iterations=iterations,
                    output_dir=root / f"run-{iterations}",
                )
                experiment = AlphaEvolveExperiment(client, module.evaluate, 2)
                with (
                    patch(
                        EXPERIMENT_MODULE + ".config_from_alphaevolve", side_effect=offline_config
                    ),
                    warnings.catch_warnings(),
                ):
                    warnings.simplefilter("ignore", UserWarning)
                    experiment.create_experiment(self.payload)
                experiment.create_initial_program(self.candidate)
                with patch("socket.socket.connect", side_effect=AssertionError("network")):
                    experiment.start_experiment()
                    asyncio.run(run_controller_loop(experiment))
                self.assertEqual(experiment.result.metrics["quality"], expected)
                self.assertEqual(experiment.state, "COMPLETED")
                self.assertTrue((root / f"run-{iterations}" / "best" / "best_program.py").exists())


if __name__ == "__main__":
    unittest.main()
