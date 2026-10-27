"""Scoped local imports and callable transport without the Google SDK."""

import importlib
import json
import math
import os
import pickle
import py_compile
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import cloudpickle

from openevolve.compat.alpha_evolve import (
    AlphaEvolveClient,
    AlphaEvolveExperiment,
    load_evaluator,
    local_imports,
    run_controller_loop,
)
from openevolve.compat.alpha_evolve.models import (
    AlphaEvolveEvaluationInsight,
    AlphaEvolveEvaluationInsights,
    AlphaEvolveEvaluationScore,
    AlphaEvolveEvaluationScores,
    AlphaEvolveProgramEvaluation,
)
from openevolve.compat.alpha_evolve.visualization import get_score
from openevolve.compat.alphaevolve import adapt_evaluator


def sdk_namespace():
    return {
        name: value
        for name, value in sys.modules.items()
        if name == "alpha_evolve" or name.startswith("alpha_evolve.")
    }


EVALUATOR = """from __future__ import annotations
from dataclasses import dataclass
from alpha_evolve.models import (
    AlphaEvolveEvaluationScore, AlphaEvolveEvaluationScores,
    AlphaEvolveEvaluationInsight, AlphaEvolveEvaluationInsights,
    AlphaEvolveProgramEvaluation,
)

@dataclass
class EvaluationInput:
    score: float

def evaluate(candidate):
    namespace = {}
    exec(candidate["content"]["files"][0]["content"], namespace)
    return AlphaEvolveProgramEvaluation(
        scores=AlphaEvolveEvaluationScores(scores=[
            AlphaEvolveEvaluationScore(metric="quality", score=EvaluationInput(namespace["x"]).score)
        ]),
        insights=AlphaEvolveEvaluationInsights(insights=[
            AlphaEvolveEvaluationInsight(label="Hint", text="中文 explanation")
        ]),
    )
"""


class TestLocalAlphaEvolveModels(unittest.TestCase):
    def test_nested_dump_preserves_scores_insights_and_is_independent(self):
        model = AlphaEvolveProgramEvaluation(
            scores=AlphaEvolveEvaluationScores(
                scores=[AlphaEvolveEvaluationScore(metric="quality", score=-4.5)]
            ),
            insights=AlphaEvolveEvaluationInsights(
                insights=[AlphaEvolveEvaluationInsight(label="Hint", text="中文")]
            ),
        )
        dumped = model.model_dump()
        self.assertEqual(
            dumped,
            {
                "scores": {"scores": [{"metric": "quality", "score": -4.5}]},
                "insights": {"insights": [{"label": "Hint", "text": "中文"}]},
            },
        )
        dumped["scores"]["scores"][0]["score"] = 999
        self.assertEqual(model.scores.scores[0].score, -4.5)
        self.assertIn("中文", json.dumps(model.model_dump(), ensure_ascii=False))

    def test_null_defaults_match_evaluator_failure_envelope(self):
        evaluation = AlphaEvolveProgramEvaluation(
            scores=AlphaEvolveEvaluationScores(
                scores=[AlphaEvolveEvaluationScore(metric="quality")]
            )
        )
        self.assertEqual(
            evaluation.model_dump(),
            {
                "scores": {"scores": [{"metric": "quality", "score": None}]},
                "insights": None,
            },
        )

    def test_model_or_dump_are_validated_by_existing_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            program = Path(directory) / "program.py"
            program.write_text("x = 1\n")
            for value in (-3.0, None, -1e12, math.nan, math.inf):
                model = AlphaEvolveProgramEvaluation(
                    scores=AlphaEvolveEvaluationScores(
                        scores=[AlphaEvolveEvaluationScore(metric="quality", score=value)]
                    )
                )
                for result in (model, model.model_dump()):
                    with self.subTest(value=value, result_type=type(result)):
                        adapted = adapt_evaluator(
                            lambda candidate: result, objective_metric="quality"
                        )
                        converted = adapted(program)
                        expected = -3.0 if value == -3.0 else -1e12
                        self.assertEqual(converted.metrics["combined_score"], expected)
                        self.assertEqual("error" in converted.metrics, expected != -3.0)

    def test_dataclasses_do_not_silently_coerce_invalid_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            program = Path(directory) / "program.py"
            program.write_text("x = 1\n")
            for score in ("2", True):
                model = AlphaEvolveProgramEvaluation(
                    scores=AlphaEvolveEvaluationScores(
                        scores=[AlphaEvolveEvaluationScore(metric="quality", score=score)]
                    )
                )
                with self.subTest(score=score), self.assertRaises(TypeError):
                    adapt_evaluator(lambda candidate: model, objective_metric="quality")(program)

    def test_score_lookup_accepts_dicts_and_attribute_envelopes(self):
        evaluation = AlphaEvolveProgramEvaluation(
            scores=AlphaEvolveEvaluationScores(
                scores=[
                    AlphaEvolveEvaluationScore(metric="quality", score=-2.5),
                    AlphaEvolveEvaluationScore(metric="zero", score=0),
                ]
            )
        )
        for program in (
            {"evaluation": evaluation.model_dump()},
            SimpleNamespace(evaluation=evaluation),
        ):
            with self.subTest(program_type=type(program)):
                self.assertEqual(get_score(program, "quality"), -2.5)
                self.assertEqual(get_score(program, "zero"), 0)
                self.assertEqual(get_score(program, "absent"), -math.inf)

    def test_score_lookup_treats_malformed_and_nonfinite_values_as_missing(self):
        for program in (None, {}, {"evaluation": None}, {"evaluation": {"scores": 123}}):
            with self.subTest(program=program):
                self.assertEqual(get_score(program, "quality"), -math.inf)
        for value in (None, "bad", {}, True, math.nan, math.inf, -math.inf):
            program = {
                "evaluation": {"scores": {"scores": [{"metric": "quality", "score": value}]}}
            }
            with self.subTest(value=value):
                self.assertEqual(get_score(program, "quality"), -math.inf)


class TestLocalAlphaEvolveImports(unittest.TestCase):
    def setUp(self):
        self.before = sdk_namespace()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.evaluator_file = self.root / "evaluate.py"
        self.evaluator_file.write_text(EVALUATOR, encoding="utf-8")

    def tearDown(self):
        self.assertEqual(sdk_namespace(), self.before)
        self.assertFalse(
            any(name.startswith("_openevolve_local_ae_evaluator_") for name in sys.modules)
        )

    def test_all_supported_aliases_resolve_only_inside_context(self):
        with patch("socket.socket.connect", side_effect=AssertionError("network")):
            with local_imports():
                package = importlib.import_module("alpha_evolve")
                client = importlib.import_module("alpha_evolve.client")
                experiment = importlib.import_module("alpha_evolve.experiment")
                controller = importlib.import_module("alpha_evolve.controller")
                models = importlib.import_module("alpha_evolve.models")
                visualization = importlib.import_module("alpha_evolve.visualization")
                self.assertIs(client.AlphaEvolveClient, AlphaEvolveClient)
                self.assertIs(experiment.AlphaEvolveExperiment, AlphaEvolveExperiment)
                self.assertIs(controller.run_controller_loop, run_controller_loop)
                self.assertIs(models.AlphaEvolveEvaluationScore, AlphaEvolveEvaluationScore)
                self.assertIs(visualization.get_score, get_score)
                self.assertIs(package.models, models)
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module("alpha_evolve.cloud_only")
        self.assertEqual(sdk_namespace(), self.before)

    def test_exception_and_interrupt_remove_owned_aliases(self):
        for exception in (RuntimeError("import failed"), KeyboardInterrupt()):
            with self.subTest(exception=type(exception)), self.assertRaises(type(exception)):
                with local_imports():
                    raise exception
            self.assertEqual(sdk_namespace(), self.before)

    def test_preexisting_root_submodule_or_none_placeholder_is_never_replaced(self):
        for name, value in (
            ("alpha_evolve", ModuleType("alpha_evolve")),
            ("alpha_evolve.models", ModuleType("alpha_evolve.models")),
            ("alpha_evolve", None),
        ):
            with self.subTest(name=name, value=value), patch.dict(sys.modules, {name: value}):
                with self.assertRaisesRegex(RuntimeError, "already loaded"):
                    with local_imports():
                        self.fail("must not enter")
                self.assertIs(sys.modules[name], value)

    def test_nested_context_rejection_preserves_outer_aliases(self):
        with local_imports():
            outer = sdk_namespace()
            with self.assertRaisesRegex(RuntimeError, "already loaded"):
                with local_imports():
                    self.fail("must not enter")
            self.assertEqual(sdk_namespace(), outer)

    def test_context_is_repeatable_after_cleanup(self):
        for _ in range(2):
            with local_imports():
                self.assertIs(
                    importlib.import_module("alpha_evolve.models").AlphaEvolveProgramEvaluation,
                    AlphaEvolveProgramEvaluation,
                )
            self.assertEqual(sdk_namespace(), self.before)

    def test_cleanup_does_not_remove_replacements_owned_by_other_code(self):
        replacement = ModuleType("replacement")
        try:
            with local_imports():
                sys.modules["alpha_evolve.models"] = replacement
            self.assertIs(sys.modules["alpha_evolve.models"], replacement)
            self.assertNotIn("alpha_evolve", sys.modules)
        finally:
            if sys.modules.get("alpha_evolve.models") is replacement:
                del sys.modules["alpha_evolve.models"]

    def test_worker_thread_rejected_before_import_side_effects(self):
        errors = []

        def enter():
            try:
                with local_imports():
                    self.fail("must not enter")
            except RuntimeError as error:
                errors.append(str(error))

        worker = threading.Thread(target=enter)
        worker.start()
        worker.join(timeout=5)
        self.assertEqual(len(errors), 1)
        self.assertIn("main thread", errors[0])

    def test_file_loader_handles_dataclass_decorators_and_removes_import_names(self):
        with patch("socket.socket.connect", side_effect=AssertionError("network")):
            evaluator = load_evaluator(self.evaluator_file, "evaluate")
        self.assertNotIn(evaluator.__module__, sys.modules)
        result = evaluator({"content": {"files": [{"content": "x = -3.0\n"}]}})
        self.assertIsInstance(result, AlphaEvolveProgramEvaluation)
        self.assertEqual(result.scores.scores[0].score, -3.0)
        self.assertEqual(result.insights.insights[0].text, "中文 explanation")

    def test_file_loader_cleans_up_import_failures_and_missing_callables(self):
        for source, name, exception in (
            ("raise RuntimeError('failed import')", "evaluate", RuntimeError),
            ("raise KeyboardInterrupt()", "evaluate", KeyboardInterrupt),
            ("evaluate = 123", "evaluate", ValueError),
            ("def evaluate(candidate): pass", "missing", ValueError),
            ("from .sibling import evaluate", "evaluate", ImportError),
        ):
            self.evaluator_file.write_text(source)
            with self.subTest(source=source), self.assertRaises(exception):
                load_evaluator(self.evaluator_file, name)
            self.assertEqual(sdk_namespace(), self.before)

    def test_file_loader_uses_current_source_even_with_same_timestamp_size_cached_bytecode(self):
        self.evaluator_file.write_text("def evaluate(candidate): return 1\n")
        py_compile.compile(str(self.evaluator_file), doraise=True)
        before = self.evaluator_file.stat()
        self.assertEqual(load_evaluator(self.evaluator_file, "evaluate")({}), 1)
        self.evaluator_file.write_text("def evaluate(candidate): return 2\n")
        os.utime(self.evaluator_file, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertEqual(self.evaluator_file.stat().st_size, before.st_size)
        self.assertEqual(load_evaluator(self.evaluator_file, "evaluate")({}), 2)

    def test_file_loader_rejects_bad_options_before_executing_file(self):
        self.evaluator_file.write_text("raise AssertionError('must not execute')")
        other = self.root / "evaluate.txt"
        other.write_text(EVALUATOR)
        for path, name in (
            (self.root / "missing.py", "evaluate"),
            (other, "evaluate"),
            (self.evaluator_file, ""),
            (self.evaluator_file, "bad.name"),
            (self.evaluator_file, None),
        ):
            with self.subTest(path=path, name=name), self.assertRaises(ValueError):
                load_evaluator(path, name)

    def test_file_loader_does_not_overwrite_already_imported_cloud_sdk(self):
        sdk = ModuleType("alpha_evolve")
        self.evaluator_file.write_text("raise AssertionError('must not execute')")
        with patch.dict(sys.modules, {"alpha_evolve": sdk}):
            with self.assertRaisesRegex(RuntimeError, "already loaded"):
                load_evaluator(self.evaluator_file, "evaluate")
            self.assertIs(sys.modules["alpha_evolve"], sdk)

    def test_loaded_callable_survives_standard_pickle_load_in_fresh_worker_without_sdk(self):
        evaluator = load_evaluator(self.evaluator_file, "evaluate")
        adapted = adapt_evaluator(evaluator, objective_metric="quality")
        payload = self.root / "evaluator.pkl"
        payload.write_bytes(cloudpickle.dumps(adapted))
        program = self.root / "program.py"
        program.write_text("x = -7.5\n")
        worker = """import importlib.abc, json, pickle, socket, sys
class NoCloudSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "alpha_evolve" or fullname.startswith("alpha_evolve."):
            raise AssertionError("worker must not import cloud SDK or temporary aliases")
sys.meta_path.insert(0, NoCloudSDK())
def no_network(*args, **kwargs):
    raise AssertionError("worker must not connect")
socket.socket.connect = no_network
with open(sys.argv[1], "rb") as stream:
    evaluator = pickle.load(stream)
result = evaluator(sys.argv[2])
assert not any(n == "alpha_evolve" or n.startswith("alpha_evolve.") for n in sys.modules)
print(json.dumps({"metrics": result.metrics, "artifacts": result.artifacts}))
"""
        process = subprocess.run(
            [sys.executable, "-c", worker, str(payload), str(program)],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual(result["metrics"], {"quality": -7.5, "combined_score": -7.5})
        self.assertEqual(
            json.loads(result["artifacts"]["alphaevolve_insights"]),
            [{"label": "Hint", "text": "中文 explanation"}],
        )
        # Standard pickle can also load cloudpickle's by-value callable locally.
        self.assertEqual(pickle.loads(payload.read_bytes())(program).metrics["quality"], -7.5)


if __name__ == "__main__":
    unittest.main()
