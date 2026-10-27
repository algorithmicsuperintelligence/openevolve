"""Offline tests for OpenEvolve-to-AlphaEvolve evaluator interoperability."""

import copy
import inspect
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cloudpickle

from openevolve.compat.alphaevolve import adapt_evaluator
from openevolve.compat.alphaevolve_reverse import adapt_openevolve_evaluator
from openevolve.evaluation_result import EvaluationResult


def candidate(source="x = 1\n", **file_fields):
    return {"content": {"files": [{"path": "src/main.py", "content": source, **file_fields}]}}


def scores(result):
    return {entry["metric"]: entry["score"] for entry in result["scores"]["scores"]}


class TestReverseAlphaEvolveAdapter(unittest.TestCase):
    def adapt(self, evaluator, **options):
        return adapt_openevolve_evaluator(evaluator, objective_metric="fitness", **options)

    def test_exact_source_private_file_and_cleanup_with_ignored_candidate_path(self):
        paths = []
        source = "# 中文\r\nx = 1\r\n"

        def evaluate(path):
            path = Path(path)
            paths.append(path)
            self.assertEqual(path.name, "program.py")
            self.assertEqual(path.read_bytes(), source.encode("utf-8"))
            return {"fitness": -3, "diversity": 2}

        for name in ("../../outside.py", "/absolute.py", "src/main.py"):
            supplied = candidate(source, path=name)
            original = copy.deepcopy(supplied)
            self.assertEqual(
                scores(self.adapt(evaluate)(supplied)), {"fitness": -3.0, "diversity": 2.0}
            )
            self.assertEqual(supplied, original)
        self.assertEqual(len({path.parent for path in paths}), 3)
        for path in paths:
            self.assertFalse(path.exists())
            self.assertFalse(path.parent.exists())

    def test_sdk_style_candidate_objects(self):
        supplied = SimpleNamespace(
            content=SimpleNamespace(
                files=[SimpleNamespace(content="pass\n", program_language="python")]
            )
        )
        self.assertEqual(
            scores(self.adapt(lambda path: {"fitness": 1})(supplied)), {"fitness": 1.0}
        )

    def test_malformed_or_multifile_candidates_do_not_call_evaluator(self):
        evaluator = Mock()
        for supplied in (
            None,
            {},
            {"content": {}},
            {"content": {"files": []}},
            {"content": {"files": "code"}},
            {"content": {"files": [{"content": "a"}, {"content": "b"}]}},
        ):
            with self.subTest(candidate=supplied), self.assertRaises(ValueError):
                self.adapt(evaluator)(supplied)
        evaluator.assert_not_called()

    def test_non_string_source_and_non_python_language_are_rejected(self):
        for source in (None, b"pass", 10):
            with self.subTest(source=source), self.assertRaises(TypeError):
                self.adapt(lambda path: {"fitness": 1})(candidate(source))
        with self.assertRaisesRegex(ValueError, "Python"):
            self.adapt(lambda path: {"fitness": 1})(candidate(program_language="cpp"))

    def test_empty_source_can_be_scored_by_the_evaluator(self):
        result = self.adapt(lambda path: {"fitness": len(Path(path).read_bytes())})(candidate(""))
        self.assertEqual(scores(result), {"fitness": 0.0})

    def test_numeric_metrics_are_preserved_and_boolean_flags_are_not_objectives(self):
        metrics = {"fitness": -10, "size": 3.5, "passed": True, "skipped": False, "timeout": False}
        original = metrics.copy()
        result = self.adapt(lambda path: metrics)(candidate())
        self.assertEqual(scores(result), {"fitness": -10.0, "size": 3.5})
        self.assertNotIn("insights", result)
        self.assertEqual(metrics, original)
        json.dumps(result, allow_nan=False)

    def test_evaluation_result_warns_without_exporting_text_or_binary_artifacts(self):
        supplied = EvaluationResult(
            {"fitness": 2}, {"private-key": "private-value", "plot": b"png"}
        )
        original = copy.deepcopy(supplied)
        with self.assertWarnsRegex(UserWarning, "artifacts are not exported") as caught:
            result = self.adapt(lambda path: supplied)(candidate())
        self.assertEqual(scores(result), {"fitness": 2.0})
        for secret in ("private-key", "private-value", "plot"):
            self.assertNotIn(secret, str(caught.warning))
            self.assertNotIn(secret, json.dumps(result))
        self.assertEqual(supplied, original)

    def test_explicit_failure_marks_every_available_metric_and_the_objective_failed(self):
        for supplied in (
            {"fitness": 2, "size": 999, "error": 0.0},
            {"size": 999, "timeout": True},
            {"error": "private exception text"},
            EvaluationResult({"fitness": 2, "size": 999}, {"error": "private file path"}),
        ):
            with self.subTest(result=supplied):
                original = copy.deepcopy(supplied)
                result = self.adapt(lambda path: supplied)(candidate())
                self.assertEqual(scores(result)["fitness"], -1e12)
                self.assertTrue(all(value == -1e12 for value in scores(result).values()))
                self.assertNotIn("error", scores(result))
                self.assertNotIn("timeout", scores(result))
                self.assertIn("insights", result)
                self.assertNotIn("private", json.dumps(result))
                self.assertEqual(supplied, original)

    def test_non_finite_auxiliary_or_objective_metrics_fail_all_scores(self):
        for metric in ("fitness", "size"):
            for value in (float("nan"), float("inf"), -float("inf"), 10**1000):
                supplied = {"fitness": -1, "size": 10, metric: value}
                with self.subTest(metric=metric, value=value):
                    result = self.adapt(lambda path: supplied)(candidate())
                    self.assertEqual(scores(result), {"fitness": -1e12, "size": -1e12})
                    json.dumps(result, allow_nan=False)

    def test_evaluator_exception_is_sanitized_and_temp_files_are_cleaned(self):
        paths = []

        def evaluate(path):
            paths.append(Path(path))
            raise RuntimeError("private-credential /private/path")

        result = self.adapt(evaluate)(candidate())
        self.assertEqual(scores(result), {"fitness": -1e12})
        self.assertIn("RuntimeError", json.dumps(result))
        self.assertNotIn("private-credential", json.dumps(result))
        self.assertNotIn(str(paths[0]), json.dumps(result))
        self.assertFalse(paths[0].parent.exists())

    def test_keyboard_interrupt_propagates_and_still_cleans_up(self):
        paths = []

        def evaluate(path):
            paths.append(Path(path))
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.adapt(evaluate)(candidate())
        self.assertFalse(paths[0].parent.exists())

    def test_malformed_result_contracts_raise_after_cleanup(self):
        for supplied in (
            None,
            [],
            {},
            {"other": 1},
            {"fitness": None},
            {"fitness": "1"},
            {"fitness": True},
            {"fitness": 1, "": 2},
            {None: 1},
            EvaluationResult([], {}),
            EvaluationResult({"fitness": 1}, []),
        ):
            paths = []

            def evaluate(path):
                paths.append(Path(path))
                return supplied

            with self.subTest(result=supplied), self.assertRaises((TypeError, ValueError)):
                self.adapt(evaluate)(candidate())
            self.assertFalse(paths[0].parent.exists())

    def test_async_functions_and_async_callables_are_rejected(self):
        async def evaluate(path):
            return {"fitness": 1}

        class AsyncEvaluator:
            async def __call__(self, path):
                return {"fitness": 1}

        for evaluator in (evaluate, AsyncEvaluator()):
            with (
                self.subTest(evaluator=evaluator),
                self.assertRaisesRegex(TypeError, "synchronous"),
            ):
                self.adapt(evaluator)

    def test_unexpected_awaitable_result_is_closed_and_rejected(self):
        async def evaluate():
            return {"fitness": 1}

        coroutine = evaluate()
        with self.assertRaisesRegex(TypeError, "synchronously"):
            self.adapt(lambda path: coroutine)(candidate())
        self.assertEqual(inspect.getcoroutinestate(coroutine), inspect.CORO_CLOSED)

    def test_configuration_validation_and_custom_failure_score(self):
        for objective in (None, "", "  ", "error", "timeout"):
            with self.subTest(objective=objective), self.assertRaises(ValueError):
                adapt_openevolve_evaluator(lambda path: {}, objective_metric=objective)
        for failure in (True, None, "-1", float("nan"), float("inf"), 10**1000):
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                self.adapt(lambda path: {}, failure_score=failure)
        with self.assertRaises(TypeError):
            self.adapt(None)
        result = self.adapt(lambda path: {"error": 0}, failure_score=-1e20)(candidate())
        self.assertEqual(scores(result), {"fitness": -1e20})

    def test_parallel_calls_use_independent_temporary_files(self):
        barrier = threading.Barrier(2)
        paths = []

        def evaluate(path):
            paths.append(Path(path))
            barrier.wait(timeout=5)
            return {"fitness": int(Path(path).read_text())}

        adapter = self.adapt(evaluate)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(adapter, [candidate("1"), candidate("2")]))
        self.assertEqual([scores(result)["fitness"] for result in results], [1.0, 2.0])
        self.assertEqual(len({path.parent for path in paths}), 2)
        self.assertTrue(all(not path.parent.exists() for path in paths))

    def test_callable_serialization_does_not_require_the_google_sdk(self):
        adapter = cloudpickle.loads(cloudpickle.dumps(self.adapt(lambda path: {"fitness": 2})))
        self.assertEqual(scores(adapter(candidate())), {"fitness": 2.0})

    def test_round_trip_through_existing_inbound_adapter(self):
        reverse = self.adapt(lambda path: {"fitness": -3.0, "diversity": 2.0})
        forward = adapt_evaluator(reverse, objective_metric="fitness")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "initial.py"
            path.write_text("pass\n", encoding="utf-8")
            self.assertEqual(
                forward(str(path)).metrics,
                {"fitness": -3.0, "diversity": 2.0, "combined_score": -3.0},
            )


if __name__ == "__main__":
    unittest.main()
