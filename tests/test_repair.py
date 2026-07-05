"""
Tests for the LLM-based code repair feature.

Covers:
  - EvaluatorRepairRequest exception construction
  - Evaluator._pending_repairs / get_pending_repair()
  - evaluate_program() behaviour when repair is disabled / enabled
  - _attempt_repair() success and failure paths
  - repair_history propagation into pending_artifacts
  - EvolutionTracer repair statistics
  - iteration.py repair metadata interception logic
"""

import asyncio
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from openevolve.config import EvaluatorConfig
from openevolve.evaluation_result import EvaluationResult, EvaluatorRepairRequest
from openevolve.evaluator import Evaluator
from openevolve.evolution_trace import EvolutionTrace, EvolutionTracer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_eval_file(body: str) -> str:
    """Write a Python evaluator snippet to a temp file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    # Injecting the openevolve path so the eval file can import EvaluatorRepairRequest
    f.write("import sys, os\n")
    f.write("sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n")
    f.write(body)
    f.close()
    return f.name


def _make_evaluator(config: EvaluatorConfig, eval_file: str, **kwargs) -> Evaluator:
    """Create an Evaluator without LLM ensemble or prompt sampler (unless supplied)."""
    return Evaluator(
        config=config,
        evaluation_file=eval_file,
        llm_ensemble=kwargs.get("llm_ensemble"),
        prompt_sampler=kwargs.get("prompt_sampler"),
        suffix=".py",
    )


# ---------------------------------------------------------------------------
# EvaluatorRepairRequest
# ---------------------------------------------------------------------------

class TestEvaluatorRepairRequest(unittest.TestCase):
    """Basic construction and attribute tests."""

    def test_message_is_str(self):
        req = EvaluatorRepairRequest("compile error", "int x = ]")
        self.assertEqual(str(req), "compile error")

    def test_all_fields_explicit(self):
        req = EvaluatorRepairRequest(
            message="bad code",
            broken_code="int x = ]",
            repair_context="full compiler output",
            language="cpp",
        )
        self.assertEqual(req.broken_code, "int x = ]")
        self.assertEqual(req.repair_context, "full compiler output")
        self.assertEqual(req.language, "cpp")

    def test_repair_context_defaults_to_message(self):
        req = EvaluatorRepairRequest("error msg", "broken")
        self.assertEqual(req.repair_context, "error msg")

    def test_language_defaults_to_python(self):
        req = EvaluatorRepairRequest("err", "code")
        self.assertEqual(req.language, "python")

    def test_is_exception(self):
        req = EvaluatorRepairRequest("err", "code")
        self.assertIsInstance(req, Exception)


# ---------------------------------------------------------------------------
# get_pending_repair
# ---------------------------------------------------------------------------

class TestGetPendingRepair(unittest.TestCase):
    """Tests for the _pending_repairs side-channel."""

    def setUp(self):
        eval_src = "def evaluate(path): return {'score': 1.0}\n"
        self.eval_file = _make_eval_file(eval_src)
        self.evaluator = _make_evaluator(EvaluatorConfig(), self.eval_file)

    def tearDown(self):
        os.unlink(self.eval_file)

    def test_returns_none_when_absent(self):
        self.assertIsNone(self.evaluator.get_pending_repair("no-such-id"))

    def test_returns_code_and_clears(self):
        self.evaluator._pending_repairs["prog-1"] = "repaired source"
        result = self.evaluator.get_pending_repair("prog-1")
        self.assertEqual(result, "repaired source")
        # Second call must return None (one-shot)
        self.assertIsNone(self.evaluator.get_pending_repair("prog-1"))

    def test_independent_ids(self):
        self.evaluator._pending_repairs["a"] = "code_a"
        self.evaluator._pending_repairs["b"] = "code_b"
        self.assertEqual(self.evaluator.get_pending_repair("a"), "code_a")
        self.assertEqual(self.evaluator.get_pending_repair("b"), "code_b")


# ---------------------------------------------------------------------------
# evaluate_program with repair
# ---------------------------------------------------------------------------

class TestEvaluatorRepairFlow(unittest.TestCase):
    """
    Tests evaluate_program() when the evaluation function raises
    EvaluatorRepairRequest. Uses patch.object to control _direct_evaluate
    and _repair_code so no real LLM calls or file compilation occurs.
    """

    def setUp(self):
        eval_src = "def evaluate(path): return {'combined_score': 0.9}\n"
        self.eval_file = _make_eval_file(eval_src)

    def tearDown(self):
        os.unlink(self.eval_file)

    def _run(self, coro):
        return asyncio.run(coro)

    # ------------------------------------------------------------------
    # repair disabled
    # ------------------------------------------------------------------

    def test_repair_disabled_returns_zero_score(self):
        """When repair_on_failure=False, a repair request yields score 0."""
        config = EvaluatorConfig(repair_on_failure=False, cascade_evaluation=False)
        ev = _make_evaluator(config, self.eval_file)

        repair_req = EvaluatorRepairRequest("compile error", "broken cpp", language="cpp")
        success_result = EvaluationResult(metrics={"combined_score": 0.9})

        with patch.object(ev, "_direct_evaluate", new=AsyncMock(
            side_effect=[repair_req, success_result]
        )):
            metrics = self._run(ev.evaluate_program("broken code", "prog-1"))

        self.assertEqual(metrics.get("combined_score"), 0.0)
        self.assertIsNone(ev.get_pending_repair("prog-1"))

    def test_repair_disabled_stores_compile_error_artifact(self):
        config = EvaluatorConfig(repair_on_failure=False, cascade_evaluation=False)
        ev = _make_evaluator(config, self.eval_file)

        repair_req = EvaluatorRepairRequest("compile error", "bad code", language="cpp")
        with patch.object(ev, "_direct_evaluate", new=AsyncMock(side_effect=repair_req)):
            self._run(ev.evaluate_program("bad code", "prog-2"))

        artifacts = ev.get_pending_artifacts("prog-2")
        self.assertIsNotNone(artifacts)
        self.assertIn("compile_error", artifacts)

    # ------------------------------------------------------------------
    # repair enabled, succeeds
    # ------------------------------------------------------------------

    def test_repair_succeeds_first_attempt(self):
        """Repair succeeds on the first LLM attempt → real metrics returned."""
        config = EvaluatorConfig(
            repair_on_failure=True,
            max_repair_attempts=2,
            repair_diff_based=False,
            cascade_evaluation=False,
        )
        mock_llm = MagicMock()
        ev = _make_evaluator(config, self.eval_file, llm_ensemble=mock_llm)

        repair_req = EvaluatorRepairRequest("compile error", "broken", language="cpp")
        success = EvaluationResult(metrics={"combined_score": 0.85})

        # First _direct_evaluate raises repair request; second returns success.
        with patch.object(ev, "_direct_evaluate", new=AsyncMock(
            side_effect=[repair_req, success]
        )):
            with patch.object(ev, "_repair_code", new=AsyncMock(return_value="fixed code")):
                metrics = self._run(ev.evaluate_program("broken", "prog-3"))

        self.assertAlmostEqual(metrics.get("combined_score"), 0.85)
        self.assertEqual(ev.get_pending_repair("prog-3"), "fixed code")

    def test_repair_stores_history_in_artifacts(self):
        config = EvaluatorConfig(
            repair_on_failure=True,
            max_repair_attempts=2,
            cascade_evaluation=False,
        )
        mock_llm = MagicMock()
        ev = _make_evaluator(config, self.eval_file, llm_ensemble=mock_llm)

        repair_req = EvaluatorRepairRequest("oops", "broken", language="cpp")
        success = EvaluationResult(metrics={"combined_score": 0.7})

        with patch.object(ev, "_direct_evaluate", new=AsyncMock(
            side_effect=[repair_req, success]
        )):
            with patch.object(ev, "_repair_code", new=AsyncMock(return_value="fixed")):
                self._run(ev.evaluate_program("broken", "prog-4"))

        artifacts = ev.get_pending_artifacts("prog-4")
        self.assertIsNotNone(artifacts)
        history = artifacts.get("repair_history", [])
        self.assertEqual(len(history), 1)
        self.assertTrue(history[0]["succeeded"])

    def test_repair_succeeds_second_attempt(self):
        """First repair attempt also fails; second succeeds."""
        config = EvaluatorConfig(
            repair_on_failure=True,
            max_repair_attempts=2,
            cascade_evaluation=False,
        )
        mock_llm = MagicMock()
        ev = _make_evaluator(config, self.eval_file, llm_ensemble=mock_llm)

        repair_req1 = EvaluatorRepairRequest("err1", "broken1", language="cpp")
        repair_req2 = EvaluatorRepairRequest("err2", "broken2", language="cpp")
        success = EvaluationResult(metrics={"combined_score": 0.6})

        with patch.object(ev, "_direct_evaluate", new=AsyncMock(
            side_effect=[repair_req1, repair_req2, success]
        )):
            # _repair_code returns a (different) fix each call
            with patch.object(ev, "_repair_code", new=AsyncMock(
                side_effect=["fix1", "fix2"]
            )):
                metrics = self._run(ev.evaluate_program("broken1", "prog-5"))

        self.assertAlmostEqual(metrics.get("combined_score"), 0.6)
        self.assertEqual(ev.get_pending_repair("prog-5"), "fix2")

        artifacts = ev.get_pending_artifacts("prog-5")
        history = artifacts.get("repair_history", [])
        self.assertEqual(len(history), 2)
        self.assertFalse(history[0]["succeeded"])
        self.assertTrue(history[1]["succeeded"])

    # ------------------------------------------------------------------
    # repair enabled, all attempts fail
    # ------------------------------------------------------------------

    def test_repair_all_attempts_fail_returns_zero(self):
        config = EvaluatorConfig(
            repair_on_failure=True,
            max_repair_attempts=2,
            cascade_evaluation=False,
        )
        mock_llm = MagicMock()
        ev = _make_evaluator(config, self.eval_file, llm_ensemble=mock_llm)

        repair_req = EvaluatorRepairRequest("err", "broken", language="cpp")

        with patch.object(ev, "_direct_evaluate", new=AsyncMock(side_effect=repair_req)):
            # _repair_code always returns "fixed" but re-evaluation always fails
            with patch.object(ev, "_repair_code", new=AsyncMock(return_value="fixed")):
                metrics = self._run(ev.evaluate_program("broken", "prog-6"))

        # All attempts failed → score 0
        self.assertEqual(metrics.get("combined_score"), 0.0)
        # No pending repair stored
        self.assertIsNone(ev.get_pending_repair("prog-6"))
        # repair_failed flag set in artifacts
        artifacts = ev.get_pending_artifacts("prog-6")
        self.assertTrue(artifacts.get("repair_failed"))

    def test_repair_code_returns_none_aborts(self):
        """If _repair_code can't parse a fix, repair aborts cleanly."""
        config = EvaluatorConfig(
            repair_on_failure=True,
            max_repair_attempts=3,
            cascade_evaluation=False,
        )
        mock_llm = MagicMock()
        ev = _make_evaluator(config, self.eval_file, llm_ensemble=mock_llm)

        repair_req = EvaluatorRepairRequest("err", "broken", language="cpp")
        with patch.object(ev, "_direct_evaluate", new=AsyncMock(side_effect=repair_req)):
            with patch.object(ev, "_repair_code", new=AsyncMock(return_value=None)):
                metrics = self._run(ev.evaluate_program("broken", "prog-7"))

        self.assertEqual(metrics.get("combined_score"), 0.0)
        self.assertIsNone(ev.get_pending_repair("prog-7"))


# ---------------------------------------------------------------------------
# _repair_code template safety (brace escaping)
# ---------------------------------------------------------------------------

class TestRepairCodeTemplateSafety(unittest.TestCase):
    """
    _repair_code must not raise KeyError when broken_code contains C++ braces.
    We test the safe substitution logic by supplying a minimal mock template
    directly in the template manager.
    """

    def setUp(self):
        eval_src = "def evaluate(path): return {'score': 1.0}\n"
        self.eval_file = _make_eval_file(eval_src)

    def tearDown(self):
        os.unlink(self.eval_file)

    def _run(self, coro):
        return asyncio.run(coro)

    def test_cpp_braces_in_broken_code_do_not_crash(self):
        """Code like 'namespace foo { {} }' must not crash the template formatter."""
        config = EvaluatorConfig(repair_on_failure=True, max_repair_attempts=1)
        mock_llm = AsyncMock()
        # Return a valid code-fenced response so parse_full_rewrite succeeds
        mock_llm.generate_with_context = AsyncMock(
            return_value="```cpp\nint main(){}\n```"
        )

        def _get_template(name):
            if name == "repair_full_rewrite_user":
                return "fix: {broken_code}"
            if name == "system_message":
                return "You are an expert programmer."
            raise ValueError(f"Template '{name}' not found")

        mock_sampler = MagicMock()
        mock_sampler.template_manager.get_template.side_effect = _get_template

        ev = _make_evaluator(
            config, self.eval_file,
            llm_ensemble=mock_llm,
            prompt_sampler=mock_sampler,
        )

        # C++ code with many braces
        cpp_with_braces = "namespace ns { struct S { void f(){} }; }"
        result = self._run(ev._repair_code(
            broken_code=cpp_with_braces,
            error_message="compile error",
            repair_context="full output",
            language="cpp",
        ))
        # Should return the parsed code, not crash
        self.assertEqual(result, "int main(){}")


# ---------------------------------------------------------------------------
# EvolutionTracer repair stats
# ---------------------------------------------------------------------------

class TestEvolutionTracerRepairStats(unittest.TestCase):
    """Tests for repair_triggered / repair_succeeded counters."""

    def _make_tracer(self) -> EvolutionTracer:
        return EvolutionTracer(enabled=False)

    def _make_trace(self, repair_history=None) -> EvolutionTrace:
        meta = {}
        if repair_history is not None:
            meta["repair_history"] = repair_history
        return EvolutionTrace(
            iteration=1,
            timestamp=0.0,
            parent_id="p1",
            child_id="c1",
            parent_metrics={"combined_score": 0.5},
            child_metrics={"combined_score": 0.6},
            metadata=meta,
        )

    def test_no_repair_no_stats_change(self):
        tracer = self._make_tracer()
        trace = self._make_trace(repair_history=None)
        tracer._update_stats(trace)
        self.assertEqual(tracer.stats["repair_triggered"], 0)
        self.assertEqual(tracer.stats["repair_succeeded"], 0)

    def test_repair_triggered_incremented(self):
        tracer = self._make_tracer()
        trace = self._make_trace(repair_history=[{"attempt": 1, "error": "err", "succeeded": False}])
        tracer._update_stats(trace)
        self.assertEqual(tracer.stats["repair_triggered"], 1)
        self.assertEqual(tracer.stats["repair_succeeded"], 0)

    def test_repair_triggered_and_succeeded(self):
        tracer = self._make_tracer()
        trace = self._make_trace(repair_history=[
            {"attempt": 1, "error": "err1", "succeeded": False},
            {"attempt": 2, "error": None, "succeeded": True},
        ])
        tracer._update_stats(trace)
        self.assertEqual(tracer.stats["repair_triggered"], 1)
        self.assertEqual(tracer.stats["repair_succeeded"], 1)

    def test_get_statistics_includes_rates(self):
        tracer = self._make_tracer()
        # Simulate 4 traces: 2 triggered repair, 1 succeeded
        for _ in range(2):
            tracer._update_stats(self._make_trace())  # no repair
        tracer._update_stats(self._make_trace([{"attempt": 1, "error": "e", "succeeded": True}]))
        tracer._update_stats(self._make_trace([{"attempt": 1, "error": "e", "succeeded": False}]))

        stats = tracer.get_statistics()
        self.assertEqual(stats["repair_triggered"], 2)
        self.assertEqual(stats["repair_succeeded"], 1)
        self.assertAlmostEqual(stats["repair_trigger_rate"], 0.5)
        self.assertAlmostEqual(stats["repair_success_rate"], 0.5)

    def test_repair_rates_zero_when_none_triggered(self):
        tracer = self._make_tracer()
        tracer._update_stats(self._make_trace())
        stats = tracer.get_statistics()
        self.assertEqual(stats["repair_trigger_rate"], 0)
        self.assertEqual(stats["repair_success_rate"], 0)


# ---------------------------------------------------------------------------
# Repair metadata interception logic (mirrors iteration.py)
# ---------------------------------------------------------------------------

class TestRepairMetadataInterception(unittest.TestCase):
    """
    Tests the logic that iteration.py and process_parallel.py use to
    intercept repaired code from the evaluator side-channel.
    """

    def _simulate_interception(self, evaluator, child_id, original_code):
        """Replicate the logic in iteration.py after evaluate_program returns."""
        artifacts = evaluator.get_pending_artifacts(child_id)
        repaired_code = evaluator.get_pending_repair(child_id)
        repair_metadata = {}
        if repaired_code is not None:
            repair_metadata["original_llm_code"] = original_code
            repair_metadata["repair_history"] = (artifacts or {}).pop("repair_history", [])
            child_code = repaired_code
        else:
            child_code = original_code
        return child_code, repair_metadata, artifacts

    def setUp(self):
        eval_src = "def evaluate(path): return {'score': 1.0}\n"
        self.eval_file = _make_eval_file(eval_src)
        self.evaluator = _make_evaluator(EvaluatorConfig(), self.eval_file)

    def tearDown(self):
        os.unlink(self.eval_file)

    def test_no_repair_child_code_unchanged(self):
        original = "original code"
        child_code, repair_meta, _ = self._simulate_interception(
            self.evaluator, "prog-x", original
        )
        self.assertEqual(child_code, original)
        self.assertEqual(repair_meta, {})

    def test_repair_child_code_replaced(self):
        original = "broken code"
        repaired = "fixed code"
        history = [{"attempt": 1, "error": None, "succeeded": True}]

        self.evaluator._pending_repairs["prog-y"] = repaired
        self.evaluator._pending_artifacts["prog-y"] = {"repair_history": history}

        child_code, repair_meta, artifacts = self._simulate_interception(
            self.evaluator, "prog-y", original
        )

        self.assertEqual(child_code, repaired)
        self.assertEqual(repair_meta["original_llm_code"], original)
        self.assertEqual(repair_meta["repair_history"], history)
        # repair_history should have been popped from artifacts
        self.assertNotIn("repair_history", (artifacts or {}))

    def test_repair_history_not_in_llm_artifacts_after_interception(self):
        """repair_history must be moved to metadata, not remain in prompt artifacts."""
        self.evaluator._pending_repairs["prog-z"] = "fixed"
        self.evaluator._pending_artifacts["prog-z"] = {
            "repair_history": [{"attempt": 1, "succeeded": True}],
            "domain_breakdown": "some llm artifact",
        }

        _, _, remaining_artifacts = self._simulate_interception(
            self.evaluator, "prog-z", "original"
        )

        self.assertNotIn("repair_history", remaining_artifacts)
        self.assertIn("domain_breakdown", remaining_artifacts)


if __name__ == "__main__":
    unittest.main()
