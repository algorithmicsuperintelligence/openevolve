"""Keep valid LLM judge feedback when an ensemble response is incomplete."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from openevolve.config import EvaluatorConfig
from openevolve.evaluator import Evaluator


class TestLLMEvaluationEnsemble(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        evaluation_file = Path(self.directory.name) / "evaluate.py"
        evaluation_file.write_text('def evaluate(path):\n    return {"combined_score": 0.5}\n')
        self.ensemble = Mock(weights=[0.25, 0.75])
        self.sampler = Mock()
        self.sampler.build_prompt.return_value = {"system": "judge", "user": "code"}
        self.evaluator = Evaluator(
            EvaluatorConfig(cascade_evaluation=False, use_llm_feedback=True),
            str(evaluation_file),
            llm_ensemble=self.ensemble,
            prompt_sampler=self.sampler,
        )
        self.addCleanup(self.evaluator._executor.shutdown)

    async def evaluate_responses(self, responses):
        self.ensemble.generate_all_with_context = AsyncMock(return_value=responses)
        result = await self.evaluator._llm_evaluate("x = 1")
        return self.evaluator._process_evaluation_result(result)

    async def test_invalid_judge_does_not_discard_valid_feedback(self):
        for invalid in ("not JSON", "[]", "null"):
            for responses in ([invalid, '{"quality": 0.8}'], ['{"quality": 0.8}', invalid]):
                with self.subTest(responses=responses):
                    result = await self.evaluate_responses(responses)
                    self.assertAlmostEqual(result.metrics["quality"], 0.8)

    async def test_missing_metric_uses_weights_of_judges_that_report_it(self):
        result = await self.evaluate_responses(
            ['{"quality": 0.4, "clarity": 0.6}', '{"quality": 0.8}']
        )
        self.assertAlmostEqual(result.metrics["quality"], 0.7)
        self.assertAlmostEqual(result.metrics["clarity"], 0.6)

    async def test_zero_weight_judge_does_not_create_a_metric(self):
        self.ensemble.weights = [0.0, 1.0]
        result = await self.evaluate_responses(['{"quality": 0.9}', '{"clarity": 0.8}'])
        self.assertEqual(result.metrics, {"clarity": 0.8})

    async def test_all_invalid_judges_leave_measured_fitness_unchanged(self):
        self.ensemble.generate_all_with_context = AsyncMock(return_value=["invalid", "null"])
        metrics = await self.evaluator.evaluate_program("x = 1", "all-invalid")
        self.assertEqual(metrics, {"combined_score": 0.5})

    async def test_valid_feedback_still_contributes_to_combined_fitness(self):
        self.ensemble.generate_all_with_context = AsyncMock(
            return_value=["invalid", '{"quality": 0.8, "comment": "clear"}']
        )
        metrics = await self.evaluator.evaluate_program("x = 1", "one-valid")
        self.assertAlmostEqual(metrics["combined_score"], 0.59)
        self.assertAlmostEqual(
            metrics["llm_average"], 0.8 * self.evaluator.config.llm_feedback_weight
        )

    async def test_nonfinite_metrics_do_not_poison_other_judges(self):
        for invalid in ("NaN", "Infinity", "-Infinity", "1e999", "-1e999"):
            for invalid_index in (0, 1):
                with self.subTest(invalid=invalid, invalid_index=invalid_index):
                    responses = ['{"quality": 0.8}'] * 2
                    responses[invalid_index] = '{"quality": ' + invalid + ', "clarity": 0.6}'
                    result = await self.evaluate_responses(responses)
                    self.assertEqual(set(result.metrics), {"quality", "clarity"})
                    self.assertAlmostEqual(result.metrics["quality"], 0.8)
                    self.assertAlmostEqual(result.metrics["clarity"], 0.6)

    async def test_unrepresentable_integer_keeps_other_metrics(self):
        response = '{"quality": ' + "1" + "0" * 400 + ', "clarity": 0.6}'
        result = await self.evaluate_responses([response, '{"quality": 0.8}'])
        self.assertEqual(set(result.metrics), {"quality", "clarity"})
        self.assertAlmostEqual(result.metrics["quality"], 0.8)
        self.assertAlmostEqual(result.metrics["clarity"], 0.6)

    async def test_boolean_judge_fields_remain_artifacts(self):
        result = await self.evaluate_responses(
            ['{"quality": 0.4, "valid": true}', '{"quality": 0.8, "timeout": false}']
        )
        self.assertAlmostEqual(result.metrics["quality"], 0.7)
        self.assertEqual(set(result.metrics), {"quality"})
        self.assertEqual(result.artifacts, {"valid": True, "timeout": False})

    async def test_invalid_numeric_feedback_leaves_measured_fitness_unchanged(self):
        self.ensemble.generate_all_with_context = AsyncMock(
            return_value=['{"quality": NaN}', '{"quality": Infinity}']
        )
        metrics = await self.evaluator.evaluate_program("x = 1", "nonfinite-judges")
        self.assertEqual(metrics, {"combined_score": 0.5})

    async def test_valid_judge_survives_nonfinite_peer_in_combined_fitness(self):
        self.ensemble.generate_all_with_context = AsyncMock(
            return_value=['{"quality": NaN}', '{"quality": 0.8}']
        )
        metrics = await self.evaluator.evaluate_program("x = 1", "finite-peer")
        self.assertAlmostEqual(metrics["combined_score"], 0.59)
        self.assertAlmostEqual(metrics["llm_quality"], 0.08)

    async def test_finite_scores_are_not_clamped(self):
        result = await self.evaluate_responses(['{"quality": -2, "clarity": 3}', "{}"])
        self.assertEqual(result.metrics, {"quality": -2.0, "clarity": 3.0})
