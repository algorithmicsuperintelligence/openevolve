"""Zero-weight LLM judges must not issue requests or misalign feedback weights."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from openevolve.config import EvaluatorConfig, LLMModelConfig
from openevolve.evaluator import Evaluator
from openevolve.llm.ensemble import LLMEnsemble


class TestDisabledLLMJudges(unittest.IsolatedAsyncioTestCase):
    def make_ensemble(self, weights, responses):
        clients = []

        def init_client(config):
            client = Mock(model=config.name)
            client.generate_with_context = AsyncMock(return_value=responses[len(clients)])
            clients.append(client)
            return client

        ensemble = LLMEnsemble(
            [
                LLMModelConfig(name=f"judge-{i}", weight=weight, init_client=init_client)
                for i, weight in enumerate(weights)
            ]
        )
        return ensemble, clients

    def make_evaluator(self, ensemble, database=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        evaluation_file = Path(directory.name) / "evaluate.py"
        evaluation_file.write_text('def evaluate(path):\n    return {"combined_score": 0.5}\n')
        sampler = Mock()
        sampler.build_prompt.return_value = {"system": "judge", "user": "code"}
        evaluator = Evaluator(
            EvaluatorConfig(cascade_evaluation=False, use_llm_feedback=True),
            str(evaluation_file),
            llm_ensemble=ensemble,
            prompt_sampler=sampler,
            database=database,
        )
        self.addCleanup(evaluator._executor.shutdown)
        return evaluator

    async def test_zero_weight_judges_are_skipped_in_every_position(self):
        for weights in ([0.0, 1.0], [1.0, 0.0], [0.0, 1.0, 0.0, 3.0, 0.0]):
            with self.subTest(weights=weights):
                replies = [f'{{"quality": {i / 10}}}' for i in range(len(weights))]
                ensemble, clients = self.make_ensemble(weights, replies)
                responses = await ensemble.generate_all_with_context("judge", [])

                self.assertEqual(
                    responses,
                    [reply if weight > 0 else None for weight, reply in zip(weights, replies)],
                )
                for weight, client in zip(weights, clients):
                    if weight == 0:
                        client.generate_with_context.assert_not_awaited()
                    else:
                        client.generate_with_context.assert_awaited_once_with("judge", [])

    async def test_enabled_judges_preserve_response_order_and_options(self):
        replies = ['{"quality": 0.6}', '{"quality": 0.8}']
        ensemble, clients = self.make_ensemble([1.0, 3.0], replies)
        messages = [{"role": "user", "content": "code"}]

        responses = await ensemble.generate_all_with_context(
            "judge", messages, temperature=0.2, max_tokens=50
        )

        self.assertEqual(responses, replies)
        for client in clients:
            client.generate_with_context.assert_awaited_once_with(
                "judge", messages, temperature=0.2, max_tokens=50
            )

    async def test_disabled_judge_failure_cannot_discard_active_feedback(self):
        for weights in ([0.0, 1.0], [1.0, 0.0]):
            with self.subTest(weights=weights):
                ensemble, clients = self.make_ensemble(weights, ['{"quality": 0.8}'] * 2)
                disabled = clients[weights.index(0.0)].generate_with_context
                disabled.side_effect = RuntimeError("disabled judge is unavailable")
                evaluator = self.make_evaluator(ensemble)

                metrics = await evaluator.evaluate_program("x = 1", "disabled-judge")

                disabled.assert_not_awaited()
                self.assertAlmostEqual(metrics["llm_quality"], 0.08)
                self.assertAlmostEqual(metrics["combined_score"], 0.59)

    async def test_skipped_slots_preserve_weighted_metrics_and_prompt_logging(self):
        replies = [
            '{"quality": 0.6, "clarity": 0.4}',
            '{"quality": 100, "disabled_only": 100}',
            '{"quality": 0.8}',
        ]
        ensemble, clients = self.make_ensemble([1.0, 0.0, 3.0], replies)
        database = Mock()
        database.config.feature_dimensions = []
        evaluator = self.make_evaluator(ensemble, database=database)

        result = await evaluator._llm_evaluate("x = 1", "weighted-judges")
        result = evaluator._process_evaluation_result(result)

        self.assertAlmostEqual(result.metrics["quality"], 0.75)
        self.assertAlmostEqual(result.metrics["clarity"], 0.4)
        self.assertNotIn("disabled_only", result.metrics)
        clients[1].generate_with_context.assert_not_awaited()
        database.log_prompt.assert_called_once_with(
            program_id="weighted-judges",
            template_key="evaluation",
            prompt={"system": "judge", "user": "code"},
            responses=[replies[0], None, replies[2]],
        )

    async def test_enabled_judge_errors_still_propagate(self):
        ensemble, clients = self.make_ensemble([1.0, 1.0], ['{"quality": 0.6}'] * 2)
        clients[1].generate_with_context.side_effect = RuntimeError("active judge failed")

        with self.assertRaisesRegex(RuntimeError, "active judge failed"):
            await ensemble.generate_all_with_context("judge", [])

        for client in clients:
            client.generate_with_context.assert_awaited_once_with("judge", [])
