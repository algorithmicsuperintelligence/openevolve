"""Behavior and compatibility tests for pluggable island scheduling."""

import asyncio
import unittest
from concurrent.futures import Future
from dataclasses import FrozenInstanceError
from enum import IntEnum
from unittest.mock import patch

import numpy as np

from openevolve import run_evolution
from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.process_parallel import ProcessParallelController, SerializableResult
from openevolve.selection import IslandSelectionContext


class TestIslandSelector(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.config.database.num_islands = 3
        self.config.evaluator.parallel_evaluations = 2
        self.database = ProgramDatabase(self.config.database)

    def _run_with_submit_mock(self, selector=None, iterations=7):
        controller = ProcessParallelController(
            self.config, "unused.py", self.database, island_selector=selector
        )
        controller.executor = object()  # run_evolution checks that a pool was started.
        submitted = []

        def submit(iteration, island_id):
            submitted.append((iteration, island_id))
            future = Future()
            future.set_result(SerializableResult(error="test result"))
            return future

        with patch.object(controller, "_submit_iteration", side_effect=submit):
            asyncio.run(controller.run_evolution(1, iterations))
        return submitted

    def test_default_round_robin_scheduling_is_unchanged(self):
        with patch.object(self.database, "get_island_stats", side_effect=AssertionError):
            submitted = self._run_with_submit_mock(iterations=6)
        self.assertEqual(submitted, [(1, 0), (2, 1), (3, 2), (4, 0), (5, 1), (6, 2)])

    def test_custom_selector_controls_initial_and_replacement_tasks(self):
        contexts = []

        def selector(context):
            contexts.append(context)
            return 2

        submitted = self._run_with_submit_mock(selector=selector)
        self.assertEqual(submitted, [(i, 2) for i in range(1, 8)])
        self.assertEqual([context.iteration for context in contexts], list(range(1, 8)))
        self.assertEqual(contexts[0].pending_counts, (0, 0, 0))
        self.assertEqual(contexts[1].pending_counts, (0, 0, 1))
        self.assertEqual(contexts[0].islands[0].population_size, 0)
        self.assertIsInstance(contexts[0], IslandSelectionContext)
        with self.assertRaises(FrozenInstanceError):
            contexts[0].pending_counts = (0, 0, 0)

    def test_invalid_island_is_rejected_before_submission(self):
        for invalid in (-1, 3, True, "1"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "island_selector must return"):
                    self._run_with_submit_mock(selector=lambda context: invalid)

    def test_integral_selector_results_are_normalized(self):
        class Island(IntEnum):
            SECOND = 1

        for island_id in (np.int64(1), Island.SECOND):
            with self.subTest(island_id=island_id):
                self.assertEqual(
                    self._run_with_submit_mock(selector=lambda context: island_id, iterations=1),
                    [(1, 1)],
                )

    def test_selector_receives_population_scores(self):
        self.database.add(
            Program(id="candidate", code="pass", metrics={"combined_score": 0.75}),
            target_island=1,
        )
        contexts = []

        def select_best(context):
            contexts.append(context)
            return max(range(len(context.islands)), key=lambda i: context.islands[i].best_score)

        self.assertEqual(self._run_with_submit_mock(select_best, iterations=1), [(1, 1)])
        self.assertEqual(contexts[0].islands[1].population_size, 1)
        self.assertEqual(contexts[0].islands[1].best_score, 0.75)
        self.assertEqual(contexts[0].islands[1].diversity, 0.0)

    def test_public_api_forwards_selector(self):
        async def fake_run(*args, **kwargs):
            return "forwarded"

        selector = lambda context: 0
        with patch("openevolve.api._run_evolution_async", side_effect=fake_run) as mock_run:
            result = run_evolution("program", "evaluator", island_selector=selector)

        self.assertEqual(result, "forwarded")
        self.assertIs(mock_run.call_args.args[-1], selector)
