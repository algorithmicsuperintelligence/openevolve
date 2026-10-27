"""Public run iteration budgets must bound the work submitted to the scheduler."""

import tempfile
import unittest
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from openevolve.config import Config
from openevolve.controller import OpenEvolve
from openevolve.database import Program
from openevolve.process_parallel import SerializableResult


class TestRunIterationBudget(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.workspace = Path(workspace.name)
        self.initial_code = "def solve(): return 0\n"
        initial_program = self.workspace / "initial.py"
        initial_program.write_text(self.initial_code)

        config = Config()
        config.max_iterations = 3
        config.database.in_memory = True
        config.database.num_islands = 1
        config.evaluator.parallel_evaluations = 1
        config.checkpoint_interval = 100

        evaluator = Mock()
        evaluator.evaluate_program = AsyncMock(return_value={"combined_score": 0.0})
        evaluator.get_pending_artifacts.return_value = None
        self.evaluator = evaluator

        self.executor = Mock(spec=ProcessPoolExecutor)
        self.executor.submit.side_effect = self.complete_iteration
        patches = ExitStack()
        self.addCleanup(patches.close)
        patches.enter_context(
            patch("openevolve.process_parallel.ProcessPoolExecutor", return_value=self.executor)
        )
        patches.enter_context(patch("openevolve.controller.Evaluator", return_value=evaluator))
        patches.enter_context(patch("openevolve.controller.LLMEnsemble"))
        patches.enter_context(patch("openevolve.controller.signal.signal"))
        patches.enter_context(patch.object(OpenEvolve, "_setup_logging"))

        self.controller = OpenEvolve(
            initial_program_path=str(initial_program),
            evaluation_file=str(self.workspace / "unused_evaluator.py"),
            config=config,
            output_dir=str(self.workspace / "output"),
        )

    @staticmethod
    def complete_iteration(worker, iteration, snapshot, parent_id, inspiration_ids):
        """Replace worker execution with a completed synthetic result."""
        program = Program(
            id=f"child_{iteration}",
            parent_id=parent_id,
            code=f"def solve(): return {iteration}\n",
            metrics={"combined_score": float(iteration)},
            iteration_found=iteration,
        )
        future = Future()
        future.set_result(
            SerializableResult(
                child_program_dict=program.to_dict(),
                parent_id=parent_id,
                iteration=iteration,
                target_island=snapshot["sampling_island"],
            )
        )
        return future

    async def test_zero_only_evaluates_initial_program(self):
        best = await self.controller.run(iterations=0)

        self.executor.submit.assert_not_called()
        self.evaluator.evaluate_program.assert_awaited_once_with(self.initial_code, best.id)
        self.assertEqual(best.code, self.initial_code)
        self.assertEqual(best.iteration_found, 0)
        self.assertEqual(len(self.controller.database.programs), 1)

    async def test_none_uses_configured_iterations(self):
        best = await self.controller.run(iterations=None)

        self.assertEqual(self.executor.submit.call_count, 3)
        self.assertEqual([call.args[1] for call in self.executor.submit.call_args_list], [1, 2, 3])
        self.evaluator.evaluate_program.assert_awaited_once()
        self.assertEqual(best.id, "child_3")

    async def test_positive_iterations_override_config(self):
        best = await self.controller.run(iterations=1)

        self.executor.submit.assert_called_once()
        self.assertEqual(self.executor.submit.call_args.args[1], 1)
        self.evaluator.evaluate_program.assert_awaited_once()
        self.assertEqual(best.id, "child_1")


if __name__ == "__main__":
    unittest.main()
