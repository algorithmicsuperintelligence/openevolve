"""Exercise progress and continuation against the database interface."""

import asyncio
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from openevolve.config import Config, LLMModelConfig
from openevolve.api import run_evolution
from openevolve.controller import OpenEvolve
from openevolve.database import Program, ProgramDatabase
from openevolve.database_memory import InMemoryProgramDatabase
from openevolve.process_parallel import ProcessParallelController, SerializableResult


class ImmediateExecutor:
    """Complete worker results locally without making model requests."""

    def __init__(self, succeed=False):
        self.iterations = []
        self.contexts = []
        self.succeed = succeed

    def submit(self, fn, iteration, context):
        self.iterations.append(iteration)
        self.contexts.append(context)
        future = Future()
        if self.succeed:
            child = Program(
                id=f"child-{iteration}",
                code=f"return {iteration}",
                parent_id=context.parent.id,
                metrics={"combined_score": 1.0},
            )
            future.set_result(
                SerializableResult(
                    iteration=iteration,
                    child_program_dict=child.to_dict(),
                    parent_id=context.parent.id,
                    target_island=context.target_island,
                    artifacts={"stderr": "child evidence"},
                    prompt={"system": "system", "user": "user"},
                    llm_response="response",
                )
            )
        else:
            future.set_result(SerializableResult(iteration=iteration, error="No valid code"))
        return future

    def shutdown(self, **kwargs):
        pass


class TestIterationCounting(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.program_path = Path(self.temp_dir.name) / "program.py"
        self.program_path.write_text("def solve(): return 1\n")
        self.eval_path = Path(self.temp_dir.name) / "evaluator.py"
        self.eval_path.write_text("def evaluate(path): return {'combined_score': 0.5}\n")
        self.config = Config()
        self.config.database.num_islands = 2
        self.config.evaluator.parallel_evaluations = 1
        self.config.evaluator.cascade_evaluation = False
        self.store = InMemoryProgramDatabase(self.config.database)
        # This facade raises AttributeError for backing maps, config, and progress fields.
        self.database = Mock(spec_set=ProgramDatabase, wraps=self.store)

    def controller(self):
        with patch("openevolve.controller.LLMEnsemble"), patch.object(OpenEvolve, "_setup_logging"):
            controller = OpenEvolve(
                str(self.program_path),
                str(self.eval_path),
                self.config,
                output_dir=self.temp_dir.name,
                database=self.database,
            )
        controller.evaluator.evaluate_program = AsyncMock(return_value={"combined_score": 0.5})
        controller.evaluator.get_pending_artifacts = Mock(return_value={"stdout": "initial"})
        return controller

    def test_fresh_run_and_continuation_count_failed_iterations(self):
        controller = self.controller()
        executor = ImmediateExecutor()

        def start(parallel):
            parallel.executor = executor

        with patch.object(ProcessParallelController, "start", start), patch("signal.signal"):
            asyncio.run(controller.run(iterations=3))
            self.assertEqual(executor.iterations, [1, 2, 3])
            self.assertEqual(self.database.get_state().last_iteration, 3)
            # A second controller can continue against the same database instance.
            next_controller = self.controller()
            asyncio.run(next_controller.run(iterations=2))

        self.assertEqual(executor.iterations, [1, 2, 3, 4, 5])
        self.assertEqual(self.database.get_state().last_iteration, 5)
        self.assertEqual(self.database.get_state().program_count, 1)
        controller.evaluator.evaluate_program.assert_awaited_once()
        next_controller.evaluator.evaluate_program.assert_not_awaited()
        self.assertEqual(self.database.sample_from_island.call_count, 5)
        self.assertEqual(self.database.get_top_programs.call_count, 5)
        self.assertFalse((Path(self.temp_dir.name) / "checkpoints").exists())
        self.assertTrue((Path(self.temp_dir.name) / "best" / "best_program.py").exists())

    def test_zero_iterations_only_evaluates_initial_program(self):
        controller = self.controller()
        executor = ImmediateExecutor()
        with (
            patch.object(
                ProcessParallelController, "start", lambda p: setattr(p, "executor", executor)
            ),
            patch("signal.signal"),
        ):
            asyncio.run(controller.run(iterations=0))
        self.assertEqual(executor.iterations, [])
        self.assertEqual(self.database.get_state().last_iteration, 0)
        self.assertEqual(self.database.get_state().program_count, 1)
        controller.evaluator.evaluate_program.assert_awaited_once()

    def test_library_api_accepts_an_existing_database(self):
        self.config.llm.models = [LLMModelConfig(name="test")]
        self.database.add(Program(id="existing", code="return 1", metrics={"combined_score": 0.5}))
        self.database.record_iteration(10)
        executor = ImmediateExecutor()
        with (
            patch.object(
                ProcessParallelController, "start", lambda p: setattr(p, "executor", executor)
            ),
            patch.object(OpenEvolve, "_setup_logging"),
            patch("openevolve.controller.LLMEnsemble"),
            patch("signal.signal"),
        ):
            result = run_evolution(
                self.program_path,
                self.eval_path,
                config=self.config,
                iterations=1,
                output_dir=self.temp_dir.name,
                cleanup=False,
                database=self.database,
            )
        self.assertEqual(executor.iterations, [11])
        self.assertEqual(self.database.get_state().last_iteration, 11)
        self.assertEqual(result.best_program.id, "existing")

    def test_successful_result_uses_only_interface_and_records_target_stop(self):
        self.database.add(
            Program(id="initial", code="return 0", metrics={"combined_score": 0.5}), target_island=0
        )
        tracer = Mock()
        parallel = ProcessParallelController(
            self.config, str(self.eval_path), self.database, tracer
        )
        parallel.executor = ImmediateExecutor(succeed=True)
        result = asyncio.run(parallel.run_evolution(1, 1, target_score=1.0))
        self.assertEqual(result.id, "child-1")
        self.assertEqual(self.database.get_state().last_iteration, 1)
        self.assertEqual(self.database.get_artifacts(result.id), {"stderr": "child evidence"})
        self.assertEqual(
            self.database.get_prompt_history(result.id)["diff_user"]["responses"], ["response"]
        )
        self.assertEqual(self.database.get_island_stats()[0]["generation"], 1)
        tracer.log_trace.assert_called_once()

    def test_worker_exception_also_advances_progress(self):
        self.database.add(Program(id="initial", code="pass"))
        parallel = ProcessParallelController(self.config, str(self.eval_path), self.database)
        parallel.executor = Mock()
        future = Future()
        future.set_exception(RuntimeError("worker exited"))
        parallel.executor.submit.return_value = future
        asyncio.run(parallel.run_evolution(1, 1))
        self.assertEqual(self.database.get_state().last_iteration, 1)
