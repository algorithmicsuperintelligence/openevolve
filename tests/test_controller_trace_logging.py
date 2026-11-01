"""
Tests that the parallel controller writes every child to the evolution trace,
including children whose parent left the archive while they were evaluated
"""

import asyncio
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test")

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.evolution_trace import EvolutionTracer
from openevolve.process_parallel import ProcessParallelController, SerializableResult

EVALUATOR_CODE = """
def evaluate(program_path):
    return {"combined_score": 0.5}
"""


def _future(result):
    future = MagicMock()
    future.done.return_value = True
    future.result.return_value = result
    return future


def _child(iteration, parent_id):
    return SerializableResult(
        child_program_dict={
            "id": f"child_{iteration}",
            "code": f"def evolved_{iteration}(): return {iteration}",
            "language": "python",
            "parent_id": parent_id,
            "generation": 1,
            "metrics": {"combined_score": 0.7},
            "iteration_found": iteration,
            "metadata": {
                "changes": "test",
                "island": 0,
                "parent_metrics": {"combined_score": 0.25},
            },
        },
        parent_id=parent_id,
        iteration_time=0.1,
        iteration=iteration,
        target_island=0,
    )


class TestControllerTraceLogging(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.eval_file = os.path.join(self.test_dir, "evaluator.py")
        with open(self.eval_file, "w") as f:
            f.write(EVALUATOR_CODE)

        self.config = Config()
        self.config.evaluator.parallel_evaluations = 1
        self.config.evaluator.timeout = 10
        self.config.database.num_islands = 1
        self.config.database.in_memory = True

        self.database = ProgramDatabase(self.config.database)
        self.database.add(
            Program(
                id="seed",
                code="def seed(): return 0",
                language="python",
                metrics={"combined_score": 0.5},
                iteration_found=0,
            )
        )
        self.trace_path = Path(self.test_dir) / "evolution_trace.jsonl"
        self.tracer = EvolutionTracer(output_path=str(self.trace_path), buffer_size=1)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _run(self, results):
        async def run_test():
            controller = ProcessParallelController(
                self.config, self.eval_file, self.database, evolution_tracer=self.tracer
            )
            with patch.object(
                controller,
                "_submit_iteration",
                side_effect=lambda i, island: _future(results[i]),
            ):
                controller.start()
                try:
                    await controller.run_evolution(
                        start_iteration=1, max_iterations=len(results), target_score=None
                    )
                finally:
                    controller.stop()

        asyncio.run(run_test())
        self.tracer.close()

    def _read(self, path):
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_child_of_evicted_parent_is_traced(self):
        # "evicted" was never in (or has left) the archive when the child comes back
        self._run({1: _child(1, "evicted"), 2: _child(2, "seed")})

        traced = {t["iteration"]: t for t in self._read(self.trace_path)}
        self.assertEqual(sorted(traced), [1, 2])
        self.assertEqual(traced[1]["parent_id"], "evicted")
        self.assertEqual(traced[1]["parent_metrics"], {"combined_score": 0.25})
        # A parent still in the archive is traced from the archive, as before
        self.assertEqual(traced[2]["parent_metrics"], {"combined_score": 0.5})


if __name__ == "__main__":
    unittest.main()
