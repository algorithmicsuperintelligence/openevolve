"""
Tests for runs where every iteration fails (issue #288)

When the LLM endpoint is down, every iteration errors out before producing a
candidate program. Such a run used to finish with "Evolution complete!", exit
code 0 and the untouched seed saved as the best program, which is
indistinguishable from a run that simply found nothing better.
"""

import asyncio
import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

# Set dummy API key for testing
os.environ["OPENAI_API_KEY"] = "test"

from openevolve import EvolutionFailedError, OpenEvolve
from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.process_parallel import ProcessParallelController, SerializableResult

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EVALUATOR_CODE = """
def evaluate(program_path):
    return {"combined_score": 0.5}
"""


def _error_future(iteration, error="LLM generation failed: Connection error."):
    future = MagicMock()
    future.done.return_value = True
    future.result.return_value = SerializableResult(error=error, iteration=iteration)
    return future


def _child_future(iteration, island_id):
    future = MagicMock()
    future.done.return_value = True
    future.result.return_value = SerializableResult(
        child_program_dict={
            "id": f"child_{iteration}",
            "code": f"def evolved_{iteration}(): return {iteration}",
            "language": "python",
            "parent_id": "seed",
            "generation": 1,
            "metrics": {"combined_score": 0.4},
            "iteration_found": iteration,
            "metadata": {"changes": "test", "island": island_id},
        },
        parent_id="seed",
        iteration_time=0.1,
        iteration=iteration,
    )
    return future


class TestControllerCountsCandidates(unittest.TestCase):
    """ProcessParallelController must report whether any iteration produced a candidate"""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.eval_file = os.path.join(self.test_dir, "evaluator.py")
        with open(self.eval_file, "w") as f:
            f.write(EVALUATOR_CODE)

        self.config = Config()
        self.config.evaluator.parallel_evaluations = 2
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

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _run(self, make_future, max_iterations=4, before_run=None):
        async def run_test():
            controller = ProcessParallelController(self.config, self.eval_file, self.database)
            with patch.object(controller, "_submit_iteration", side_effect=make_future):
                controller.start()
                try:
                    if before_run:
                        before_run(controller)
                    await controller.run_evolution(
                        start_iteration=1, max_iterations=max_iterations, target_score=None
                    )
                finally:
                    controller.stop()
            return controller

        return asyncio.run(run_test())

    def test_all_errors_produce_no_candidates(self):
        controller = self._run(lambda i, island: _error_future(i))

        self.assertTrue(controller.produced_no_candidates)
        self.assertEqual(controller.completed_iterations, 4)
        self.assertEqual(controller.successful_iterations, 0)
        self.assertEqual(
            controller.last_iteration_error, "LLM generation failed: Connection error."
        )

    def test_single_success_counts_as_progress(self):
        """Intermittent failures are still tolerated as long as something was generated"""
        controller = self._run(
            lambda i, island: _child_future(i, island) if i == 3 else _error_future(i)
        )

        self.assertFalse(controller.produced_no_candidates)
        self.assertEqual(controller.successful_iterations, 1)

    def test_shutdown_is_not_a_failure(self):
        """An interrupted run is not reported as a failed one"""
        controller = self._run(
            lambda i, island: _error_future(i),
            before_run=lambda c: c.request_shutdown(),
        )

        self.assertFalse(controller.produced_no_candidates)


class TestOpenEvolveRunFailsWithoutCandidates(unittest.TestCase):
    """OpenEvolve.run must raise instead of presenting the seed as the evolved best"""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.program_file = os.path.join(self.test_dir, "initial_program.py")
        with open(self.program_file, "w") as f:
            f.write("# EVOLVE-BLOCK-START\ndef f():\n    return 0\n# EVOLVE-BLOCK-END\n")
        self.eval_file = os.path.join(self.test_dir, "evaluator.py")
        with open(self.eval_file, "w") as f:
            f.write(EVALUATOR_CODE)
        self.output_dir = os.path.join(self.test_dir, "out")

        self.config = Config()
        self.config.evaluator.parallel_evaluations = 1
        self.config.database.num_islands = 1
        self.config.database.in_memory = True

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _run(self, make_future):
        openevolve = OpenEvolve(
            initial_program_path=self.program_file,
            evaluation_file=self.eval_file,
            config=self.config,
            output_dir=self.output_dir,
        )
        with patch.object(ProcessParallelController, "_submit_iteration", autospec=True) as m:
            m.side_effect = lambda _self, i, island=None: make_future(i, island)
            return asyncio.run(openevolve.run(iterations=3))

    def test_raises_and_does_not_save_seed_as_best(self):
        with self.assertRaises(EvolutionFailedError) as ctx:
            self._run(lambda i, island: _error_future(i))

        self.assertIn("0/3 iterations produced a candidate program", str(ctx.exception))
        self.assertIn("Connection error", str(ctx.exception))
        self.assertFalse(os.path.exists(os.path.join(self.output_dir, "best")))

    def test_partial_success_still_returns_best(self):
        best = self._run(lambda i, island: _child_future(i, island) if i == 2 else _error_future(i))

        self.assertIsNotNone(best)
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "best")))


class _DeadEndpointHandler(http.server.BaseHTTPRequestHandler):
    """Answers every LLM request with HTTP 404, like a missing deployment"""

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = json.dumps(
            {"error": {"code": "DeploymentNotFound", "message": "deployment does not exist"}}
        ).encode()
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class TestCLIExitCodeWithDeadEndpoint(unittest.TestCase):
    """End-to-end CLI check against a dead LLM endpoint (reproduction from issue #288)"""

    def setUp(self):
        self.server = http.server.HTTPServer(("127.0.0.1", 0), _DeadEndpointHandler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_exits_non_zero_without_success_banner(self):
        program_file = os.path.join(self.test_dir, "initial_program.py")
        with open(program_file, "w") as f:
            f.write("# EVOLVE-BLOCK-START\ndef f():\n    return 0\n# EVOLVE-BLOCK-END\n")
        eval_file = os.path.join(self.test_dir, "evaluator.py")
        with open(eval_file, "w") as f:
            f.write(EVALUATOR_CODE)
        config_file = os.path.join(self.test_dir, "config.yaml")
        with open(config_file, "w") as f:
            f.write(
                "llm:\n"
                f"  api_base: http://127.0.0.1:{self.server.server_port}/v1\n"
                "  primary_model: dead-model\n"
                "  retries: 0\n"
                "  retry_delay: 0\n"
                "  timeout: 5\n"
                "database:\n"
                "  num_islands: 1\n"
                "evaluator:\n"
                "  parallel_evaluations: 1\n"
            )
        output_dir = os.path.join(self.test_dir, "out")

        env = dict(os.environ)
        env.setdefault("OPENAI_API_KEY", "test-key-for-unit-tests")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "openevolve.cli",
                program_file,
                eval_file,
                "--config",
                config_file,
                "--iterations",
                "2",
                "--output",
                output_dir,
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("0/2 iterations produced a candidate program", result.stdout)
        self.assertNotIn("Evolution complete!", result.stdout)
        self.assertFalse(os.path.exists(os.path.join(output_dir, "best")))


if __name__ == "__main__":
    unittest.main()
