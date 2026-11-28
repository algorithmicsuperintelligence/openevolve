"""Tests for max_code_length diagnostics and parse_full_rewrite nested fence handling."""

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from openevolve.config import Config, LLMModelConfig
from openevolve.controller import OpenEvolve
from openevolve.database import Program, ProgramDatabase
from openevolve.process_parallel import ProcessParallelController
from openevolve.utils.code_utils import parse_full_rewrite


class TestParseFullRewriteNestedFences(unittest.TestCase):
    def test_preserves_inner_backticks_inside_python_strings(self):
        response = (
            "Here is the updated code:\n"
            "```python\n"
            "def example():\n"
            '    doc = """```\ninner\n```"""\n'
            "    return doc\n"
            "```\n"
            "Explanation after.\n"
        )
        parsed = parse_full_rewrite(response, "python")
        self.assertIsNotNone(parsed)
        self.assertIn("return doc", parsed)
        self.assertIn('"""```', parsed)

    def test_standard_fenced_blocks_and_fallback_still_work(self):
        response = "```python\nx = 42\n```\n"
        self.assertEqual(parse_full_rewrite(response, "python"), "x = 42")
        generic = "```\ny = 99\n```"
        self.assertEqual(parse_full_rewrite(generic, "python"), "y = 99")


class TestMaxCodeLengthDiagnostics(unittest.IsolatedAsyncioTestCase):
    def test_controller_warns_when_initial_or_island_seed_exceeds_max_code_length(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            init_file = root / "init.py"
            island_file = root / "island1.py"
            eval_file = root / "eval.py"
            init_file.write_text("# " + ("a" * 300) + "\ndef solve(): return 1\n")
            island_file.write_text("# " + ("b" * 300) + "\ndef solve(): return 2\n")
            eval_file.write_text('def evaluate(path):\n    return {"combined_score": 1.0}\n')

            cfg = Config()
            cfg.max_code_length = 100
            cfg.database.num_islands = 2
            cfg.database.island_initial_programs = [None, str(island_file)]
            cfg.llm.models = [LLMModelConfig(name="local", api_key="test")]
            cfg.evaluator.cascade_evaluation = False

            with self.assertLogs("openevolve.controller", level=logging.WARNING) as cm:
                OpenEvolve(
                    initial_program_path=str(init_file),
                    evaluation_file=str(eval_file),
                    config=cfg,
                    output_dir=str(root / "out"),
                )
            joined = "\n".join(cm.output)
            self.assertIn("Initial program length", joined)
            self.assertIn("Island 1 initial program length", joined)
            self.assertIn("config.max_code_length (100)", joined)

    async def test_process_parallel_tracks_and_warns_on_length_rejections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            eval_file = root / "eval.py"
            eval_file.write_text('def evaluate(path):\n    return {"combined_score": 1.0}\n')

            cfg = Config()
            cfg.max_iterations = 2
            cfg.max_code_length = 100
            cfg.llm.models = [LLMModelConfig(name="local", api_key="test")]
            cfg.evaluator.cascade_evaluation = False
            db = ProgramDatabase(cfg.database)
            db.add(Program(id="seed", code="x = 1", metrics={"combined_score": 0.1}))

            controller = ProcessParallelController(cfg, str(eval_file), db)

            class FakeLengthErrorResult:
                def __init__(self, iteration):
                    self.iteration = iteration
                    self.parent_id = "seed"
                    self.child_program = None
                    self.error = "Generated code exceeds maximum length (250 > 100)"
                    self.prompt = None
                    self.llm_response = None
                    self.artifacts = None
                    self.iteration_time = 0.01
                    self.token_usage = None
                    self.evolution_mode = "diff"

            futures = []
            for idx in (1, 2):
                fut = Mock()
                fut.done.return_value = True
                fut.result.return_value = FakeLengthErrorResult(idx)
                futures.append(fut)
            it = iter(futures)
            controller.executor = Mock()
            controller._submit_iteration = Mock(side_effect=lambda *a, **kw: next(it))

            with self.assertLogs("openevolve.process_parallel", level=logging.WARNING) as cm:
                await controller.run_evolution(start_iteration=1, max_iterations=2)

            self.assertEqual(controller.length_rejected_iterations, 2)
            self.assertTrue(controller.produced_no_candidates)
            self.assertIn("exceeded max_code_length=100", "\n".join(cm.output))
