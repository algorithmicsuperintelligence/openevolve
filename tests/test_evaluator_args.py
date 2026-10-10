"""
Tests for forwarding evaluator_args kwargs to the evaluation function
(GitHub issue #474: --evaluator-args CLI flag + config.evaluator.evaluator_args)
"""

import asyncio
import os
import subprocess
import sys
import tempfile
import unittest

from openevolve.cli import parse_evaluator_args
from openevolve.config import Config, EvaluatorConfig
from openevolve.evaluator import Evaluator

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class EvaluatorArgsTestBase(unittest.TestCase):
    """Shared helpers for writing temporary evaluation files"""

    def setUp(self):
        self._temp_files = []

    def tearDown(self):
        for path in self._temp_files:
            if os.path.exists(path):
                os.unlink(path)

    def _write_eval_file(self, source: str) -> str:
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        )
        temp_file.write(source)
        temp_file.close()
        self._temp_files.append(temp_file.name)
        return temp_file.name

    def _create_evaluator(self, eval_file: str, evaluator_args=None) -> Evaluator:
        config = EvaluatorConfig()
        config.timeout = 10
        config.max_retries = 1
        config.cascade_evaluation = False
        config.evaluator_args = evaluator_args
        return Evaluator(
            config=config,
            evaluation_file=eval_file,
            llm_ensemble=None,
            prompt_sampler=None,
        )


class TestEvaluatorKwargsForwarding(EvaluatorArgsTestBase):
    """Test that configured kwargs reach the evaluation function"""

    def test_multi_param_evaluator_receives_kwargs(self):
        """evaluate(program_path, weight=...) receives the configured kwarg"""
        eval_file = self._write_eval_file("""
def evaluate(program_path, weight=1.0):
    return {"weighted_score": weight}
""")
        evaluator = self._create_evaluator(eval_file, evaluator_args={"weight": 2.5})

        async def run_test():
            return await evaluator.evaluate_program("x = 1", "test_weighted")

        result = asyncio.run(run_test())
        self.assertEqual(result["weighted_score"], 2.5)

    def test_var_keyword_evaluator_receives_kwargs(self):
        """evaluate(program_path, **kwargs) receives all configured kwargs"""
        eval_file = self._write_eval_file("""
def evaluate(program_path, **kwargs):
    return {"score": 1.0 if kwargs.get("factor") == 3.0 and kwargs.get("shift") == -1 else 0.0}
""")
        evaluator = self._create_evaluator(eval_file, evaluator_args={"factor": 3.0, "shift": -1})

        async def run_test():
            return await evaluator.evaluate_program("x = 1", "test_var_keyword")

        result = asyncio.run(run_test())
        self.assertEqual(result["score"], 1.0)

    def test_single_param_evaluator_unchanged_without_args(self):
        """No evaluator_args configured: single-argument evaluators keep working"""
        eval_file = self._write_eval_file("""
def evaluate(program_path):
    return {"score": 0.5}
""")
        evaluator = self._create_evaluator(eval_file)
        self.assertEqual(evaluator._evaluator_kwargs, {})

        async def run_test():
            return await evaluator.evaluate_program("x = 1", "test_plain")

        result = asyncio.run(run_test())
        self.assertEqual(result["score"], 0.5)

    def test_configured_empty_args_unchanged(self):
        """Explicitly empty evaluator_args dict behaves like no args"""
        eval_file = self._write_eval_file("""
def evaluate(program_path):
    return {"score": 0.7}
""")
        evaluator = self._create_evaluator(eval_file, evaluator_args={})
        self.assertEqual(evaluator._evaluator_kwargs, {})

        async def run_test():
            return await evaluator.evaluate_program("x = 1", "test_empty")

        result = asyncio.run(run_test())
        self.assertEqual(result["score"], 0.7)


class TestEvaluatorKwargsValidation(EvaluatorArgsTestBase):
    """Test startup validation of evaluator_args against the evaluate() signature"""

    def test_unsupported_key_raises_at_startup(self):
        """Configured key not accepted by evaluate() fails at load time with the key named"""
        eval_file = self._write_eval_file("""
def evaluate(program_path):
    return {"score": 1.0}
""")
        with self.assertRaises(ValueError) as ctx:
            self._create_evaluator(eval_file, evaluator_args={"weight": 1.0})

        message = str(ctx.exception)
        self.assertIn("weight", message)
        self.assertIn("evaluate", message)
        self.assertIn(eval_file, message)

    def test_program_path_collision_raises(self):
        """A kwarg named like the positional program path parameter is rejected"""
        eval_file = self._write_eval_file("""
def evaluate(program_path):
    return {"score": 1.0}
""")
        with self.assertRaises(ValueError) as ctx:
            self._create_evaluator(eval_file, evaluator_args={"program_path": "other.py"})

        self.assertIn("program_path", str(ctx.exception))

    def test_program_path_collision_raises_for_var_keyword_evaluator(self):
        """A configured key shadowing the positional param of a **kwargs evaluator is rejected"""
        eval_file = self._write_eval_file("""
def evaluate(program_path, **kwargs):
    return {"score": kwargs.get("weight", 1.0)}
""")
        with self.assertRaises(ValueError) as ctx:
            self._create_evaluator(eval_file, evaluator_args={"program_path": "other.py"})

        message = str(ctx.exception)
        self.assertIn("program_path", message)
        self.assertIn("multiple values", message)

    def test_var_keyword_evaluator_still_accepts_non_colliding_kwargs(self):
        """**kwargs evaluators keep receiving configured keys that do not shadow parameters"""
        eval_file = self._write_eval_file("""
def evaluate(program_path, **kwargs):
    return {"score": kwargs.get("weight", 1.0)}
""")
        evaluator = self._create_evaluator(eval_file, evaluator_args={"weight": 0.25})

        async def run_test():
            return await evaluator.evaluate_program("x = 1", "test_vk_no_collision")

        result = asyncio.run(run_test())
        self.assertEqual(result["score"], 0.25)

    def test_error_lists_actual_signature(self):
        """The actionable error includes the evaluator's actual signature"""
        eval_file = self._write_eval_file("""
def evaluate(program_path):
    return {"score": 1.0}
""")
        with self.assertRaises(ValueError) as ctx:
            self._create_evaluator(eval_file, evaluator_args={"a": 1, "b": 2})

        message = str(ctx.exception)
        self.assertIn("evaluate(program_path)", message)
        self.assertIn("'a'", message)
        self.assertIn("'b'", message)


class TestEvaluatorArgsConfig(unittest.TestCase):
    """Test config plumbing (YAML load + worker dict round-trip)"""

    def test_yaml_load_evaluator_args(self):
        """evaluator.evaluator_args loads from YAML via dacite"""
        import tempfile

        config_content = """
evaluator:
  timeout: 60
  evaluator_args:
    weight: 2.0
    label: "test"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name
        try:
            from openevolve.config import load_config

            config = load_config(config_path)
            self.assertEqual(config.evaluator.evaluator_args, {"weight": 2.0, "label": "test"})
        finally:
            os.unlink(config_path)

    def test_worker_round_trip_preserves_evaluator_args(self):
        """to_dict() -> EvaluatorConfig(**d['evaluator']) keeps the field (process pool path)"""
        config = Config(language="python")
        config.evaluator.evaluator_args = {"weight": 3.5, "mode": "fast"}

        config_dict = config.to_dict()
        reconstructed = EvaluatorConfig(**config_dict["evaluator"])

        self.assertEqual(reconstructed.evaluator_args, {"weight": 3.5, "mode": "fast"})

    def test_default_is_none(self):
        """Default config has no evaluator_args (backward compatible)"""
        self.assertIsNone(EvaluatorConfig().evaluator_args)


class TestParseEvaluatorArgs(unittest.TestCase):
    """Test the CLI --evaluator-args JSON parsing helper"""

    def test_valid_json_object(self):
        self.assertEqual(
            parse_evaluator_args('{"weight": 2.0, "label": "x"}'), {"weight": 2.0, "label": "x"}
        )

    def test_empty_object(self):
        self.assertEqual(parse_evaluator_args("{}"), {})

    def test_invalid_json_raises(self):
        with self.assertRaises(ValueError) as ctx:
            parse_evaluator_args("{'weight': 2.0}")  # single quotes are not JSON
        self.assertIn("not valid JSON", str(ctx.exception))

    def test_non_object_json_raises(self):
        with self.assertRaises(ValueError) as ctx:
            parse_evaluator_args("[1, 2, 3]")
        self.assertIn("JSON object", str(ctx.exception))


class TestCLIEvaluatorArgsExitCode(unittest.TestCase):
    """End-to-end CLI check: invalid --evaluator-args JSON exits with code 1"""

    def test_invalid_json_exits_1(self):
        program_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        program_file.write("x = 1\n")
        program_file.close()
        eval_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        eval_file.write("def evaluate(program_path):\n    return {'score': 1.0}\n")
        eval_file.close()
        try:
            env = dict(os.environ)
            env.setdefault("OPENAI_API_KEY", "test-key-for-unit-tests")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "openevolve.cli",
                    program_file.name,
                    eval_file.name,
                    "--evaluator-args",
                    "{invalid json",
                ],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("not valid JSON", result.stdout + result.stderr)
        finally:
            os.unlink(program_file.name)
            os.unlink(eval_file.name)


if __name__ == "__main__":
    unittest.main()
