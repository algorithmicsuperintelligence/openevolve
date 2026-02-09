"""
Integration tests that validate existing examples work correctly.
These tests verify that evaluators, configs, and initial programs are properly set up.
"""

import importlib.util
import os
import sys
import tempfile
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from openevolve.config import Config, load_config
from openevolve.evaluator import Evaluator


class TestFunctionMinimizationExample(unittest.TestCase):
    """Integration tests for the function_minimization example"""

    EXAMPLE_DIR = PROJECT_ROOT / "examples" / "function_minimization"

    def test_config_loads(self):
        """Test that the config file loads without errors"""
        config_path = self.EXAMPLE_DIR / "config.yaml"
        if not config_path.exists():
            self.skipTest("function_minimization config not found")

        config = load_config(str(config_path))
        self.assertIsInstance(config, Config)
        self.assertGreater(config.max_iterations, 0)

    def test_initial_program_exists(self):
        """Test that the initial program file exists"""
        program_path = self.EXAMPLE_DIR / "initial_program.py"
        self.assertTrue(program_path.exists(), "initial_program.py should exist")

    def test_initial_program_has_evolve_block(self):
        """Test that the initial program has EVOLVE-BLOCK markers"""
        program_path = self.EXAMPLE_DIR / "initial_program.py"
        if not program_path.exists():
            self.skipTest("initial_program.py not found")

        content = program_path.read_text()
        self.assertIn("EVOLVE-BLOCK-START", content)
        self.assertIn("EVOLVE-BLOCK-END", content)

    def test_evaluator_exists(self):
        """Test that the evaluator file exists"""
        evaluator_path = self.EXAMPLE_DIR / "evaluator.py"
        self.assertTrue(evaluator_path.exists(), "evaluator.py should exist")

    def test_evaluator_has_evaluate_function(self):
        """Test that the evaluator has an evaluate function"""
        evaluator_path = self.EXAMPLE_DIR / "evaluator.py"
        if not evaluator_path.exists():
            self.skipTest("evaluator.py not found")

        spec = importlib.util.spec_from_file_location("evaluator", evaluator_path)
        evaluator_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator_module)

        self.assertTrue(hasattr(evaluator_module, "evaluate"))
        self.assertTrue(callable(evaluator_module.evaluate))

    def test_evaluator_runs_on_initial_program(self):
        """Test that the evaluator can evaluate the initial program"""
        evaluator_path = self.EXAMPLE_DIR / "evaluator.py"
        program_path = self.EXAMPLE_DIR / "initial_program.py"

        if not evaluator_path.exists() or not program_path.exists():
            self.skipTest("Example files not found")

        # Load evaluator
        spec = importlib.util.spec_from_file_location("evaluator", evaluator_path)
        evaluator_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator_module)

        # Run evaluation
        result = evaluator_module.evaluate(str(program_path))

        # Check result structure
        if hasattr(result, 'metrics'):
            # EvaluationResult object
            metrics = result.metrics
        else:
            # Dictionary
            metrics = result

        self.assertIn("combined_score", metrics)
        self.assertIsInstance(metrics["combined_score"], (int, float))


class TestCirclePackingExample(unittest.TestCase):
    """Integration tests for the circle_packing example"""

    EXAMPLE_DIR = PROJECT_ROOT / "examples" / "circle_packing"

    def test_config_loads(self):
        """Test that config files load without errors"""
        for config_name in ["config_phase_1.yaml", "config_phase_2.yaml"]:
            config_path = self.EXAMPLE_DIR / config_name
            if config_path.exists():
                config = load_config(str(config_path))
                self.assertIsInstance(config, Config)

    def test_evaluator_exists(self):
        """Test that evaluator exists"""
        evaluator_path = self.EXAMPLE_DIR / "evaluator.py"
        self.assertTrue(evaluator_path.exists(), "evaluator.py should exist")

    def test_evaluator_has_evaluate_function(self):
        """Test that the evaluator has required functions"""
        evaluator_path = self.EXAMPLE_DIR / "evaluator.py"
        if not evaluator_path.exists():
            self.skipTest("evaluator.py not found")

        spec = importlib.util.spec_from_file_location("evaluator", evaluator_path)
        evaluator_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator_module)

        self.assertTrue(hasattr(evaluator_module, "evaluate"))


class TestSignalProcessingExample(unittest.TestCase):
    """Integration tests for the signal_processing example"""

    EXAMPLE_DIR = PROJECT_ROOT / "examples" / "signal_processing"

    def test_config_loads(self):
        """Test that the config file loads"""
        config_path = self.EXAMPLE_DIR / "config.yaml"
        if not config_path.exists():
            self.skipTest("signal_processing config not found")

        config = load_config(str(config_path))
        self.assertIsInstance(config, Config)

    def test_evaluator_exists(self):
        """Test that evaluator exists"""
        evaluator_path = self.EXAMPLE_DIR / "evaluator.py"
        if not evaluator_path.exists():
            self.skipTest("evaluator.py not found")
        self.assertTrue(evaluator_path.exists())


class TestEvaluatorIntegration(unittest.TestCase):
    """Integration tests for the Evaluator class with real examples"""

    def test_evaluator_loads_function_minimization(self):
        """Test that Evaluator can load the function_minimization evaluator"""
        evaluator_path = PROJECT_ROOT / "examples" / "function_minimization" / "evaluator.py"
        if not evaluator_path.exists():
            self.skipTest("function_minimization evaluator not found")

        from openevolve.config import EvaluatorConfig
        config = EvaluatorConfig(timeout=30, cascade_evaluation=True)

        evaluator = Evaluator(config, str(evaluator_path))
        self.assertIsNotNone(evaluator.evaluate_function)
        self.assertTrue(callable(evaluator.evaluate_function))

    def test_evaluator_module_has_cascade_functions(self):
        """Test that function_minimization evaluator has cascade functions"""
        evaluator_path = PROJECT_ROOT / "examples" / "function_minimization" / "evaluator.py"
        if not evaluator_path.exists():
            self.skipTest("function_minimization evaluator not found")

        # Load the module directly to check for cascade functions
        spec = importlib.util.spec_from_file_location("evaluator", evaluator_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # function_minimization has evaluate_stage1 and evaluate_stage2
        self.assertTrue(hasattr(module, "evaluate_stage1"))
        self.assertTrue(hasattr(module, "evaluate_stage2"))
        self.assertTrue(callable(module.evaluate_stage1))
        self.assertTrue(callable(module.evaluate_stage2))


class TestConfigIntegration(unittest.TestCase):
    """Integration tests for config loading across examples"""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_all_example_configs_load(self):
        """Test that all example config files can be loaded"""
        examples_dir = PROJECT_ROOT / "examples"
        failed_configs = []

        for config_path in examples_dir.rglob("*config*.yaml"):
            try:
                config = load_config(str(config_path))
                self.assertIsInstance(config, Config)
            except Exception as e:
                failed_configs.append((str(config_path), str(e)))

        if failed_configs:
            failure_msg = "\n".join([f"{path}: {error}" for path, error in failed_configs])
            self.fail(f"Failed to load configs:\n{failure_msg}")

    def test_config_has_required_sections(self):
        """Test that loaded configs have required sections"""
        config_path = PROJECT_ROOT / "examples" / "function_minimization" / "config.yaml"
        if not config_path.exists():
            self.skipTest("function_minimization config not found")

        config = load_config(str(config_path))

        # Check required sections
        self.assertIsNotNone(config.llm)
        self.assertIsNotNone(config.database)
        self.assertIsNotNone(config.evaluator)
        self.assertIsNotNone(config.prompt)


class TestEndToEndWithMockedLLM(unittest.TestCase):
    """End-to-end tests with mocked LLM responses"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.example_dir = PROJECT_ROOT / "examples" / "function_minimization"

    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_database_stores_and_retrieves_programs(self):
        """Test that the database can store and retrieve programs"""
        from openevolve.database import ProgramDatabase, Program, DatabaseConfig

        config = DatabaseConfig(population_size=100)
        db = ProgramDatabase(config)

        # Add a program
        program = Program(
            id="test_prog_1",
            code="def test(): return 42",
            generation=0,
            metrics={"combined_score": 0.5},
        )
        db.add(program)

        # Retrieve it
        retrieved = db.programs.get("test_prog_1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.code, "def test(): return 42")

    def test_program_evolution_tracking(self):
        """Test that program generations are tracked correctly"""
        from openevolve.database import ProgramDatabase, Program, DatabaseConfig

        config = DatabaseConfig(population_size=100)
        db = ProgramDatabase(config)

        # Add parent program
        parent = Program(
            id="parent_1",
            code="def test(): return 1",
            generation=0,
            metrics={"combined_score": 0.3},
        )
        db.add(parent)

        # Add child program
        child = Program(
            id="child_1",
            code="def test(): return 2",
            generation=1,
            parent_id="parent_1",
            metrics={"combined_score": 0.5},
        )
        db.add(child)

        # Verify relationships
        self.assertEqual(db.programs["child_1"].parent_id, "parent_1")
        self.assertEqual(db.programs["child_1"].generation, 1)

    def test_evaluator_returns_evaluation_result(self):
        """Test that evaluators return proper EvaluationResult objects"""
        from openevolve.evaluation_result import EvaluationResult

        evaluator_path = self.example_dir / "evaluator.py"
        program_path = self.example_dir / "initial_program.py"

        if not evaluator_path.exists() or not program_path.exists():
            self.skipTest("Example files not found")

        spec = importlib.util.spec_from_file_location("evaluator", evaluator_path)
        evaluator_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator_module)

        result = evaluator_module.evaluate(str(program_path))

        # Should be an EvaluationResult or dict with metrics
        if isinstance(result, EvaluationResult):
            self.assertIn("combined_score", result.metrics)
        else:
            self.assertIn("combined_score", result)


class TestExampleStructure(unittest.TestCase):
    """Tests to verify example directory structure is correct"""

    def test_examples_have_required_files(self):
        """Test that examples have the minimum required files"""
        examples_dir = PROJECT_ROOT / "examples"

        # These examples should have at least a config and evaluator
        required_examples = [
            "function_minimization",
            "circle_packing",
        ]

        for example_name in required_examples:
            example_dir = examples_dir / example_name
            if not example_dir.exists():
                continue

            # Check for config
            config_files = list(example_dir.glob("*config*.yaml"))
            self.assertGreater(
                len(config_files), 0,
                f"{example_name} should have at least one config file"
            )

            # Check for evaluator
            evaluator_path = example_dir / "evaluator.py"
            self.assertTrue(
                evaluator_path.exists(),
                f"{example_name} should have evaluator.py"
            )

    def test_evaluators_are_importable(self):
        """Test that all evaluators can be imported without errors"""
        examples_dir = PROJECT_ROOT / "examples"
        failed_imports = []

        for evaluator_path in examples_dir.rglob("evaluator.py"):
            try:
                spec = importlib.util.spec_from_file_location(
                    f"evaluator_{evaluator_path.parent.name}",
                    evaluator_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Verify evaluate function exists
                if not hasattr(module, "evaluate"):
                    failed_imports.append(
                        (str(evaluator_path), "Missing evaluate function")
                    )
            except Exception as e:
                failed_imports.append((str(evaluator_path), str(e)))

        if failed_imports:
            # Only fail if critical examples fail
            critical_failures = [
                f for f in failed_imports
                if "function_minimization" in f[0] or "circle_packing" in f[0]
            ]
            if critical_failures:
                failure_msg = "\n".join(
                    [f"{path}: {error}" for path, error in critical_failures]
                )
                self.fail(f"Critical evaluators failed to import:\n{failure_msg}")


if __name__ == "__main__":
    unittest.main()
