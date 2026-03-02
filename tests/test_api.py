"""
Test the library API functionality
"""
import unittest
import unittest.mock
import tempfile
import os
from pathlib import Path

from openevolve.api import (
    run_evolution, 
    evolve_function, 
    evolve_algorithm, 
    evolve_code,
    EvolutionResult,
    _prepare_program,
    _prepare_evaluator
)
from openevolve.config import Config


class TestAPIFunctions(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_evolution_result_class(self):
        """Test EvolutionResult dataclass"""
        result = EvolutionResult(
            best_program=None,
            best_score=0.85,
            best_code="def test(): pass",
            metrics={"score": 0.85, "runtime": 1.2},
            output_dir="/tmp/test"
        )
        
        self.assertEqual(result.best_score, 0.85)
        self.assertEqual(result.best_code, "def test(): pass")
        self.assertIn("0.8500", str(result))
    
    def test_prepare_program_from_file(self):
        """Test _prepare_program with existing file"""
        program_file = os.path.join(self.temp_dir, "test_program.py")
        with open(program_file, 'w') as f:
            f.write("def test(): return 42")
        
        temp_files = []
        result = _prepare_program(program_file, self.temp_dir, temp_files)
        
        self.assertEqual(result, program_file)
        self.assertEqual(len(temp_files), 0)
    
    def test_prepare_program_from_string(self):
        """Test _prepare_program with code string"""
        code = "def test(): return 42"
        temp_files = []
        
        result = _prepare_program(code, self.temp_dir, temp_files)
        
        self.assertTrue(os.path.exists(result))
        self.assertEqual(len(temp_files), 1)
        
        with open(result, 'r') as f:
            content = f.read()
            self.assertIn("EVOLVE-BLOCK-START", content)
            self.assertIn("EVOLVE-BLOCK-END", content)
            self.assertIn("def test(): return 42", content)
    
    def test_prepare_program_from_list(self):
        """Test _prepare_program with list of lines"""
        lines = ["def test():", "    return 42"]
        temp_files = []
        
        result = _prepare_program(lines, self.temp_dir, temp_files)
        
        self.assertTrue(os.path.exists(result))
        self.assertEqual(len(temp_files), 1)
        
        with open(result, 'r') as f:
            content = f.read()
            self.assertIn("def test():\n    return 42", content)
    
    def test_prepare_program_with_existing_markers(self):
        """Test _prepare_program doesn't add duplicate markers"""
        code = """# EVOLVE-BLOCK-START
def test(): 
    return 42
# EVOLVE-BLOCK-END"""
        temp_files = []
        
        result = _prepare_program(code, self.temp_dir, temp_files)
        
        with open(result, 'r') as f:
            content = f.read()
            # Should not have nested markers
            self.assertEqual(content.count("EVOLVE-BLOCK-START"), 1)
            self.assertEqual(content.count("EVOLVE-BLOCK-END"), 1)
    
    def test_prepare_evaluator_from_file(self):
        """Test _prepare_evaluator with existing file"""
        eval_file = os.path.join(self.temp_dir, "evaluator.py") 
        with open(eval_file, 'w') as f:
            f.write("def evaluate(path): return {'score': 1.0}")
        
        temp_files = []
        result = _prepare_evaluator(eval_file, self.temp_dir, temp_files)
        
        self.assertEqual(result, eval_file)
        self.assertEqual(len(temp_files), 0)
    
    def test_prepare_evaluator_from_callable(self):
        """Test _prepare_evaluator with callable function"""
        def my_evaluator(program_path):
            return {"score": 0.8, "test": "passed"}

        temp_files = []
        result = _prepare_evaluator(my_evaluator, self.temp_dir, temp_files)

        self.assertTrue(os.path.exists(result))
        # 2 temp files: the pickle file + the wrapper .py
        self.assertEqual(len(temp_files), 2)

        with open(result, 'r') as f:
            content = f.read()
            self.assertIn("def evaluate(program_path)", content)
    
    def test_prepare_evaluator_from_string(self):
        """Test _prepare_evaluator with code string"""
        code = "def evaluate(path): return {'score': 0.9}"
        temp_files = []
        
        result = _prepare_evaluator(code, self.temp_dir, temp_files)
        
        self.assertTrue(os.path.exists(result))
        self.assertEqual(len(temp_files), 1)
        
        with open(result, 'r') as f:
            content = f.read()
            self.assertEqual(content, code)
    
    def test_prepare_evaluator_string_without_evaluate_function(self):
        """Test _prepare_evaluator raises error for invalid code string"""
        code = "def my_function(): pass"
        temp_files = []
        
        with self.assertRaises(ValueError):
            _prepare_evaluator(code, self.temp_dir, temp_files)
    
    def test_evolve_function_basic(self):
        """Test evolve_function with simple test case"""
        def initial_sort(arr):
            # Simple bubble sort
            for i in range(len(arr)):
                for j in range(len(arr)-1):
                    if arr[j] > arr[j+1]:
                        arr[j], arr[j+1] = arr[j+1], arr[j]
            return arr
        
        test_cases = [
            ([3, 1, 2], [1, 2, 3]),
            ([5, 2], [2, 5]),
        ]
        
        # Mock the async controller to avoid actual evolution
        with unittest.mock.patch('openevolve.api._run_evolution_async') as mock_async:
            mock_async.return_value = EvolutionResult(
                best_program=None,
                best_score=1.0,
                best_code="def initial_sort(arr): return sorted(arr)",
                metrics={"score": 1.0, "test_pass_rate": 1.0},
                output_dir=None
            )
            
            result = evolve_function(initial_sort, test_cases, iterations=1)
            
            self.assertIsInstance(result, EvolutionResult)
            self.assertEqual(result.best_score, 1.0)
            mock_async.assert_called_once()
    
    def test_evolve_algorithm_basic(self):
        """Test evolve_algorithm with simple class"""
        class SimpleAlgorithm:
            def process(self, data):
                return sum(data)
        
        def benchmark(instance):
            result = instance.process([1, 2, 3])
            return {"score": 1.0 if result == 6 else 0.0}
        
        # Mock the controller
        with unittest.mock.patch('openevolve.api._run_evolution_async') as mock_async:
            mock_async.return_value = EvolutionResult(
                best_program=None,
                best_score=1.0,
                best_code="class SimpleAlgorithm: pass",
                metrics={"score": 1.0},
                output_dir=None
            )
            
            result = evolve_algorithm(SimpleAlgorithm, benchmark, iterations=1)
            
            self.assertIsInstance(result, EvolutionResult)
            self.assertEqual(result.best_score, 1.0)
            mock_async.assert_called_once()
    
    def test_evolve_code_basic(self):
        """Test evolve_code with string input"""
        code = "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
        
        def evaluator(program_path):
            return {"score": 0.5, "correctness": True}
        
        # Mock the controller
        with unittest.mock.patch('openevolve.api._run_evolution_async') as mock_async:
            mock_async.return_value = EvolutionResult(
                best_program=None,
                best_score=0.8,
                best_code=code,
                metrics={"score": 0.8},
                output_dir=None
            )
            
            result = evolve_code(code, evaluator, iterations=1)
            
            self.assertIsInstance(result, EvolutionResult)
            self.assertEqual(result.best_score, 0.8)
            mock_async.assert_called_once()
    
    def test_run_evolution_with_config_object(self):
        """Test run_evolution with Config object"""
        config = Config()
        config.num_iterations = 5
        
        # Mock the controller
        with unittest.mock.patch('openevolve.api._run_evolution_async') as mock_async:
            mock_async.return_value = EvolutionResult(
                best_program=None,
                best_score=0.9,
                best_code="def test(): pass",
                metrics={"score": 0.9},
                output_dir=None
            )
            
            result = run_evolution(
                initial_program="def test(): pass",
                evaluator=lambda p: {"score": 1.0},
                config=config,
                iterations=10
            )
            
            self.assertIsInstance(result, EvolutionResult) 
            self.assertEqual(result.best_score, 0.9)
            mock_async.assert_called_once()
    
    def test_run_evolution_cleanup_false(self):
        """Test run_evolution with cleanup=False"""
        with unittest.mock.patch('openevolve.api._run_evolution_async') as mock_async:
            mock_async.return_value = EvolutionResult(
                best_program=None,
                best_score=0.7,
                best_code="def test(): pass", 
                metrics={"score": 0.7},
                output_dir="/tmp/test_output"
            )
            
            result = run_evolution(
                initial_program="def test(): pass",
                evaluator=lambda p: {"score": 1.0},
                cleanup=False,
                output_dir="/tmp/test_output"
            )
            
            self.assertEqual(result.output_dir, "/tmp/test_output")
            mock_async.assert_called_once()


class TestEvaluatorCrossProcess(unittest.TestCase):
    """Test that callable evaluators work across process boundaries"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_callable_evaluator_works_in_subprocess(self):
        """Test that a callable evaluator serialized by _prepare_evaluator
        can be loaded and executed in a separate process (simulating
        ProcessPoolExecutor workers)."""
        def my_evaluator(program_path):
            return {"score": 0.42, "passed": True}

        temp_files = []
        eval_file = _prepare_evaluator(my_evaluator, self.temp_dir, temp_files)

        # Load and run the evaluator in a subprocess — this is what
        # process_parallel.py workers do.
        import subprocess, sys, json
        result = subprocess.run(
            [
                sys.executable, "-c",
                f"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("eval_mod", {eval_file!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print(json.dumps(mod.evaluate("dummy_path.py")))
"""
            ],
            capture_output=True, text=True, timeout=10
        )
        self.assertEqual(result.returncode, 0, f"Subprocess failed: {result.stderr}")
        metrics = json.loads(result.stdout.strip())
        self.assertAlmostEqual(metrics["score"], 0.42)
        self.assertTrue(metrics["passed"])

    def test_callable_evaluator_with_closure(self):
        """Test that a closure (capturing local variables) works across processes."""
        threshold = 0.5
        func_name = "my_func"

        def closure_evaluator(program_path):
            return {"score": threshold, "func": func_name}

        temp_files = []
        eval_file = _prepare_evaluator(closure_evaluator, self.temp_dir, temp_files)

        import subprocess, sys, json
        result = subprocess.run(
            [
                sys.executable, "-c",
                f"""
import importlib.util, json
spec = importlib.util.spec_from_file_location("eval_mod", {eval_file!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print(json.dumps(mod.evaluate("dummy.py")))
"""
            ],
            capture_output=True, text=True, timeout=10
        )
        self.assertEqual(result.returncode, 0, f"Subprocess failed: {result.stderr}")
        metrics = json.loads(result.stdout.strip())
        self.assertAlmostEqual(metrics["score"], 0.5)
        self.assertEqual(metrics["func"], "my_func")


if __name__ == '__main__':
    unittest.main()