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
        self.assertEqual(len(temp_files), 1)

        with open(result, 'r') as f:
            content = f.read()
            self.assertIn("def evaluate(program_path)", content)
            self.assertIn("my_evaluator", content)

    def test_prepare_evaluator_callable_works_in_subprocess(self):
        """Test that callable evaluator can be executed in a subprocess"""
        import subprocess
        import sys

        def my_evaluator(program_path):
            return {"score": 0.8, "combined_score": 0.8}

        temp_files = []
        eval_file = _prepare_evaluator(my_evaluator, self.temp_dir, temp_files)

        # Write a dummy program file for the evaluator to receive
        program_file = os.path.join(self.temp_dir, "dummy_program.py")
        with open(program_file, 'w') as f:
            f.write("x = 1\n")

        # Run the evaluator in a subprocess (simulating process-based parallelism)
        test_script = os.path.join(self.temp_dir, "run_eval.py")
        with open(test_script, 'w') as f:
            f.write(f"""
import sys
import importlib.util
spec = importlib.util.spec_from_file_location("evaluator", {eval_file!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
result = mod.evaluate({program_file!r})
assert isinstance(result, dict), f"Expected dict, got {{type(result)}}"
assert result["score"] == 0.8, f"Expected score 0.8, got {{result['score']}}"
print("OK")
""")

        proc = subprocess.run(
            [sys.executable, test_script],
            capture_output=True, text=True, timeout=10
        )
        self.assertEqual(proc.returncode, 0, f"Subprocess failed: {proc.stderr}")
        self.assertIn("OK", proc.stdout)
    
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

    def test_evolve_function_evaluator_works_in_subprocess(self):
        """Test that evolve_function generates an evaluator that works in a subprocess.

        This is a regression test for the bug where callable evaluators stored in
        globals() could not be accessed by process-based worker subprocesses.
        """
        import subprocess
        import sys

        def bubble_sort(arr):
            for i in range(len(arr)):
                for j in range(len(arr) - 1):
                    if arr[j] > arr[j + 1]:
                        arr[j], arr[j + 1] = arr[j + 1], arr[j]
            return arr

        test_cases = [([3, 1, 2], [1, 2, 3]), ([5, 2, 8], [2, 5, 8])]

        # Call evolve_function but intercept the evaluator code it generates
        # by capturing what gets passed to run_evolution
        with unittest.mock.patch('openevolve.api.run_evolution') as mock_run:
            mock_run.return_value = EvolutionResult(
                best_program=None, best_score=1.0,
                best_code="", metrics={}, output_dir=None
            )
            evolve_function(bubble_sort, test_cases, iterations=1)

            # Extract the evaluator code string passed to run_evolution
            call_kwargs = mock_run.call_args
            evaluator_code = call_kwargs.kwargs.get('evaluator') or call_kwargs[1].get('evaluator')

        self.assertIsInstance(evaluator_code, str, "evolve_function should pass evaluator as code string")
        self.assertIn("def evaluate(program_path)", evaluator_code)
        self.assertIn("combined_score", evaluator_code)

        # Write the evaluator to a file
        eval_file = os.path.join(self.temp_dir, "eval_test.py")
        with open(eval_file, 'w') as f:
            f.write(evaluator_code)

        # Write a correct program for the evaluator to test
        program_file = os.path.join(self.temp_dir, "program.py")
        with open(program_file, 'w') as f:
            f.write("def bubble_sort(arr):\n    return sorted(arr)\n")

        # Run in a subprocess to verify it works across process boundaries
        test_script = os.path.join(self.temp_dir, "run_eval.py")
        with open(test_script, 'w') as f:
            f.write(f"""
import importlib.util
spec = importlib.util.spec_from_file_location("evaluator", {eval_file!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
result = mod.evaluate({program_file!r})
assert result["combined_score"] == 1.0, f"Expected 1.0, got {{result['combined_score']}}"
assert result["tests_passed"] == 2, f"Expected 2, got {{result['tests_passed']}}"
print("OK")
""")

        proc = subprocess.run(
            [sys.executable, test_script],
            capture_output=True, text=True, timeout=10
        )
        self.assertEqual(proc.returncode, 0, f"Subprocess failed: {proc.stderr}")
        self.assertIn("OK", proc.stdout)
    
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


if __name__ == '__main__':
    unittest.main()