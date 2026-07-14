"""
Integration tests for OpenEvolve library API with real LLM inference
Tests the end-to-end flow of using OpenEvolve as a library
"""

import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path

from openevolve import run_evolution, evolve_function, evolve_code, evolve_algorithm
from openevolve.config import Config, LLMModelConfig

# Reuse the shared test model constant so this config tracks the server's model.
sys.path.append(str(Path(__file__).parent.parent))
from test_utils import TEST_MODEL


def _get_library_test_config(port: int = 8000) -> Config:
    """Get config for library API tests with optillm server"""
    config = Config()
    config.max_iterations = 100
    config.checkpoint_interval = 1
    config.database.in_memory = True
    config.evaluator.cascade_evaluation = False
    config.evaluator.parallel_evaluations = 1
    config.evaluator.timeout = 60

    # Configure to use optillm server
    base_url = f"http://localhost:{port}/v1"
    config.llm.api_base = base_url
    config.llm.timeout = 120
    config.llm.retries = 0
    # Bound generation length so the (non-stopping) dhara model does not ramble to
    # the 4096-token default on every call (see tests/test_utils.py for details).
    config.llm.max_tokens = 256
    config.llm.models = [
        LLMModelConfig(
            name=TEST_MODEL,
            api_key="optillm",
            api_base=base_url,
            weight=1.0,
            max_tokens=256,
            timeout=120,
            retries=0,
        )
    ]
    return config


class TestLibraryAPIIntegration:
    """Test OpenEvolve library API with real LLM integration"""

    @pytest.mark.slow
    def test_evolve_function_real_integration(self, optillm_server, temp_workspace):
        """Test evolve_function with real optillm server - simple optimization task"""

        def simple_multiply(x, y):
            """A simple function that can be optimized"""
            # Inefficient implementation that can be improved
            result = 0
            for i in range(x):
                result += y
            return result

        # Test cases - the function should return x * y
        test_cases = [((2, 3), 6), ((4, 5), 20), ((1, 7), 7), ((0, 10), 0)]

        print("Testing evolve_function with real LLM...")

        # Run evolution with minimal iterations for testing
        result = evolve_function(
            simple_multiply,
            test_cases,
            iterations=2,  # Very small number for CI speed
            output_dir=str(temp_workspace / "evolve_function_output"),
            cleanup=False,  # Keep files for inspection
            config=_get_library_test_config(optillm_server["port"]),
        )

        # Verify the result structure
        assert result is not None
        assert hasattr(result, "best_score")
        assert hasattr(result, "best_code")
        assert hasattr(result, "metrics")
        assert hasattr(result, "output_dir")

        # Basic checks
        assert result.best_score >= 0.0
        assert "def simple_multiply" in result.best_code
        assert result.output_dir == str(temp_workspace / "evolve_function_output")

        # Check that output directory was created
        output_path = Path(result.output_dir)
        assert output_path.exists()
        assert (output_path / "best").exists()

        print(f"✅ evolve_function completed successfully!")
        print(f"   Best score: {result.best_score}")
        print(f"   Output dir: {result.output_dir}")
        print(f"   Code length: {len(result.best_code)} chars")

    @pytest.mark.slow
    def test_evolve_code_real_integration(self, optillm_server, temp_workspace):
        """Test evolve_code with real optillm server - code string optimization"""

        # Initial code that can be optimized
        initial_code = """
# EVOLVE-BLOCK-START
def fibonacci(n):
    # Inefficient recursive implementation
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
# EVOLVE-BLOCK-END
"""

        def fibonacci_evaluator(program_path):
            """Simple evaluator for fibonacci function"""
            try:
                # Import the evolved program
                import importlib.util

                spec = importlib.util.spec_from_file_location("evolved", program_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Test the function
                if hasattr(module, "fibonacci"):
                    fib = module.fibonacci

                    # Test cases
                    test_cases = [(0, 0), (1, 1), (2, 1), (3, 2), (4, 3), (5, 5)]

                    correct = 0
                    for input_val, expected in test_cases:
                        try:
                            result = fib(input_val)
                            if result == expected:
                                correct += 1
                        except:
                            pass

                    accuracy = correct / len(test_cases)
                    return {
                        "score": accuracy,
                        "correctness": accuracy,
                        "test_cases_passed": correct,
                        "combined_score": accuracy,  # Use accuracy as combined score
                    }
                else:
                    return {"score": 0.0, "error": "fibonacci function not found"}

            except Exception as e:
                return {"score": 0.0, "error": str(e)}

        print("Testing evolve_code with real LLM...")

        # Run evolution
        result = evolve_code(
            initial_code,
            fibonacci_evaluator,
            iterations=1,  # Minimal for CI speed
            output_dir=str(temp_workspace / "evolve_code_output"),
            cleanup=False,  # Keep output directory
            config=_get_library_test_config(optillm_server["port"]),
        )

        # Verify result structure
        assert result is not None
        assert result.best_score >= 0.0
        assert "fibonacci" in result.best_code.lower()
        assert "# EVOLVE-BLOCK-START" in result.best_code
        assert "# EVOLVE-BLOCK-END" in result.best_code

        # Check output directory
        output_path = Path(result.output_dir)
        assert output_path.exists()

        print(f"✅ evolve_code completed successfully!")
        print(f"   Best score: {result.best_score}")
        print(f"   Output dir: {result.output_dir}")

    @pytest.mark.slow
    def test_run_evolution_real_integration(self, optillm_server, temp_workspace):
        """Test run_evolution with real optillm server - basic program evolution"""

        # Create initial program file
        initial_program = temp_workspace / "initial_program.py"
        initial_program.write_text("""
# Simple sorting program to evolve
# EVOLVE-BLOCK-START
def sort_numbers(numbers):
    # Basic bubble sort implementation
    n = len(numbers)
    for i in range(n):
        for j in range(0, n - i - 1):
            if numbers[j] > numbers[j + 1]:
                numbers[j], numbers[j + 1] = numbers[j + 1], numbers[j]
    return numbers
# EVOLVE-BLOCK-END
""")

        # Create evaluator file
        evaluator_file = temp_workspace / "evaluator.py"
        evaluator_file.write_text("""
def evaluate(program_path):
    \"\"\"Evaluate sorting function performance\"\"\"
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("program", program_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, 'sort_numbers'):
            sort_func = module.sort_numbers
            
            # Test cases
            test_cases = [
                [3, 1, 4, 1, 5],
                [9, 2, 6, 5, 3],
                [1],
                [],
                [2, 1]
            ]
            
            correct = 0
            for test_case in test_cases:
                try:
                    input_copy = test_case.copy()
                    result = sort_func(input_copy)
                    expected = sorted(test_case)
                    if result == expected:
                        correct += 1
                except:
                    pass
            
            accuracy = correct / len(test_cases) if test_cases else 0
            return {
                "score": accuracy,
                "correctness": accuracy,
                "complexity": 10,  # Fixed complexity for simplicity
                "combined_score": accuracy  # Use accuracy as combined score
            }
        else:
            return {"score": 0.0, "error": "sort_numbers function not found"}
            
    except Exception as e:
        return {"score": 0.0, "error": str(e)}
""")

        print("Testing run_evolution with real LLM...")

        # Run evolution using file paths (most common usage)
        result = run_evolution(
            initial_program=str(initial_program),
            evaluator=str(evaluator_file),
            iterations=1,  # Minimal for CI speed
            output_dir=str(temp_workspace / "run_evolution_output"),
            cleanup=False,  # Keep output directory
            config=_get_library_test_config(optillm_server["port"]),
        )

        # Verify result
        assert result is not None
        assert result.best_score >= 0.0
        assert "sort_numbers" in result.best_code

        # Check that files were created
        output_path = Path(result.output_dir)
        assert output_path.exists()
        assert (output_path / "best").exists()
        assert (output_path / "checkpoints").exists()

        print(f"✅ run_evolution completed successfully!")
        print(f"   Best score: {result.best_score}")
        print(f"   Output dir: {result.output_dir}")

        # Test string input as well
        print("Testing run_evolution with string inputs...")

        result2 = run_evolution(
            initial_program=initial_program.read_text(),
            evaluator=lambda path: {"score": 0.8, "test": "passed"},  # Simple callable evaluator
            iterations=1,
            output_dir=str(temp_workspace / "run_evolution_string_output"),
            cleanup=False,  # Keep output directory
            config=_get_library_test_config(optillm_server["port"]),
        )

        assert result2 is not None
        assert result2.best_score >= 0.0

        print(f"✅ run_evolution with string inputs completed!")


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for integration tests"""
    temp_dir = tempfile.mkdtemp()
    workspace = Path(temp_dir)
    yield workspace
    shutil.rmtree(temp_dir, ignore_errors=True)
