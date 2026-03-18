"""
High-level API for using OpenEvolve as a library
"""

import asyncio
import tempfile
import os
import uuid
import inspect
from typing import Union, Callable, Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from openevolve.controller import OpenEvolve
from openevolve.config import Config, load_config, LLMModelConfig
from openevolve.database import Program


@dataclass
class EvolutionResult:
    """Result of an evolution run"""

    best_program: Optional[Program]
    best_score: float
    best_code: str
    metrics: Dict[str, Any]
    output_dir: Optional[str]

    def __repr__(self):
        return f"EvolutionResult(best_score={self.best_score:.4f})"


def run_evolution(
    initial_program: Union[str, Path, List[str]],
    evaluator: Union[str, Path, Callable],
    config: Union[str, Path, Config, None] = None,
    iterations: Optional[int] = None,
    output_dir: Optional[str] = None,
    cleanup: bool = True,
) -> EvolutionResult:
    """
    Run evolution with flexible inputs - the main library API

    Args:
        initial_program: Can be:
            - Path to a program file (str or Path)
            - Program code as a string
            - List of code lines
        evaluator: Can be:
            - Path to an evaluator file (str or Path)
            - Callable function that takes (program_path) and returns metrics dict
        config: Can be:
            - Path to config YAML file (str or Path)
            - Config object
            - None for defaults
        iterations: Number of iterations (overrides config)
        output_dir: Output directory (None for temp directory)
        cleanup: If True, clean up temp files after evolution

    Returns:
        EvolutionResult with best program and metrics

    Examples:
        # Using file paths (original way)
        result = run_evolution(
            'program.py',
            'evaluator.py'
        )

        # Using code strings
        result = run_evolution(
            initial_program='''
                # EVOLVE-BLOCK-START
                def solve(x):
                    return x * 2
                # EVOLVE-BLOCK-END
            ''',
            evaluator=lambda path: {"score": evaluate_program(path)},
            iterations=100
        )

        # Using a custom evaluator function
        def my_evaluator(program_path):
            # Run tests, benchmarks, etc.
            return {"score": 0.95, "runtime": 1.2}

        result = run_evolution(
            initial_program=generate_initial_code(),
            evaluator=my_evaluator
        )
    """
    return asyncio.run(
        _run_evolution_async(initial_program, evaluator, config, iterations, output_dir, cleanup)
    )


async def _run_evolution_async(
    initial_program: Union[str, Path, List[str]],
    evaluator: Union[str, Path, Callable],
    config: Union[str, Path, Config, None],
    iterations: Optional[int],
    output_dir: Optional[str],
    cleanup: bool,
) -> EvolutionResult:
    """Async implementation of run_evolution"""

    temp_dir = None
    temp_files = []

    try:
        # Handle configuration
        if config is None:
            config_obj = Config()
        elif isinstance(config, Config):
            config_obj = config
        else:
            config_obj = load_config(str(config))

        # Validate that LLM models are configured
        if not config_obj.llm.models:
            raise ValueError(
                "No LLM models configured. Please provide a config with LLM models, or set up "
                "your configuration with models. For example:\n\n"
                "from openevolve.config import Config, LLMModelConfig\n"
                "config = Config()\n"
                "config.llm.models = [LLMModelConfig(name='gpt-4', api_key='your-key')]\n"
                "result = run_evolution(program, evaluator, config=config)"
            )

        # Set up output directory
        if output_dir is None and cleanup:
            temp_dir = tempfile.mkdtemp(prefix="openevolve_")
            actual_output_dir = temp_dir
        else:
            actual_output_dir = output_dir or "openevolve_output"
            os.makedirs(actual_output_dir, exist_ok=True)

        # Process initial program
        program_path = _prepare_program(initial_program, temp_dir, temp_files)

        # Process evaluator
        evaluator_path = _prepare_evaluator(evaluator, temp_dir, temp_files)

        # Auto-disable cascade evaluation if the evaluator doesn't define stage functions
        if config_obj.evaluator.cascade_evaluation:
            with open(evaluator_path, "r") as f:
                eval_content = f.read()
            if "evaluate_stage1" not in eval_content:
                config_obj.evaluator.cascade_evaluation = False

        # Create and run controller
        controller = OpenEvolve(
            initial_program_path=program_path,
            evaluation_file=evaluator_path,
            config=config_obj,
            output_dir=actual_output_dir,
        )

        best_program = await controller.run(iterations=iterations)

        # Prepare result
        best_score = 0.0
        metrics = {}
        best_code = ""

        if best_program:
            best_code = best_program.code
            metrics = best_program.metrics or {}

            if "combined_score" in metrics:
                best_score = metrics["combined_score"]
            elif metrics:
                numeric_metrics = [v for v in metrics.values() if isinstance(v, (int, float))]
                if numeric_metrics:
                    best_score = sum(numeric_metrics) / len(numeric_metrics)

        return EvolutionResult(
            best_program=best_program,
            best_score=best_score,
            best_code=best_code,
            metrics=metrics,
            output_dir=actual_output_dir if not cleanup else None,
        )

    finally:
        # Cleanup temporary files if requested
        if cleanup:
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
            if temp_dir and os.path.exists(temp_dir):
                import shutil

                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass


def _prepare_program(
    initial_program: Union[str, Path, List[str]], temp_dir: Optional[str], temp_files: List[str]
) -> str:
    """Convert program input to a file path"""

    # If already a file path, use it directly
    if isinstance(initial_program, (str, Path)):
        if os.path.exists(str(initial_program)):
            return str(initial_program)

    # Otherwise, treat as code and write to temp file
    if isinstance(initial_program, list):
        code = "\n".join(initial_program)
    else:
        code = str(initial_program)

    # Ensure code has evolution markers if it doesn't already
    if "EVOLVE-BLOCK-START" not in code:
        # Wrap entire code in evolution block
        code = f"""# EVOLVE-BLOCK-START
{code}
# EVOLVE-BLOCK-END"""

    # Write to temp file
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()

    program_file = os.path.join(temp_dir, f"program_{uuid.uuid4().hex[:8]}.py")
    with open(program_file, "w") as f:
        f.write(code)
    temp_files.append(program_file)

    return program_file


def _prepare_evaluator(
    evaluator: Union[str, Path, Callable], temp_dir: Optional[str], temp_files: List[str]
) -> str:
    """Convert evaluator input to a file path"""

    # If already a file path, use it directly
    if isinstance(evaluator, (str, Path)):
        if os.path.exists(str(evaluator)):
            return str(evaluator)

    # If it's a callable, create a wrapper module
    if callable(evaluator):
        # Try to get the source code of the callable so it can be serialized
        # into a standalone file that works in subprocesses
        try:
            func_source = inspect.getsource(evaluator)
            # Dedent in case the function was defined inside another scope
            import textwrap

            func_source = textwrap.dedent(func_source)
            func_name = evaluator.__name__

            # Build a self-contained evaluator module with the function source
            # and an evaluate() entry point that calls it
            evaluator_code = f"""
# Auto-generated evaluator from user-provided callable
import importlib.util
import sys
import os
import copy
import json
import time

{func_source}

def evaluate(program_path):
    '''Wrapper that calls the user-provided evaluator function'''
    return {func_name}(program_path)
"""
        except (OSError, TypeError):
            # If we can't get source (e.g. built-in, lambda, or closure),
            # fall back to the globals-based approach
            evaluator_id = f"_openevolve_evaluator_{uuid.uuid4().hex[:8]}"
            globals()[evaluator_id] = evaluator

            evaluator_code = f"""
# Wrapper for user-provided evaluator function
import {__name__} as api_module

def evaluate(program_path):
    '''Wrapper for user-provided evaluator function'''
    user_evaluator = getattr(api_module, '{evaluator_id}')
    return user_evaluator(program_path)
"""
    else:
        # Treat as code string
        evaluator_code = str(evaluator)

        # Ensure it has an evaluate function
        if "def evaluate" not in evaluator_code:
            raise ValueError("Evaluator code must contain an 'evaluate(program_path)' function")

    # Write to temp file
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()

    eval_file = os.path.join(temp_dir, f"evaluator_{uuid.uuid4().hex[:8]}.py")
    with open(eval_file, "w") as f:
        f.write(evaluator_code)
    temp_files.append(eval_file)

    return eval_file


# Additional helper functions for common use cases


def evolve_function(
    func: Callable, test_cases: List[Tuple[Any, Any]], iterations: int = 100, **kwargs
) -> EvolutionResult:
    """
    Evolve a Python function based on test cases

    Args:
        func: Initial function to evolve
        test_cases: List of (input, expected_output) tuples
        iterations: Number of evolution iterations
        **kwargs: Additional arguments for run_evolution

    Returns:
        EvolutionResult with optimized function

    Example:
        def initial_sort(arr):
            # Slow bubble sort
            for i in range(len(arr)):
                for j in range(len(arr)-1):
                    if arr[j] > arr[j+1]:
                        arr[j], arr[j+1] = arr[j+1], arr[j]
            return arr

        result = evolve_function(
            initial_sort,
            test_cases=[
                ([3, 1, 2], [1, 2, 3]),
                ([5, 2, 8, 1], [1, 2, 5, 8]),
            ],
            iterations=50
        )
        print(f"Optimized function score: {result.best_score}")
    """

    # Get function source code
    func_source = inspect.getsource(func)
    func_name = func.__name__

    # Ensure the function source has evolution markers
    if "EVOLVE-BLOCK-START" not in func_source:
        # Try to add markers around the function body
        lines = func_source.split("\n")
        func_def_line = next(i for i, line in enumerate(lines) if line.strip().startswith("def "))

        # Find the end of the function (simplified approach)
        indent = len(lines[func_def_line]) - len(lines[func_def_line].lstrip())
        func_end = len(lines)
        for i in range(func_def_line + 1, len(lines)):
            if lines[i].strip() and (len(lines[i]) - len(lines[i].lstrip())) <= indent:
                func_end = i
                break

        # Insert evolution markers
        lines.insert(func_def_line + 1, " " * (indent + 4) + "# EVOLVE-BLOCK-START")
        lines.insert(func_end + 1, " " * (indent + 4) + "# EVOLVE-BLOCK-END")
        func_source = "\n".join(lines)

    # Create a self-contained evaluator as a code string so it works in subprocesses.
    # Closure-based evaluators fail with process-based parallelism because subprocess
    # workers cannot access the parent process's memory.
    evaluator_code = f"""
import importlib.util
import copy

FUNC_NAME = {func_name!r}
TEST_CASES = {test_cases!r}

def evaluate(program_path):
    '''Auto-generated evaluator for evolve_function'''
    # Load the evolved program
    spec = importlib.util.spec_from_file_location("evolved", program_path)
    if spec is None or spec.loader is None:
        return {{"combined_score": 0.0, "score": 0.0, "error": "Failed to load program"}}

    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return {{"combined_score": 0.0, "score": 0.0, "error": f"Failed to execute program: {{str(e)}}"}}

    if not hasattr(module, FUNC_NAME):
        return {{"combined_score": 0.0, "score": 0.0, "error": f"Function '{{FUNC_NAME}}' not found"}}

    evolved_func = getattr(module, FUNC_NAME)
    correct = 0
    total = len(TEST_CASES)
    errors = []

    for input_val, expected in TEST_CASES:
        try:
            # Handle case where input is a list/mutable - make a copy
            if isinstance(input_val, list):
                test_input = input_val.copy()
            else:
                test_input = input_val

            result = evolved_func(test_input)
            if result == expected:
                correct += 1
            else:
                errors.append(f"Input {{input_val}}: expected {{expected}}, got {{result}}")
        except Exception as e:
            errors.append(f"Input {{input_val}}: {{str(e)}}")

    score = correct / total if total > 0 else 0.0
    return {{
        "combined_score": score,
        "score": score,
        "test_pass_rate": score,
        "tests_passed": correct,
        "total_tests": total,
        "errors": errors[:3],
    }}
"""

    return run_evolution(
        initial_program=func_source, evaluator=evaluator_code, iterations=iterations, **kwargs
    )


def evolve_algorithm(
    algorithm_class: type, benchmark: Callable, iterations: int = 100, **kwargs
) -> EvolutionResult:
    """
    Evolve an algorithm class based on a benchmark

    Args:
        algorithm_class: Initial algorithm class to evolve
        benchmark: Function that takes an instance and returns metrics
        iterations: Number of evolution iterations
        **kwargs: Additional arguments for run_evolution

    Returns:
        EvolutionResult with optimized algorithm

    Example:
        class SortAlgorithm:
            def sort(self, arr):
                # Simple bubble sort
                return sorted(arr)  # placeholder

        def benchmark_sort(instance):
            import time
            test_data = [list(range(100, 0, -1))]  # Reverse sorted

            start = time.time()
            for data in test_data:
                result = instance.sort(data.copy())
                if result != sorted(data):
                    return {"score": 0.0}

            duration = time.time() - start
            return {
                "score": 1.0,
                "runtime": duration,
                "performance": 1.0 / (duration + 0.001)
            }

        result = evolve_algorithm(SortAlgorithm, benchmark_sort, iterations=50)
    """

    # Get class source code
    class_source = inspect.getsource(algorithm_class)

    # Ensure the class has evolution markers
    if "EVOLVE-BLOCK-START" not in class_source:
        lines = class_source.split("\n")
        # Find class definition
        class_def_line = next(
            i for i, line in enumerate(lines) if line.strip().startswith("class ")
        )

        # Add evolution markers around the class body
        indent = len(lines[class_def_line]) - len(lines[class_def_line].lstrip())
        lines.insert(class_def_line + 1, " " * (indent + 4) + "# EVOLVE-BLOCK-START")
        lines.append(" " * (indent + 4) + "# EVOLVE-BLOCK-END")
        class_source = "\n".join(lines)

    # Create a self-contained evaluator as a code string so it works in subprocesses.
    import textwrap

    class_name = algorithm_class.__name__
    benchmark_source = textwrap.dedent(inspect.getsource(benchmark))

    evaluator_code = f"""
import importlib.util

CLASS_NAME = {class_name!r}

{benchmark_source}

def evaluate(program_path):
    '''Auto-generated evaluator for evolve_algorithm'''
    spec = importlib.util.spec_from_file_location("evolved", program_path)
    if spec is None or spec.loader is None:
        return {{"combined_score": 0.0, "score": 0.0, "error": "Failed to load program"}}

    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return {{"combined_score": 0.0, "score": 0.0, "error": f"Failed to execute program: {{str(e)}}"}}

    if not hasattr(module, CLASS_NAME):
        return {{"combined_score": 0.0, "score": 0.0, "error": f"Class '{{CLASS_NAME}}' not found"}}

    AlgorithmClass = getattr(module, CLASS_NAME)

    try:
        instance = AlgorithmClass()
        metrics = {benchmark.__name__}(instance)
        if not isinstance(metrics, dict):
            metrics = {{"score": metrics}}
        if "combined_score" not in metrics:
            metrics["combined_score"] = metrics.get("score", 0.0)
        return metrics
    except Exception as e:
        return {{"combined_score": 0.0, "score": 0.0, "error": str(e)}}
"""

    return run_evolution(
        initial_program=class_source, evaluator=evaluator_code, iterations=iterations, **kwargs
    )


def evolve_code(
    initial_code: str, evaluator: Callable[[str], Dict[str, Any]], iterations: int = 100, **kwargs
) -> EvolutionResult:
    """
    Evolve arbitrary code with a custom evaluator

    Args:
        initial_code: Initial code to evolve
        evaluator: Function that takes a program path and returns metrics
        iterations: Number of evolution iterations
        **kwargs: Additional arguments for run_evolution

    Returns:
        EvolutionResult with optimized code

    Example:
        initial_code = '''
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        '''

        def eval_fib(program_path):
            # Evaluate fibonacci implementation
            import importlib.util
            import time

            spec = importlib.util.spec_from_file_location("fib", program_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            try:
                start = time.time()
                result = module.fibonacci(20)
                duration = time.time() - start

                correct = result == 6765
                return {
                    "score": 1.0 if correct else 0.0,
                    "runtime": duration,
                    "correctness": correct
                }
            except:
                return {"score": 0.0}

        result = evolve_code(initial_code, eval_fib, iterations=50)
    """
    return run_evolution(
        initial_program=initial_code, evaluator=evaluator, iterations=iterations, **kwargs
    )
