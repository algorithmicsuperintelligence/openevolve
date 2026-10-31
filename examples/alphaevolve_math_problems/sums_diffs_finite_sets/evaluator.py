# ===--------------------------------------------------------------------------------------===#
#
# This file implements the evaluator for the sums and differences of finite sets problem.
#
# ===--------------------------------------------------------------------------------------===#
#
# Some of the code in this file is adapted from:
#
# google-deepmind/alphaevolve_results:
# Licensed under the Apache License v2.0.
#
# ===--------------------------------------------------------------------------------------===#

import math
import os
import sys
import time
from importlib import __import__

import numpy as np

BENCHMARK = 1.158417281556896


def verify_c6_solution(u_set: np.ndarray, c6_achieved: float):
    """Verifies the C6 lower bound solution."""

    if not isinstance(u_set, np.ndarray) or u_set.ndim != 1:
        raise ValueError("Solution U must be a 1D numpy array of integers.")

    try:
        integer_values = [int(value) for value in u_set]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Set U must contain finite integers.") from exc
    if any(value != integer for value, integer in zip(u_set, integer_values)):
        raise ValueError("Set U must contain finite integers.")

    # Use Python integers so sums, differences, and 2 * max(U) + 1 cannot wrap.
    u_set = np.array(integer_values, dtype=object)

    # Verify constraints
    if 0 not in u_set:
        raise ValueError("Set U must contain 0.")
    if np.any(u_set < 0):
        raise ValueError("Set U must contain non-negative integers.")

    # Re-calculate the C6 bound using NumPy
    u_plus_u = np.unique(u_set[:, None] + u_set[None, :])
    u_minus_u = np.unique(u_set[:, None] - u_set[None, :])

    size_U_plus_U = len(u_plus_u)
    size_U_minus_U = len(u_minus_u)
    max_U = int(np.max(u_set))
    if max_U == 0:
        raise ValueError("Set U must contain a positive integer.")

    ratio = size_U_minus_U / size_U_plus_U
    log_ratio = np.log(ratio)
    log_denom = math.log(2 * max_U + 1)

    computed_c6 = 1 + log_ratio / log_denom

    # Check for consistency
    if not np.isclose(computed_c6, c6_achieved):
        raise ValueError(f"C6 mismatch: reported {c6_achieved:.6f}, computed {computed_c6:.6f}")

    print(f"C6 lower bound achieved: {c6_achieved:.6f}")
    print(f"Known best bound (AlphaEvolve): {BENCHMARK}")

    if c6_achieved > BENCHMARK:
        print("Successfully found a new, better lower bound!")
    else:
        print("Result is not better than the known lower bounds.")


def evaluate(program_path: str):
    try:
        abs_program_path = os.path.abspath(program_path)
        program_dir = os.path.dirname(abs_program_path)
        module_name = os.path.splitext(os.path.basename(program_path))[0]

        try:
            sys.path.insert(0, program_dir)
            program = __import__(module_name)
            start_time = time.time()
            u_set, c6_bound = program.run()
            end_time = time.time()
            eval_time = end_time - start_time
        finally:
            if program_dir in sys.path:
                sys.path.remove(program_dir)

        verify_c6_solution(u_set, c6_bound)

        return {
            "c6_bound": float(c6_bound),
            "combined_score": float(c6_bound) / BENCHMARK,
            "set_size": len(u_set),
            "max_val": int(np.max(u_set)),
            "eval_time": float(eval_time),
        }
    except Exception as e:
        return {"combined_score": 0.0, "error": str(e)}
