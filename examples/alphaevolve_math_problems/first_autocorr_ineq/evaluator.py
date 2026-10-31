# ===--------------------------------------------------------------------------------------===#
#
# This file implements the evaluator for the first autocorrelation inequality problem.
#
# ===--------------------------------------------------------------------------------------===#
#
# Some of the code in this file is adapted from:
#
# google-deepmind/alphaevolve_results:
# Licensed under the Apache License v2.0.
#
# ===--------------------------------------------------------------------------------------===#

import sys
import os
from importlib import __import__
import time
import numpy as np

# known bounds
BENCHMARK = 1.5052939684401607


def verify_autocorrelation_solution(f_values: np.ndarray, c1_achieved: float, n_points: int):
    """Verify the autocorrelation solution for UPPER BOUND optimization"""

    # Check shape
    if f_values.shape != (n_points,):
        raise ValueError(f"Expected function values shape {(n_points,)}. Got {f_values.shape}.")

    if not np.isfinite(f_values).all():
        raise ValueError("Function values must be finite")

    # Check non-negativity
    if np.any(f_values < 0.0):
        raise ValueError("Function must be non-negative.")

    # Recompute C1 to verify
    dx = 0.5 / n_points
    f_nonneg = np.maximum(f_values, 0.0)

    # Compute the FULL autoconvolution
    autoconv = np.convolve(f_nonneg, f_nonneg, mode="full") * dx

    # The rest of the calculation can be simplified as we now take the max over the whole result
    integral_sq = (np.sum(f_nonneg) * dx) ** 2

    if integral_sq < 1e-8:
        raise ValueError("Function integral is too small.")

    # The max of the full autoconv is the correct value
    computed_c1 = float(np.max(autoconv / integral_sq))

    # Verify consistency
    delta = abs(computed_c1 - c1_achieved)
    if delta > 1e-6:
        raise ValueError(
            f"C1 mismatch: reported {c1_achieved:.6f}, computed {computed_c1:.6f}, delta: {delta:.6f}"
        )


def evaluate(program_path: str):
    try:
        abs_program_path = os.path.abspath(program_path)
        program_dir = os.path.dirname(abs_program_path)
        module_name = os.path.splitext(os.path.basename(program_path))[0]

        try:
            sys.path.insert(0, program_dir)
            program = __import__(module_name)
            start_time = time.time()
            f_values, c1_achieved, loss, n_points = program.run()
            end_time = time.time()
            eval_time = end_time - start_time
        except Exception as err:
            raise err
        finally:
            if program_dir in sys.path:
                sys.path.remove(program_dir)

        verify_autocorrelation_solution(f_values, c1_achieved, n_points)
        return {
            "c1": float(c1_achieved),
            "combined_score": BENCHMARK / float(c1_achieved),
            "loss": float(loss),
            "n_points": int(n_points),
            "eval_time": float(eval_time),
        }
    except Exception as e:
        return {"combined_score": 0.0, "error": str(e)}
