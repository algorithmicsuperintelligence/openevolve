# ===--------------------------------------------------------------------------------------===#
#
# This file implements the evaluator for the second autocorrelation inequality problem.
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

BENCHMARK = 0.8962799441554086


def verify_c2_solution(f_values: np.ndarray, c2_achieved_from_opt: float, n_points: int):
    """
    Verifies the C2 lower bound solution using the rigorous, unitless, piecewise linear integral method.
    """
    if f_values.shape != (n_points,):
        raise ValueError(f"Expected function values shape {(n_points,)}. Got {f_values.shape}.")

    if not np.isfinite(f_values).all():
        raise ValueError("Function values must be finite.")

    if np.any(f_values < -1e-6):  # Allow for small floating point errors
        raise ValueError("Function must be non-negative.")

    f_nonneg = np.maximum(f_values, 0.0)
    # The raw, unscaled convolution is used
    convolution = np.convolve(f_nonneg, f_nonneg, mode="full")

    # Calculate the L2-norm squared: ||f*f||_2^2 via piecewise linear integration
    num_conv_points = len(convolution)
    x_points = np.linspace(-0.5, 0.5, num_conv_points + 2)
    x_intervals = np.diff(x_points)
    y_points = np.concatenate(([0], convolution, [0]))
    l2_norm_squared = 0.0
    for i in range(len(convolution) + 1):
        y1, y2, h = y_points[i], y_points[i + 1], x_intervals[i]
        interval_l2_squared = (h / 3) * (y1**2 + y1 * y2 + y2**2)
        l2_norm_squared += interval_l2_squared

    # Calculate the L1-norm: ||f*f||_1
    # This is an approximation of the integral of the absolute value of the autoconvolution
    norm_1 = np.sum(np.abs(convolution)) / (len(convolution) + 1)

    # Calculate the infinity-norm: ||f*f||_inf
    norm_inf = np.max(np.abs(convolution))

    computed_c2 = l2_norm_squared / (norm_1 * norm_inf)

    return computed_c2


def evaluate(program_path: str):
    try:
        abs_program_path = os.path.abspath(program_path)
        program_dir = os.path.dirname(abs_program_path)
        module_name = os.path.splitext(os.path.basename(program_path))[0]

        try:
            sys.path.insert(0, program_dir)
            program = __import__(module_name)
            start_time = time.time()
            f_values, c2_achieved_from_opt, loss, n_points = program.run()
            end_time = time.time()
            eval_time = end_time - start_time
        finally:
            if program_dir in sys.path:
                sys.path.remove(program_dir)

        c2_verified = verify_c2_solution(f_values, c2_achieved_from_opt, n_points)

        return {
            "c2": float(c2_verified),
            "combined_score": float(c2_verified) / BENCHMARK,
            "loss": float(loss),
            "n_points": int(n_points),
            "eval_time": float(eval_time),
        }
    except Exception as e:
        return {"combined_score": 0.0, "error": str(e)}
