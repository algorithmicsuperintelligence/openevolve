# ===--------------------------------------------------------------------------------------===#
#
# This file implements the evaluator for the matrix multiplication problem with tensor size
# of <2,4,5>
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

BENCHMARK = 32

# Fixed target tensor dimensions for this example: <n, m, p> = <2, 4, 5>.
# The evaluator must check the dimensions a candidate reports against these:
# the reported values are produced inside the EVOLVE-BLOCK and must not be
# trusted (see https://github.com/algorithmicsuperintelligence/openevolve/issues/514).
TARGET_N, TARGET_M, TARGET_P = 2, 4, 5


def verify_tensor_decomposition(
    decomposition: tuple[np.ndarray, np.ndarray, np.ndarray], n: int, m: int, p: int, rank: int
):
    """Verifies the correctness of the tensor decomposition."""

    # Add robustness for cases where the optimizer might fail
    if not all(isinstance(arr, np.ndarray) for arr in decomposition) or not decomposition:
        raise ValueError("Decomposition must be a tuple of NumPy arrays.")
    if any(arr.size == 0 for arr in decomposition):
        # An empty factor matrix can never reproduce the target tensor, so this
        # is a failed verification -- not something to silently skip. Skipping
        # would let a candidate score without solving the task (see #514).
        raise ValueError("Verification failed: one or more decomposition arrays are empty.")

    # Check that each factor matrix has the correct shape.
    factor_matrix_1, factor_matrix_2, factor_matrix_3 = decomposition
    if factor_matrix_1.shape != (n * m, rank):
        raise ValueError(
            f"Expected shape of factor matrix 1 is {(n * m, rank)}. Actual shape is {factor_matrix_1.shape}."
        )
    if factor_matrix_2.shape != (m * p, rank):
        raise ValueError(
            f"Expected shape of factor matrix 2 is {(m * p, rank)}. Actual shape is {factor_matrix_2.shape}."
        )
    if factor_matrix_3.shape != (n * p, rank):
        raise ValueError(
            f"Expected shape of factor matrix 3 is {(n * p, rank)}. Actual shape is {factor_matrix_3.shape}."
        )

    # Form the matrix multiplication tensor <n, m, p>.
    matmul_tensor = np.zeros((n * m, m * p, n * p), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Use the standard k*n+i indexing for the third dimension
                matmul_tensor[i * m + j, j * p + k, k * n + i] = 1

    # Check that the tensor is correctly constructed.
    constructed_tensor = np.einsum("ir,jr,kr -> ijk", *decomposition)

    # Exact check
    if not np.array_equal(constructed_tensor, matmul_tensor):
        # If the exact check fails, report the floating-point difference for diagnostics.
        diff = np.max(np.abs(constructed_tensor - matmul_tensor))
        raise ValueError(
            f"Tensor constructed by decomposition does not exactly match the target tensor. Maximum difference is {diff:.6e}."
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
            # NOTE: the reported n, m, p and rank come from the EVOLVE-BLOCK and
            # are deliberately not trusted below (see #514).
            decomposition, n, m, p, loss, _reported_rank = program.run()
            end_time = time.time()
            eval_time = end_time - start_time
        except Exception as err:
            raise err
        finally:
            if program_dir in sys.path:
                sys.path.remove(program_dir)

        # The candidate must solve the fixed <2, 4, 5> problem: reject any
        # reported dimensions that do not match the target.
        if (n, m, p) != (TARGET_N, TARGET_M, TARGET_P):
            raise ValueError(
                f"Candidate reported dimensions (n, m, p) = {(n, m, p)}; "
                f"expected {(TARGET_N, TARGET_M, TARGET_P)}."
            )

        # Derive the rank from the actual factor matrices instead of trusting
        # the candidate-reported value: a candidate could otherwise claim an
        # arbitrarily low rank and score above the state of the art.
        try:
            rank = int(decomposition[0].shape[1])
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError(f"Could not determine rank from the decomposition: {exc}")

        verify_tensor_decomposition(decomposition, n, m, p, rank)

        success_threshold = 1e-6
        if loss > success_threshold:
            print(
                f"\nWarning: Final loss {loss:.2e} is above the success threshold of {success_threshold:.2e}."
            )

        inverse_rank = BENCHMARK / rank

        return {
            "combined_score": inverse_rank,
            "loss": loss,
            "rank": rank,
            "eval_time": float(eval_time),
        }
    except Exception as e:
        return {"combined_score": 0.0, "error": str(e)}
