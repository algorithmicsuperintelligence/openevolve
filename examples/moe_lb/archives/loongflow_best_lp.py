"""
best solution for liner programming solver for moe load balance
"""

from typing import Tuple

import numpy as np
from scipy.optimize import linprog


def solve_lplb_policy(
    initial_workloads: np.ndarray, physical_expert_placement: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Solves the load balancing problem using linear programming to minimize the maximum GPU load.

    Args:
        initial_workloads: Array of shape (N,) with workload for each logical expert.
        physical_expert_placement: Array of shape (K, S) mapping experts to processing units.

    Returns:
        Tuple containing:
        - allocation_matrix: Array of shape (N, M) with optimal token allocations
        - minimized_max_load: The minimized maximum GPU load
    """
    N = initial_workloads.shape[0]  # Number of logical experts
    K, S = physical_expert_placement.shape  # K GPUs, S slots per GPU
    M = K * S  # Total processing units

    # Flatten the placement matrix to get processing unit assignments
    flat_placement = physical_expert_placement.flatten()

    # Precompute valid (i,j) pairs where expert i can be assigned to unit j
    valid_pairs = []
    var_indices = {}  # Maps (i,j) to variable index
    var_count = 0

    for i in range(N):
        for j in range(M):
            if flat_placement[j] == i:
                valid_pairs.append((i, j))
                var_indices[(i, j)] = var_count
                var_count += 1

    # Number of variables is the number of valid (i,j) pairs
    num_vars = len(valid_pairs)

    # Construct the LP problem:
    # Objective: minimize Z (the maximum load)
    # We'll represent Z as the last variable (index num_vars)
    c = np.zeros(num_vars + 1)
    c[-1] = 1  # Minimize Z

    # Constraints:
    # 1. GPU load constraints (Z >= sum of loads for each GPU)
    # 2. Flow conservation (sum of allocations for each expert = initial workload)
    # 3. Non-negativity is handled by bounds

    # GPU load constraints: one per GPU
    A_ub = np.zeros((K, num_vars + 1))
    b_ub = np.zeros(K)

    for k in range(K):
        # Get all slots in this GPU
        slots_in_gpu = range(k * S, (k + 1) * S)
        # Find all variables that contribute to this GPU's load
        for (i, j), var_idx in var_indices.items():
            if j in slots_in_gpu:
                A_ub[k, var_idx] = 1
        A_ub[k, -1] = -1  # -Z term

    # Flow conservation constraints: one per expert
    A_eq = np.zeros((N, num_vars + 1))
    b_eq = initial_workloads.copy()

    for i in range(N):
        for (expert_i, j), var_idx in var_indices.items():
            if expert_i == i:
                A_eq[i, var_idx] = 1

    # Bounds: all x_ij >= 0, Z is unbounded
    bounds = [(0, None) for _ in range(num_vars)] + [(None, None)]

    # Solve the LP problem
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not result.success:
        raise RuntimeError(f"LP solver failed: {result.message}")

    # Extract solution
    solution = result.x
    minimized_max_load = solution[-1]

    # Build the allocation matrix
    allocation_matrix = np.zeros((N, M))
    for (i, j), var_idx in var_indices.items():
        allocation_matrix[i, j] = solution[var_idx]

    return allocation_matrix, float(minimized_max_load)
