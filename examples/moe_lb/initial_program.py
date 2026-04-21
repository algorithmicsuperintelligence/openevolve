# EVOLVE-BLOCK-START
"""moe lb solver"""

import numpy as np


def solve_lplb_policy(
    initial_workloads: np.ndarray, physical_expert_placement: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Calculate and return optimal load allocation matrix & minimized maximum load from initial load and expert deployment

    Args:
        initial_workloads (np.ndarray): Shape (N,) 1D array, N is the number of logical experts.
                                        Represent the initial load of each logical expert
        physical_expert_placement (np.ndarray): Shape (ep_size, n_slots_per_rank) 2D array
                                                Define the topology of M processing units

    Returns:
        Tuple[np.ndarray, float]:
        - allocation_matrix (np.ndarray): Shape(N, M) 2D array, M is the total number of processing units.
                                          Represents the allocation amount x_ij from expert i to processing unit j.
        - minimized_max_load (float): Minimized maximum load Z.
    """

    n_logical_experts = initial_workloads.shape[0]
    ep_size, n_slots_per_rank = physical_expert_placement.shape
    n_total_slots = ep_size * n_slots_per_rank  # M, Total number of processing units

    # 1. Calculate the number of replicas for each logical expert
    # np.bincount efficiently counts occurrences of each expert ID (0, 1, ..., N-1)
    flat_placement = physical_expert_placement.flatten()
    replica_counts = np.bincount(flat_placement, minlength=n_logical_experts)

    # 2. Calculate allocation matrix x_ij
    allocation_matrix = np.zeros((n_logical_experts, n_total_slots))
    for expert_id in range(n_logical_experts):
        workload = initial_workloads[expert_id]
        num_replicas = replica_counts[expert_id]

        if num_replicas > 0:
            workload_per_replica = workload / num_replicas
            # Find the linear indices of all replicas of this expert
            replica_indices = np.where(flat_placement == expert_id)[0]
            # Fill the allocation matrix with the evenly distributed load
            allocation_matrix[expert_id, replica_indices] = workload_per_replica

    # 3. Calculate final GPU loads and find the maximum load Z
    # First, calculate the total load of each processing unit (slot)
    final_slot_loads = np.sum(allocation_matrix, axis=0)
    # Then, aggregate the slot loads by GPU
    gpu_loads = final_slot_loads.reshape((ep_size, n_slots_per_rank)).sum(axis=1)
    # Finally, find the maximum load among all GPUs
    minimized_max_load = np.max(gpu_loads) if gpu_loads.size > 0 else 0.0

    return allocation_matrix, float(minimized_max_load)


# EVOLVE-BLOCK-END
