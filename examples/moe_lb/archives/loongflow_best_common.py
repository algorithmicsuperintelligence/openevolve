"""
best solution for general solver for moe load balance
"""

from typing import Tuple

import numpy as np


def solve_lplb_policy(
    initial_workloads: np.ndarray, physical_expert_placement: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Optimized load balancing solver using enhanced dynamic threshold with prioritized processing.

    Key Improvements:
    1. Weighted initialization (70/30 split) for better starting point
    2. Adaptive threshold with progressive tightening
    3. Proportional transfer limits for stability
    4. Optimized processing order
    5. Early termination conditions

    Args:
        initial_workloads: Shape (N,), initial token counts for N logical experts
        physical_expert_placement: Shape (ep_size, n_slots_per_rank), defines processing unit topology

    Returns:
        allocation_matrix: Shape (N, M), token allocation from experts to processing units
        minimized_max_load: Float, the minimized maximum load across all processing units
    """
    # Problem dimensions and flattened placement
    n_logical_experts = initial_workloads.shape[0]
    ep_size, n_slots_per_rank = physical_expert_placement.shape
    n_total_slots = ep_size * n_slots_per_rank
    flat_placement = physical_expert_placement.flatten()

    # Helper functions
    def calculate_gpu_loads(alloc_matrix):
        slot_loads = np.sum(alloc_matrix, axis=0)
        return slot_loads.reshape((ep_size, n_slots_per_rank)).sum(axis=1)

    def get_gpu_slots(gpu_idx):
        return range(gpu_idx * n_slots_per_rank, (gpu_idx + 1) * n_slots_per_rank)

    def calculate_safe_transfer(alloc_matrix, src_slot, dst_slot):
        src_load = np.sum(alloc_matrix[:, src_slot])
        dst_load = np.sum(alloc_matrix[:, dst_slot])
        mean_load = np.mean(calculate_gpu_loads(alloc_matrix))

        src_gpu = src_slot // n_slots_per_rank
        dst_gpu = dst_slot // n_slots_per_rank
        gpu_loads = calculate_gpu_loads(alloc_matrix)

        # Calculate maximum safe transfer amount
        transfer = min(
            src_load * 0.3,  # Max 30% of source slot's load
            (gpu_loads[src_gpu] - gpu_loads[dst_gpu]) * 0.4,  # Proportional to imbalance
            mean_load - dst_load,  # Don't overfill target
        )
        return max(transfer, 0)

    def execute_transfer(alloc_matrix, expert, src_slot, dst_slot, amount):
        alloc_matrix[expert, src_slot] -= amount
        alloc_matrix[expert, dst_slot] += amount

    # Enhanced Initialization Phase (70/30 weighted distribution)
    allocation_matrix = np.zeros((n_logical_experts, n_total_slots))

    for expert_id in range(n_logical_experts):
        replica_indices = np.where(flat_placement == expert_id)[0]
        if len(replica_indices) > 0:
            # Get current loads of replicas
            replica_loads = np.sum(allocation_matrix[:, replica_indices], axis=0)

            # 70% to least loaded replica, 30% distributed among others
            primary_replica = replica_indices[np.argmin(replica_loads)]
            allocation_matrix[expert_id, primary_replica] = initial_workloads[expert_id] * 0.7
            remaining = initial_workloads[expert_id] * 0.3

            if len(replica_indices) > 1:
                allocation_matrix[expert_id, replica_indices] += remaining / len(replica_indices)
            else:
                allocation_matrix[expert_id, primary_replica] += remaining

    # Track best solution
    best_allocation = allocation_matrix.copy()
    best_Z = np.max(calculate_gpu_loads(best_allocation))

    # Adaptive Iterative Refinement
    max_iterations = 100
    for iteration in range(max_iterations):
        # Calculate current loads
        slot_loads = np.sum(allocation_matrix, axis=0)
        gpu_loads = calculate_gpu_loads(allocation_matrix)
        current_Z = np.max(gpu_loads)

        # Update best solution
        if current_Z < best_Z:
            best_Z = current_Z
            best_allocation = allocation_matrix.copy()

        # Early termination if balanced
        if np.max(gpu_loads) - np.min(gpu_loads) < 1e-6:
            break

        # Adaptive threshold calculation
        mean_load = np.mean(gpu_loads)
        std_load = np.std(gpu_loads)
        threshold = mean_load + (0.8 - 0.7 * (iteration / max_iterations)) * std_load

        # Process overloaded GPUs in descending order
        overloaded_gpus = np.where(gpu_loads > threshold)[0]
        for gpu in sorted(overloaded_gpus, key=lambda x: -gpu_loads[x]):
            # Process slots in this GPU by descending load
            slots = get_gpu_slots(gpu)
            for slot in sorted(slots, key=lambda x: -slot_loads[x]):
                # Get contributing experts sorted by contribution
                contributing_experts = np.where(allocation_matrix[:, slot] > 1e-6)[0]
                for expert in sorted(
                    contributing_experts, key=lambda x: -allocation_matrix[x, slot]
                ):
                    # Get all valid target replicas sorted by load
                    replica_indices = np.where(flat_placement == expert)[0]
                    target_replicas = sorted(
                        replica_indices, key=lambda x: np.sum(allocation_matrix[:, x])
                    )

                    for target in target_replicas:
                        if target == slot:
                            continue

                        transfer = calculate_safe_transfer(allocation_matrix, slot, target)
                        if transfer > 1e-6:
                            execute_transfer(allocation_matrix, expert, slot, target, transfer)

                            # Update slot and GPU loads
                            slot_loads[slot] -= transfer
                            slot_loads[target] += transfer
                            src_gpu = slot // n_slots_per_rank
                            dst_gpu = target // n_slots_per_rank
                            gpu_loads[src_gpu] -= transfer
                            gpu_loads[dst_gpu] += transfer

                            # Early exit if we've balanced this slot
                            if allocation_matrix[expert, slot] < 1e-6:
                                break

    # Final validation and cleanup
    allocation_matrix = best_allocation

    # Fix floating point discrepancies
    for expert_id in range(n_logical_experts):
        total = np.sum(allocation_matrix[expert_id])
        if not np.isclose(total, initial_workloads[expert_id]):
            diff = initial_workloads[expert_id] - total
            replica_indices = np.where(flat_placement == expert_id)[0]
            if len(replica_indices) > 0:
                allocation_matrix[expert_id, replica_indices[0]] += diff

    # Calculate final maximum load
    final_gpu_loads = calculate_gpu_loads(allocation_matrix)
    minimized_max_load = np.max(final_gpu_loads) if final_gpu_loads.size > 0 else 0.0

    return allocation_matrix, float(minimized_max_load)
