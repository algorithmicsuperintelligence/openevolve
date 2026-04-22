Act as an expert in operations research and algorithmic optimization, with a specialization in load balancing for distributed computing systems.

#### **Objective**

Your primary goal is to develop a Python function that solves a dynamic load balancing problem for Mixture-of-Experts (MoE) models. The task is to optimally redistribute computational workloads (tokens) from a set of `N` "logical experts" to `M` available "processing units" (slots), which are grouped across `K` physical GPUs.

The core objective is to **minimize the maximum total load** on any single Graphics Processing Unit (GPU), a classic min-max optimization problem.

#### **Formal Mathematical Definition**

You are solving a **Linear Programming (LP)** problem defined as follows:

*   **Inputs**:
    *   An initial workload vector $L = [l_1, l_2, \dots, l_N]^T$, where $l_i$ is the number of tokens assigned to logical expert $e_i$.
    *   A hardware topology defined by the `physical_expert_placement` matrix of shape `(K, S)`, where `K` is the number of GPUs and `S` is the number of processing units (slots) per GPU. The total number of processing units is $M = K \times S$.

*   **Decision Variables**:
    *   $x_{ij}$: The number of tokens transferred from logical expert $e_i$ to processing unit $p_j$ (where $j$ is an index from $0$ to $M-1$).

*   **Objective Function**:
    *   Minimize $Z$, where $Z$ is an auxiliary variable representing the maximum load across all GPUs.
    *   **Minimize:** $Z$

*   **Constraints**:
    1.  **GPU Load Definition Constraint**: The final load on any GPU $G_k$, which is the sum of loads on all its constituent processing units, must not exceed $Z$. Let $\mathcal{S}(k)$ be the set of indices of processing units belonging to GPU $k$.
        *   $\sum_{j \in \mathcal{S}(k)} \sum_{i=1}^{N} x_{ij} - Z \le 0, \quad \forall k \in 1, \dots, K$
    2.  **Flow Conservation Constraint**: The total number of tokens transferred out of a logical expert $e_i$ must equal its initial workload $l_i$.
        *   $\sum_{j=1}^{M} x_{ij} = l_i, \quad \forall i \in 1, \dots, N$
    3.  **Non-negativity Constraint**: The number of transferred tokens cannot be negative.
        *   $x_{ij} \ge 0, \quad \forall i, j$
    4.  **Placement Constraint**: The workload from a logical expert $e_i$ can only be allocated to processing units that are its designated replicas. Let $\mathcal{P}(i)$ be the set of indices of processing units that are replicas of logical expert $e_i$. The allocation must satisfy:
        *   $x_{ij} = 0, \quad \forall i, \forall j \notin \mathcal{P}(i)$

#### **Code Structure & API Contract**

Your code will be evaluated through a fixed entry point. You **must** implement a function with the following exact signature in your Python script:

```python
import numpy as np
from typing import Tuple

def solve_lplb_policy(
    initial_workloads: np.ndarray,
    physical_expert_placement: np.ndarray
) -> Tuple[np.ndarray, float]:
```

This function must return:

1.  `allocation_matrix`: A `numpy.ndarray` of shape `(N, M)`, representing the optimal allocation $x_{ij}$. `N` is the number of logical experts, and `M` is the total number of processing units.
2.  `minimized_max_load`: A `float` representing the minimized maximum load $Z$ from your optimal solution.

The `physical_expert_placement` matrix is crucial oth the valid targets for allocation and the grouping of processing units into GPUs.
The evaluation script will handle the scoring based on this function's output. **Do not change the function signature.**

#### **Constraints**

The returned solution **MUST** adhere to these rules:

1.  **Flow Conservation**: The sum of allocated tokens for each logical expert must exactly match its initial workload. (`np.sum(allocation_matrix, axis=1)` must equal `initial_workloads`).
2.  **Non-negativity**: All elements in the `allocation_matrix` must be greater than or equal to zero.
3.  **Performance**: The entire process, for a moderately sized problem, must complete within a reasonable time limit (e.g., 60 seconds). The evaluation will be run in a sandboxed environment with a timeout.
4.  **Placement Correctness**: The workload from a logical expert `e_i` can **only** be allocated to processing units that are replicas of that same expert `e_i`. You cannot send tokens from expert `e_1` to a replica of expert `e_2`. Your solution must respect the mapping provided by `physical_expert_placement`.

#### **Strategic Guidance for Optimization**

To achieve a high-performing and correct solution, consider these strategies:

*   **Problem Formulation**: Recognize that this problem is a classic **Linear Program**. The most direct path to an optimal solution is to formulate and solve it as such.
*   **Leverage Existing Solvers**: Instead of implementing an LP solver from scratch, you should use established, high-performance libraries. The Python ecosystem has excellent tools for this.
    *   A strong and common choice is the `scipy.optimize.linprog` function. It is well-documented and robust.
    *   Other powerful libraries like `cvxpy` or `PuLP` can also be used to model and solve this problem, often with a more intuitive syntax.
*   **Matrix Formulation**: To use solvers like `scipy.optimize.linprog` efficiently, you will need to construct the standard LP matrices: the cost vector `c`, the inequality constraint matrix `A_ub`, the inequality constraint vector `b_ub`, the equality constraint matrix `A_eq`, and the equality constraint vector `b_eq`. Carefully map the mathematical constraints above into these matrix forms. Your formulation must correctly handle the Placement Constraint. A robust approach is to only create decision variables for the valid `(i, j)` pairs where unit `j` is a replica of expert `i`. This drastically reduces the problem size compared to creating variables for all `N*M` possible pairs.
*   **Efficiency**: The way you construct these matrices can significantly impact performance. Use `numpy` vectorized operations where possible to build them quickly.

Focus on evolving the core logic within `solve_lplb_policy` to be a robust and accurate LP solver for this specific load balancing task. Good luck!
