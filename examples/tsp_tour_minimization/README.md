# Traveling Salesman Problem (TSP) Mean Tour Length Minimization Example

This example demonstrates how OpenEvolve can improve large optimization algorithms starting from current state-of-the-art (SOTA) implementations. It also illustrates evolution via changes description, which is crucial when a single program spans thousands of lines and mixes multiple programming languages.

## Problem Description

We solve a 2D Euclidean Traveling Salesman Problem (TSP) variant: given **1000 cities** (points) on a plane, the goal is to find a **Hamiltonian cycle** with minimal total Euclidean length.  
For each evaluation instance, cities are sampled **i.i.d. uniformly** from the unit square **[0, 1] × [0, 1]**.

## Evaluation Protocol

Programs are evaluated on a **fixed benchmark set of 128 instances**. To ensure fair and comparable results across evolutionary iterations, these 128 instances are **generated once upfront** and then reused for every candidate program (i.e., the evaluation set does not change between programs).  
The reported score (e.g., mean tour length) is computed by running the solver on all 128 instances and aggregating the results.

The C++ part is compiled using C++ 17 standart via command:
```bash
g++ -std=gnu++17 -O3 -DNDEBUG -march=native -funroll-loops -ffast-math -Iinclude TSP.cpp -o bin/runner -lpthread -lm -ldl
```

## Getting Started

To run this example:

```bash
cd examples/tsp_tour_minimization
pip install -r requirements.txt
python start_evolution.py --initial_program_dir initial_program --openevolve_output_dir openevolve_output
```

## Algorithm Evolution

### 🌱 Initial Algorithm (from original UTSP paper)

The baseline solver follows the standard UTSP workflow (using slightly adjusted code from their [github](https://github.com/yimengmin/UTSP), but without using heat map training/inference):

- Builds a **KNN candidate list** for each city.
- For each restart:
  1) Generates a **random Hamiltonian cycle**.
  2) Runs **local 2-opt** until no improving move exists (restricted to candidate edges).
  3) Runs **local k-opt** (MCTS-like sampling guided by an edge “potential” derived from learned weights), updating weights from improvements.
- Runs a **fixed number of restarts** (e.g., 200) and outputs the best tour found.

### 🧬 Evolved Algorithm

The evolved solver kept the UTSP core (KNN candidates + 2-opt + k-opt with learned weights) but changed how compute is spent.

## 🚀 Key Improvements

Through evolutionary iterations, OpenEvolve discovered several key optimization concepts:

1. **Time-Capped Restarts**: Replaced a “fixed small restart count” with a hard ~159s budget, enabling far more restart attempts and letting the solver scale restarts to available time.

2. **Hybrid Restart Seeding**: Stopped restarting with random solution and instead started using cheap greedy constructions for better initial tours, while periodically perturbing (“shaking”) the current best solution to escape local minima while staying in a strong basin (Iterated Local Search / VNS behavior).

3. **Time-Aware Scheduling**: Switched restart strategy based on elapsed time and progress: early iterations favor diversity (random/greedy mix), while later iterations increasingly reuse or perturbe the best tour to intensify search where it mattered.

4. **Selective Compute Allocation**: Runs expensive k-opt only when the current tour is already close to the best (quality-ratio thresholds). Bad restarts receive only cheap improvement (2-opt), improving the time/quality trade-off.

5. **Remove O(n²) Work from k-opt**: Eliminated per-call full rebuild of `total_weight` in local k-opt by maintaining it incrementally in `update_weight_undirected(...)`, unlocking many more useful search steps within the same time budget.

6. **k-opt Breadth-over-Depth Rebalance**: Reduced k-opt depth and simulations per restart, and reinvested that compute into more restarts. (More basins are explored plus targeted intensification typically outperformes deep search on weak starting tours.)

7. **Stabilize k-opt Sampling via Potential/Threshold Tuning**: Lowered `min_potential_to_consider` and introduced a small exploration term so candidate sampling did not collapse onto a tiny set of reinforced edges, reducing stagnation and improving late-stage refinement.

8. **Faster/Stronger Weight Learning**: Increased `weight_delta_coefficient` to reinforce successful edges more aggressively, and enabled sensitivity-based decay to reduce noisy credit assignment along deep, unproductive chains.

9. **Higher Integer Precision for Move Decisions**: Increased `magnify_rate` under `int64` distances to reduce rounding artifacts in delta computations, which helped especially near the optimum where true improvements were small.

10. **Final Best-Solution Refinement Stage**: Added a small “use remaining time” intensification step that attempts additional local improvements on the best tour found.

## 📊 Results

The evolved algorithm shows substantial improvement in finding better solutions:

| Metric | Concorde (exact solver) | UTSP | UTSP (evolved) |
|--------|-------|-------|-------|
| Mean Tour Length | 23.12 | 23.39 | 23.30 |
| Average Execution Time | 6.65h | 3.35m | 28.27m |

Larger comparison table with more related methods can be found in the original UTSP paper: [arXiv](https://arxiv.org/abs/2303.10538).

In total, we achieved a $33\%$ reduction in mean tour length (relative to the gap between the exact (Concorde) solution and UTSP).

Chart depicting mean tour length metric improvement through evolution (where yellow dashed line represents UTSP algorithm metric from the original paper):
![metric_improvement_through_evolution_process](pictures/metric_improvement_through_evolution_process.png)

Comment: The initial metric does not match the UTSP baseline because we deliberately removed heat map utilization from the initial program to see what can be achieved without it. However, we kept `heat_map_train.py`, `heat_map_inference.py`, and created the corresponding evaluators and metrics for future use. Another thing that you might have noticed is that the mean tour length sometimes increases. That is expected because the optimizer targets a combined score, not tour length alone. The combined score includes average tour length, tour length variance, and average execution time.

## ⚙️ How It Works

This example demonstrates how to evolve large projects like UTSP using OpenEvolve across thousands of evolution iterations:

- **Project as a Single File**: Source code of a large project can be represented as a single carefully structured .txt file.
- **Changes Description as a Program**: The system requires the LLM to describe the changes it makes. This lets subsequent prompts represent the program not as full source code, but as a compact, explicit change description that captures the key edits. As a result, the prompt uses far fewer input tokens — reducing cost and letting the LLM focus on the most relevant context, which typically improves output quality and enables more iterations within the same budget.

## 📁 Complete File Structure

```
tsp_tour_minimization/
├── .gitignore
├── README.md
├── config.yaml
├── evaluator.py
├── evolved_program
│   ├── TSP.cpp
│   ├── config.json
│   ├── heat_map_inference.py
│   ├── heat_map_train.py
│   └── include
│       ├── additional.hpp
│       ├── context.hpp
│       ├── json.hpp
│       ├── local_2_opt_search.hpp
│       ├── local_k_opt_search.hpp
│       ├── random_solution.hpp
│       └── utils.hpp
├── initial_program
│   ├── TSP.cpp
│   ├── config.json
│   ├── heat_map_inference.py
│   ├── heat_map_train.py
│   └── include
│       ├── additional.hpp
│       ├── context.hpp
│       ├── json.hpp
│       ├── local_2_opt_search.hpp
│       ├── local_k_opt_search.hpp
│       ├── random_solution.hpp
│       └── utils.hpp
├── pictures
│   └── metric_improvement_through_evolution_process.png
├── requirements.txt
├── start_evolution.py
└── utils
    ├── code_to_query.py
    ├── heat_map_runner.py
    ├── load_data.py
    ├── runner.py
    ├── tsp_runner.py
    └── utils.py
```

## 🔍 Next Steps

Two promising components exist in the evolved codebase but appear to be unused (or at least not wired into the main execution path):

- **`evaluate_candidate_chain(...)`**: a Lin–Kernighan-style chain evaluation heuristic that could serve as a fast “should we try a deeper move here?” oracle, or as a lightweight intensification step when a tour is near-best. It’s interesting to benchmark what metric improvement it can deliver compared to the current selective k-opt gating.

- **`identify_candidates_lk_style(...)`**: an alternative candidate selection strategy that blends nearest neighbors with spatially diverse candidates. This could change the neighborhood structure significantly and might improve the solver’s ability to escape local minima (at some compute cost).

It’s likely these functions are currently not used because of how the evolution system applies edits (e.g., search/replace blocks): the functions were added successfully, but the final step — calling them from the right place — may have failed silently (if LLM responded in invalid diff format, for instance) or been overwritten by other patches. This should be carefully verified by inspecting the evolved code paths, then studying their effect with controlled A/B runs under the same time budget.

## 📚 References

- [OpenEvolve Documentation](../../README.md)
- [UTSP Paper](https://arxiv.org/abs/2303.10538)
