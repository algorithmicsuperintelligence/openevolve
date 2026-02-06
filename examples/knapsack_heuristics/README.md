# Knapsack Heuristics Evolution

This example aims to discover efficient heuristics for the classic 0/1 Knapsack problem.

## Problem Description

The [0/1 Knapsack problem](https://en.wikipedia.org/wiki/Knapsack_problem) involves selecting a subset of items, each with a weight and a value, such that the total weight does not exceed a given capacity and the total value is maximized.

While the problem is NP-hard, many heuristics exist. This example challenges the LLM to find novel heuristics that might outperform basic greedy approaches on various problem instances.

## Structure

- `initial_program.py`: A basic greedy heuristic that picks items based on their value-to-weight ratio.
- `evaluator.py`: Tests the heuristics on a suite of instances, measuring value achievement (`combined_score`) and constraint satisfaction (`correctness`).
- `config.yaml`: Configuration for the evolution process.

## How to Run

You can start the evolution from the root of the OpenEvolve repository:

```bash
python openevolve-run.py examples/knapsack_heuristics/initial_program.py \
  examples/knapsack_heuristics/evaluator.py \
  --config examples/knapsack_heuristics/config.yaml \
  --iterations 50
```

## Metrics

- `combined_score`: Normalized value achieved across all test instances. Solutions that exceed capacity for an instance receive zero score for that instance.
- `correctness`: The percentage of instances solved without exceeding the knapsack capacity.
