# Circle Packing Example

This example demonstrates how OpenEvolve can be used to tackle the challenging mathematical problem of circle packing, a classic problem in computational geometry. Specifically, we focus on packing 26 circles of varying sizes into a unit square to maximize the sum of their radii, replicating one of the tasks from the AlphaEvolve paper.

## Problem Overview

The circle packing problem involves placing n non-overlapping circles inside a container (in this case, a unit square) to optimize a specific metric. For this example:

- We pack exactly 26 circles
- Each circle must lie entirely within the unit square
- No circles may overlap
- We aim to maximize the sum of all circle radii

According to the AlphaEvolve paper, a solution with a sum of radii of approximately 2.635 is achievable for n=26. Our goal was to match or exceed this result.

## Our Approach

We structured our evolution in two phases, each with a different configuration to encourage exploration and exploitation at different stages:

### Phase 1: Initial Exploration

In the first phase, we focused on exploring different fundamental approaches to the packing problem:

- Used a constructor-based approach that places circles in strategic positions
- Explored various geometric patterns (concentric rings, grid-based arrangements, etc.)
- Developed simple optimization routines to maximize circle sizes without overlaps

Configuration highlights:
```yaml
max_iterations: 100
population_size: 60  
num_islands: 4
exploitation_ratio: 0.7
```

### Phase 2: Breaking Through the Plateau

After the initial exploration phase, we observed our solutions plateauing around 2.377. For the second phase, we reconfigured OpenEvolve to encourage more radical innovations:

- Increased the population size to promote diversity
- Lowered the exploitation ratio to favor exploration
- Updated the system prompt to suggest different optimization techniques
- Allowed for longer and more complex code solutions

Configuration highlights:
```yaml
max_iterations: 100
population_size: 70  
num_islands: 5
exploitation_ratio: 0.6
```

## Evolution Progress

We tracked the evolution over 470 generations, capturing visualizations at each checkpoint. The progression shows dramatic improvements in the packing strategy:

### Initial Solution (Generation 0)

The initial program used a simple constructive approach with a central circle and two concentric rings:

```python
# Initial attempt
# Place a large circle in the center
centers[0] = [0.5, 0.5]

# Place 8 circles around it in a ring
for i in range(8):
    angle = 2 * np.pi * i / 8
    centers[i + 1] = [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)]

# Place 16 more circles in an outer ring
for i in range(16):
    angle = 2 * np.pi * i / 16
    centers[i + 9] = [0.5 + 0.7 * np.cos(angle), 0.5 + 0.7 * np.sin(angle)]
```

This approach yielded a sum of radii of approximately 0.959.

![Initial Circle Packing](circle_packing_1.png)

### Generation 10 Breakthrough

By generation 10, OpenEvolve had already discovered a more sophisticated approach:

```python
# Generation 10
# Parameters for the arrangement (fine-tuned)
r_center = 0.1675  # Central circle radius

# 1. Place central circle
centers[0] = [0.5, 0.5]
radii[0] = r_center

# 2. First ring: 6 circles in hexagonal arrangement
r_ring1 = 0.1035
ring1_distance = r_center + r_ring1 + 0.0005  # Small gap for stability
for i in range(6):
    angle = 2 * np.pi * i / 6
    centers[i+1] = [
        0.5 + ring1_distance * np.cos(angle),
        0.5 + ring1_distance * np.sin(angle)
    ]
    radii[i+1] = r_ring1
```

The key innovations at this stage included:
- A carefully tuned hexagonal arrangement for the first ring
- Strategic placement of corner circles
- An additional optimization step to maximize each circle's radius

This approach achieved a sum of radii of approximately 1.795.

![Generation 10 Packing](circle_packing_10.png)

### Generation 100: Grid-Based Approach

By generation 100, OpenEvolve had pivoted to a grid-based approach with variable sized circles:

```python
# Generation 100
# Row 1: 5 circles
centers[0] = [0.166, 0.166]
centers[1] = [0.333, 0.166]
centers[2] = [0.500, 0.166]
centers[3] = [0.667, 0.166]
centers[4] = [0.834, 0.166]

# Row 2: 6 circles (staggered)
centers[5] = [0.100, 0.333]
centers[6] = [0.266, 0.333]
# ... additional circles
```

Key innovations:
- Grid-like pattern with staggered rows
- Variable circle sizes based on position (larger in the center)
- More aggressive optimization routine with 50 iterations

This approach achieved a sum of radii of approximately 2.201.

![Generation 100 Packing](circle_packing_190.png)

### Final Solution: Mathematical Optimization

The breakthrough came when OpenEvolve discovered the power of mathematical optimization techniques. The final solution uses:

```python
# Final solution with scipy.optimize
def construct_packing():
    # ... initialization code ...
    
    # Objective function: Negative sum of radii (to maximize)
    def objective(x):
        centers = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        return -np.sum(radii)

    # Constraint: No overlaps and circles stay within the unit square
    def constraint(x):
        centers = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        
        # Overlap constraint
        overlap_constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                overlap_constraints.append(dist - (radii[i] + radii[j]))
        # ... boundary constraints ...
        
    # Optimization using SLSQP
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
```

The key innovation in the final solution:
- Using `scipy.optimize.minimize` with SLSQP method to find the optimal configuration
- Formulating circle packing as a constrained optimization problem
- Representing both circle positions and radii as optimization variables
- Carefully crafted constraints to enforce non-overlap and boundary conditions

This approach achieved a sum of radii of 2.634, matching the AlphaEvolve paper's result of 2.635 to within 0.04%!

![Final Packing Solution](circle_packing_460.png)

## Results

Our final solution achieves:

```
Sum of radii: 2.634292402141039
Target ratio: 0.9997314619131079 (99.97% of AlphaEvolve's result)
```

This demonstrates that OpenEvolve can successfully reproduce the results from the AlphaEvolve paper on this mathematical optimization problem.

## Reproducibility & the target value

The evaluator uses a `TARGET_VALUE` (default `2.635`, the AlphaEvolve result for n=26) only to compute the `target_ratio` and `combined_score` metrics. It creates evolutionary pressure by scaling fitness — it does not leak a solution to the model, and any positive value works (see issue #117):

> You can put any other value for the target like 3.0 and it will still work. The target is used to compute the metrics and create evolutionary pressure. It by itself doesn't give a solution.

To reproduce the search without the AlphaEvolve prior, override it via the environment:

```bash
export CIRCLE_PACKING_TARGET_VALUE=3.0  # or 1.0, 10.0, ... any positive value
```

Some empirical reference points for what OpenEvolve reaches on this problem without relying on the target as a hint:

- With the default configuration, OpenEvolve reached a sum of radii of 2.634 in about 800 generations (see issue #117).
- Issue #156 reports 2.635977394746595 — surpassing the AlphaEvolve result — found at iteration 206 of a 450-iteration multi-phase run.

Note that the system messages in `config_phase_1.yaml` (line 24), `config_phase_2.yaml` (line 24), `config_phase_1_anthropic.yaml` (line 21) and `config_phase_2_anthropic.yaml` (line 22) mention the 2.635 value to the model as an encouraging upper bound. For a fully prior-free run, edit or remove those sentences in your own configs as well.

## Key Observations

The evolution process demonstrated several interesting patterns:

1. **Algorithm Transition**: OpenEvolve discovered increasingly sophisticated algorithms, from basic geometric constructions to advanced mathematical optimization techniques.

2. **Exploration-Exploitation Balance**: The two-phase approach was crucial - initial exploration of different patterns followed by exploitation and refinement of the most promising approaches.

3. **Breakthrough Discoveries**: The most significant improvements came from fundamental changes in approach (e.g., switching from manual construction to mathematical optimization), not just parameter tuning.

4. **Code Complexity Evolution**: As the solutions improved, the code grew in complexity, adopting more sophisticated mathematical techniques.

## Running the Example

To reproduce our results:

```bash
# Phase 1: Initial exploration
python openevolve-run.py examples/circle_packing/initial_program.py \
  examples/circle_packing/evaluator.py \
  --config examples/circle_packing/config_phase_1.yaml \
  --iterations 100

# Phase 2: Breaking through the plateau
python openevolve-run.py examples/circle_packing/openevolve_output/checkpoints/checkpoint_100/best_program.py \
  examples/circle_packing/evaluator.py \
  --config examples/circle_packing/config_phase_2.yaml \
  --iterations 100
```

To visualize the best solution:

```python
from examples.circle_packing.openevolve_output.best.best_program import run_packing, visualize

centers, radii, sum_radii = run_packing()
print(f"Sum of radii: {sum_radii}")
visualize(centers, radii)
```

## Conclusion

This example demonstrates OpenEvolve's ability to discover sophisticated algorithms for mathematical optimization problems. By evolving from simple constructive approaches to advanced numerical optimization techniques, OpenEvolve was able to match the results reported in the AlphaEvolve paper.

The circle packing problem shows how OpenEvolve can discover not just improvements to existing algorithms, but entirely new algorithmic approaches, transitioning from manual geometric construction to principled mathematical optimization.