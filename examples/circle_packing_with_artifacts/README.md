# Circle Packing Example with Artifacts

This example demonstrates how OpenEvolve can be used to tackle the challenging mathematical problem of circle packing, a classic problem in computational geometry. Specifically, we focus on packing 26 circles of varying sizes into a unit square to maximize the sum of their radii, replicating one of the tasks from the AlphaEvolve paper.

**Enhanced with Artifacts**: This version showcases OpenEvolve's artifacts feature, which provides detailed execution feedback to help the LLM understand what went wrong and how to fix it.

## Artifacts Enhancement

This example has been enhanced to demonstrate OpenEvolve's **artifacts side-channel**, which captures detailed execution information beyond just numeric metrics. When a program fails or succeeds, the evaluator now provides rich context that gets included in the next generation's prompt.

### What Artifacts Capture

The enhanced evaluator captures different types of information:

#### 🚨 **Failure Artifacts**
When compilation or runtime errors occur:
```
## Last Execution Output
### Stderr
```
Invalid shapes: centers=(25, 2), radii=(26,), expected (26, 2) and (26,)
```
### Failure_stage
```
shape_validation
```
### Suggestion
```
Check for syntax errors, import issues, or runtime exceptions
```
```

#### ⚠️ **Validation Artifacts**  
When geometric constraints are violated:
```
## Last Execution Output
### Boundary_violations
```
Circle 23 at (0.950000, 0.950000) with radius 0.055000 is outside unit square
Circle 24 at (0.980000, 0.980000) with radius 0.030000 is outside unit square
```
### Overlap_violations
```
Circles 5 and 12 overlap: dist=0.180000, r1+r2=0.190000
```
### Validation_report
```
Valid: False, Violations: 2 boundary, 1 overlaps
```
```

#### ✅ **Success Artifacts**
For excellent solutions (>95% of target):
```
## Last Execution Output
### Stdout
```
Excellent packing! Achieved 99.7% of target value
```
### Radius_stats
```
Min: 0.045123, Max: 0.167500, Avg: 0.101319
```
### Packing_summary
```
Sum of radii: 2.634292/2.635 = 0.9997
```
```

#### ⏱️ **Performance Artifacts**
Always included:
```
### Execution_time
```
12.45s
```
```

### How Artifacts Help Evolution

The artifacts provide crucial context that helps the LLM make better decisions:

1. **Specific Error Fixing**: Instead of just seeing `validity: 0.0`, the LLM sees exactly which circles are problematic and why
2. **Performance Guidance**: Execution time helps the LLM understand if algorithms are too slow
3. **Success Recognition**: When a solution works well, artifacts explain why it's good
4. **Debugging Context**: Full stack traces help fix syntax and runtime errors

### Example Evolution with Artifacts

Here's how artifacts help in a typical evolution scenario:

**Generation N**: Program fails with overlapping circles
```python
# Faulty code
centers[5] = [0.3, 0.3]  
centers[6] = [0.3, 0.3]  # Same position!
radii[5] = radii[6] = 0.1
```

**Artifacts captured**:
```
Circles 5 and 6 overlap: dist=0.000000, r1+r2=0.200000
```

**Generation N+1**: LLM sees the overlap details and fixes it
```python
# Fixed code  
centers[5] = [0.3, 0.3]
centers[6] = [0.5, 0.3]  # Different position!
radii[5] = radii[6] = 0.1
```

This leads to faster convergence because the LLM gets specific, actionable feedback instead of just numeric scores.

### Backward Compatibility

The artifacts enhancement is fully backward compatible:
- **Existing evaluators** continue to work unchanged 
- **Enhanced evaluators** return `EvaluationResult` with both metrics and artifacts
- **Disable artifacts** by setting `export ENABLE_ARTIFACTS=false` if needed

To run with artifacts disabled:
```bash
export ENABLE_ARTIFACTS=false
python openevolve-run.py examples/circle_packing_with_artifacts/initial_program.py \
  examples/circle_packing_with_artifacts/evaluator.py \
  --config examples/circle_packing_with_artifacts/config_phase_1.yaml
```

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

![Initial Circle Packing](../circle_packing/circle_packing_1.png)

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

![Generation 10 Packing](../circle_packing/circle_packing_10.png)

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

![Generation 100 Packing](../circle_packing/circle_packing_190.png)

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

![Final Packing Solution](../circle_packing/circle_packing_460.png)

## Results

Our final solution achieves:

```
Sum of radii: 2.634292402141039
Target ratio: 0.9997314619131079 (99.97% of AlphaEvolve's result)
```

This demonstrates that OpenEvolve can successfully reproduce the results from the AlphaEvolve paper on this mathematical optimization problem.

## Fast Convergence with Dual-Model Configuration

Using a dual-model configuration with weighted sampling, OpenEvolve achieves near-optimal results in remarkably few iterations:

![Evolution Plot](evolution_plot.png)

### Configuration

The `config.yaml` uses two Gemini models with different weights:
- `google/gemini-2.5-flash-lite` (weight: 0.8) - Fast, cost-effective for exploration
- `google/gemini-2.5-flash` (weight: 0.2) - Higher capability for breakthroughs

```yaml
llm:
  models:
    - name: "google/gemini-2.5-flash-lite"
      weight: 0.8
    - name: "google/gemini-2.5-flash"
      weight: 0.2
```

### Rapid Convergence

The plot shows the evolution of sum_radii across program versions:

- **Version 0**: Starts at ~0.96 (basic initial program)
- **Version 6**: First major improvement to ~2.09
- **Version 21**: Reaches 2.63 (99.8% of target)
- **Final**: Achieves 2.6304 sum of radii

**Key insight**: OpenEvolve discovers the mathematical optimization approach (using `scipy.optimize.minimize` with SLSQP) by version 21, achieving 99.8% of the AlphaEvolve target in just ~40 program evaluations. The dual-model approach allows rapid exploration with the lighter model while leveraging the more capable model for breakthrough discoveries.

### Why It Works

1. **Artifacts provide rich feedback**: Failed programs return detailed error information (boundary violations, overlaps), helping the LLM quickly correct mistakes
2. **MAP-Elites diversity**: The feature dimensions (`radius_variance`, `spatial_spread`) maintain diverse solutions in the population
3. **Island-based evolution**: 4 islands evolve independently, preventing premature convergence
4. **Efficient model weighting**: 80% lightweight model for broad exploration, 20% capable model for sophisticated solutions

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