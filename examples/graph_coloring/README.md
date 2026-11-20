# Graph Coloring Example

This example demonstrates how OpenEvolve can discover sophisticated graph coloring algorithms starting from a simple greedy implementation.

## Problem Description

**Graph Coloring** is a classic NP-hard problem:
- Given an undirected graph G = (V, E)
- Assign colors to vertices such that no two adjacent vertices share the same color
- Goal: Use the minimum number of colors (the **chromatic number** χ(G))

This problem has many real-world applications:
- **Scheduling**: Exam timetabling, job scheduling
- **Register allocation**: Compiler optimization
- **Frequency assignment**: Radio/cellular networks
- **Map coloring**: Cartography

## Getting Started

### Prerequisites

1. Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey) (free tier available)
2. Edit `config.yaml` and replace `<your-gemini-api-key>` with your actual API key

Alternatively, to use Claude instead of Gemini:
1. Get an Anthropic API key from [Anthropic Console](https://console.anthropic.com/)
2. In `config.yaml`, uncomment the Claude model section and comment out Gemini

### Running Evolution

To run this example:

```bash
cd examples/graph_coloring
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml --iterations 100
```

### Running Benchmarks

To test a coloring algorithm on the full DIMACS benchmark suite:

```bash
# Test the initial greedy algorithm
python run_benchmarks.py

# Test an evolved program
python run_benchmarks.py best_program.py
```

This produces a results table showing colors used vs chromatic number for each graph.

## Evaluation System

The evaluator uses a **3-stage cascade** with DIMACS benchmark graphs:

### Stage 1: Validity Check
- Tests on 3 simple graphs (Petersen, K5, cycle)
- Must produce **valid colorings** (no adjacent vertices share colors)
- Programs that fail validity are rejected immediately

### Stage 2: Quick Evaluation
- Tests on small DIMACS benchmarks (11-87 vertices)
- Includes: Mycielski graphs (myciel3-5), Queen graphs (5×5 to 8×8), and book graphs (david, huck, jean)
- Score threshold: 0.7 to proceed to Stage 3

### Stage 3: Comprehensive Evaluation
- Full DIMACS benchmark suite (22 graphs, 11-211 vertices)
- Includes challenging graphs: queen11_11, queen12_12, mulsol.i.1, zeroin.i.1
- Scoring combines coloring quality (70%) and time efficiency (30%)

### DIMACS Benchmarks

The evaluator uses standard [DIMACS graph coloring benchmarks](https://mat.tepper.cmu.edu/COLOR/instances.html) with known chromatic numbers:

| Graph | Vertices | χ (chromatic number) |
|-------|----------|---------------------|
| myciel3 | 11 | 4 |
| myciel4 | 23 | 5 |
| myciel5 | 47 | 6 |
| queen5_5 | 25 | 5 |
| queen6_6 | 36 | 7 |
| queen8_8 | 64 | 9 |
| david | 87 | 11 |
| huck | 74 | 11 |
| jean | 80 | 10 |
| anna | 138 | 11 |
| games120 | 120 | 9 |
| mulsol.i.1 | 197 | 49 |
| zeroin.i.1 | 211 | 49 |

## Algorithm Evolution

### Initial Algorithm (Simple Greedy)

The initial implementation is a basic greedy algorithm that processes vertices in order and assigns the smallest available color:

```python
def graph_coloring(graph):
    coloring = {}
    for vertex in range(graph.num_vertices):
        neighbor_colors = set()
        for neighbor in graph.get_neighbors(vertex):
            if neighbor in coloring:
                neighbor_colors.add(coloring[neighbor])

        color = 0
        while color in neighbor_colors:
            color += 1

        coloring[vertex] = color
    return coloring
```

### Evolved Algorithm (DSatur)

After 100 iterations, OpenEvolve independently discovered an optimized **DSatur algorithm**:

```python
def graph_coloring(graph):
    coloring = {}
    uncolored = set(range(graph.num_vertices))

    # Cache saturation degrees for efficiency
    sat_degrees = [0] * graph.num_vertices
    neighbor_color_sets = [set() for _ in range(graph.num_vertices)]

    while uncolored:
        # Select vertex with highest saturation degree, then highest degree
        vertex = max(uncolored,
                    key=lambda v: (sat_degrees[v], graph.get_degree(v), -v))

        # Assign smallest available color
        color = 0
        while color in neighbor_color_sets[vertex]:
            color += 1

        coloring[vertex] = color
        uncolored.remove(vertex)

        # Update saturation degrees of uncolored neighbors
        for neighbor in graph.get_neighbors(vertex):
            if neighbor in uncolored:
                if color not in neighbor_color_sets[neighbor]:
                    neighbor_color_sets[neighbor].add(color)
                    sat_degrees[neighbor] += 1

    return coloring
```

Key improvements discovered:
1. **Saturation-based vertex ordering**: Prioritizes vertices with the most distinct neighbor colors
2. **Cached neighbor color sets**: O(1) lookup for available colors
3. **Tie-breaking by degree**: When saturation is equal, prefer high-degree vertices

## Results

| Metric | Initial (Greedy) | Evolved (DSatur) |
|--------|------------------|------------------|
| Combined Score | 0.893 | **0.938** |
| Optimal Colorings | 10/22 | **15/22** |
| Total Colors | 389 | **362** |

The evolved algorithm:
- Uses **27 fewer colors** across the test suite
- Achieves **optimal coloring on 5 additional graphs**
- Was discovered by **iteration 14** (generation 2)

## References

- Welsh, D.J.A. and Powell, M.B. (1967). "An upper bound for the chromatic number of a graph and its application to timetabling problems."
- Brélaz, D. (1979). "New Methods to Color the Vertices of a Graph" (DSatur algorithm)
- Leighton, F.T. (1979). "A graph coloring algorithm for large scheduling problems" (RLF algorithm)
- DIMACS Graph Coloring Challenge: https://mat.tepper.cmu.edu/COLOR/instances.html

## Next Steps

Try modifying the config.yaml to:
- Increase iterations for more evolution
- Change LLM models or weights
- Adjust the system message to guide evolution toward specific algorithms
- Add larger DIMACS benchmarks to the test suite
