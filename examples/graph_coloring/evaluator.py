"""
Evaluator for the Graph Coloring example.

This evaluator tests evolved graph coloring algorithms on multiple test graphs
and returns metrics including:
- Number of colors used
- Validity of coloring (no conflicts)
- Performance across different graph types
- Execution time (with penalties for slow algorithms)
"""

import importlib.util
import sys
import os
import random
import time

# ============================================================
# Time Budget Configuration
# ============================================================
# Time budgets scale with graph complexity: budget = BASE + COEFF × (n² + m)
# where n = vertices, m = edges.
#
# The coefficient was determined empirically:
#   - Measured simple greedy: ~21-30 nanoseconds per n²
#   - With 30x margin (3x base × 10x exploration room): COEFF = 8.8e-7
#
# This allows O(n²) and O(n×m) algorithms to complete comfortably,
# while penalizing multi-start approaches that run many trials.
#
# Example budgets (dense graphs, m ≈ n²/2):
#   n=30:   ~2.2ms
#   n=100:  ~14ms
#   n=500:  ~331ms
#   n=1000: ~1.32s
#
BASE_TIME_BUDGET = 0.001  # 1ms base budget
COEFF_TIME = 8.8e-7  # ~880 nanoseconds per (n² + m) unit
TIME_PENALTY_WEIGHT = 0.3  # 30% of score comes from time efficiency

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_program(program_path: str):
    """Dynamically load the evolved program."""
    spec = importlib.util.spec_from_file_location("evolved_module", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_time_budget(num_vertices: int, num_edges: int) -> float:
    """
    Calculate time budget for a graph based on vertices and edges.

    Budget scales with graph complexity: budget = BASE + COEFF × (n² + m)
    This matches the O(n²) or O(n×m) complexity of typical coloring algorithms.

    Args:
        num_vertices: Number of vertices in the graph (n)
        num_edges: Number of edges in the graph (m)

    Returns:
        Time budget in seconds
    """
    complexity = num_vertices * num_vertices + num_edges
    return BASE_TIME_BUDGET + COEFF_TIME * complexity


def calculate_time_score(elapsed_time: float, time_budget: float) -> float:
    """
    Calculate time efficiency score.

    - If elapsed <= budget: score = 1.0
    - If elapsed > budget: score decays exponentially
    - Score never goes below 0.1 (to avoid completely zeroing out good colorings)

    Args:
        elapsed_time: Actual execution time in seconds
        time_budget: Allowed time budget in seconds

    Returns:
        Time score between 0.1 and 1.0
    """
    if elapsed_time <= time_budget:
        return 1.0

    # Exponential decay: each doubling of time over budget halves the score
    overage_ratio = elapsed_time / time_budget
    time_score = 1.0 / overage_ratio

    # Floor at 0.1 to not completely kill good colorings
    return max(0.1, time_score)


def create_random_graph(num_vertices: int, edge_probability: float, seed: int = None):
    """
    Create a random Erdős–Rényi graph.

    Args:
        num_vertices: Number of vertices
        edge_probability: Probability of edge between any two vertices
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)

    # Import Graph class from the module
    from initial_program import Graph

    g = Graph(num_vertices)
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if random.random() < edge_probability:
                g.add_edge(i, j)
    return g


def create_crown_graph(n: int, adversarial_ordering: bool = True):
    """
    Create a Crown Graph with 2n vertices.

    Crown graph is a classic adversarial example for greedy coloring algorithms.
    It's bipartite (chromatic number = 2), but greedy/DSatur algorithms
    processing vertices in certain orders can use up to n colors.

    Structure:
    - Two sets of n vertices: top and bottom
    - Each top vertex connects to ALL bottom vertices EXCEPT its "partner"
    - This creates a bipartite graph where each vertex has degree n-1

    With adversarial_ordering=True, vertices are relabeled so that processing
    them in order 0,1,2,... interleaves top and bottom, causing greedy
    algorithms to use many colors.

    Args:
        n: Half the number of vertices (total vertices = 2n)
        adversarial_ordering: If True, relabel vertices to be adversarial

    Returns:
        Graph with 2n vertices, chromatic number = 2
    """
    from initial_program import Graph

    g = Graph(2 * n)

    if adversarial_ordering:
        # Interleave: vertex 0=top0, 1=bottom0, 2=top1, 3=bottom1, ...
        # So even vertices are "top", odd vertices are "bottom"
        # top_i is at position 2*i, bottom_i is at position 2*i+1
        # Connect top_i (2*i) to all bottom_j (2*j+1) where i != j
        for i in range(n):
            for j in range(n):
                if i != j:
                    top_vertex = 2 * i      # top_i
                    bottom_vertex = 2 * j + 1  # bottom_j
                    g.add_edge(top_vertex, bottom_vertex)
    else:
        # Original friendly ordering
        for i in range(n):
            for j in range(n):
                if i != j:
                    g.add_edge(i, n + j)

    return g


def create_mycielski_graph(k: int):
    """
    Create a Mycielskian graph M_k.

    Mycielski's construction creates triangle-free graphs with high chromatic number.
    - M_2: Single edge (2 vertices, χ=2)
    - M_3: Cycle C5 (5 vertices, χ=3)
    - M_4: Grötzsch graph (11 vertices, χ=4)
    - M_k: χ(M_k) = k, |V| = 3*2^(k-2) - 1

    These graphs are challenging because they have no triangles (local structure
    suggests 2 colors might suffice) but require k colors globally.

    Args:
        k: The Mycielski index (k >= 2)

    Returns:
        Graph with chromatic number k
    """
    from initial_program import Graph

    if k < 2:
        raise ValueError("k must be >= 2")

    if k == 2:
        # M_2 is a single edge
        g = Graph(2)
        g.add_edge(0, 1)
        return g

    # Start with M_2 and iteratively build up
    # Current graph vertices and edges
    vertices = [0, 1]
    edges = [(0, 1)]

    for _ in range(k - 2):
        n = len(vertices)
        # Add n new vertices (u_1, ..., u_n) and one new vertex w
        new_vertices = list(range(n, 2 * n))
        w = 2 * n

        new_edges = []
        # For each original vertex v_i, connect u_i to all neighbors of v_i
        for i, v in enumerate(vertices):
            u = new_vertices[i]
            for (a, b) in edges:
                if a == v:
                    new_edges.append((u, b))
                elif b == v:
                    new_edges.append((u, a))
            # Also connect u_i to w
            new_edges.append((u, w))

        vertices = list(range(w + 1))
        edges = edges + new_edges

    g = Graph(len(vertices))
    for (a, b) in edges:
        g.add_edge(a, b)
    return g


def create_dsatur_adversarial_graph(seed: int):
    """
    Create a random graph where DSatur is known to perform suboptimally.

    These graphs are found empirically - certain random seeds produce graphs
    where simple greedy outperforms DSatur, suggesting room for hybrid
    approaches or smarter algorithms.

    Args:
        seed: Random seed (19 and 70 are known adversarial cases)

    Returns:
        Graph where DSatur may use more colors than necessary
    """
    import random
    from initial_program import Graph

    random.seed(seed)
    n = 15
    p = 0.5
    g = Graph(n)
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                g.add_edge(i, j)
    return g


def compute_chromatic_number(graph, max_colors: int = None) -> int:
    """
    Compute the exact chromatic number using backtracking.

    This is exponential time but works for small graphs (< 20 vertices).
    Uses branch and bound with greedy upper bound.

    Args:
        graph: Graph to color
        max_colors: Upper bound to try (default: use greedy result)

    Returns:
        The exact chromatic number
    """
    n = graph.num_vertices

    if max_colors is None:
        # Get greedy upper bound
        from initial_program import graph_coloring, count_colors
        max_colors = count_colors(graph_coloring(graph))

    def is_safe(vertex, color, coloring):
        for neighbor in graph.get_neighbors(vertex):
            if coloring.get(neighbor) == color:
                return False
        return True

    def can_color_with_k(k):
        """Check if graph can be colored with k colors using backtracking."""
        coloring = {}

        def backtrack(vertex):
            if vertex == n:
                return True
            for color in range(k):
                if is_safe(vertex, color, coloring):
                    coloring[vertex] = color
                    if backtrack(vertex + 1):
                        return True
                    del coloring[vertex]
            return False

        return backtrack(0)

    # Binary search for minimum k
    lo, hi = 1, max_colors
    while lo < hi:
        mid = (lo + hi) // 2
        if can_color_with_k(mid):
            hi = mid
        else:
            lo = mid + 1

    return lo


def create_chvatal_graph():
    """
    Create the Chvátal graph.

    The Chvátal graph is a 4-chromatic graph with 12 vertices that is
    triangle-free and 4-regular. It's a good test case because its
    structure doesn't give obvious clues about the chromatic number.

    Returns:
        Graph with 12 vertices, chromatic number = 4
    """
    from initial_program import Graph

    g = Graph(12)
    edges = [
        (0, 1), (0, 4), (0, 6), (0, 9),
        (1, 2), (1, 5), (1, 7),
        (2, 3), (2, 6), (2, 8),
        (3, 4), (3, 7), (3, 9),
        (4, 5), (4, 8),
        (5, 10), (5, 11),
        (6, 10), (6, 11),
        (7, 8), (7, 11),
        (8, 10),
        (9, 10), (9, 11)
    ]
    for (a, b) in edges:
        g.add_edge(a, b)
    return g


def create_test_graphs():
    """
    Create a suite of test graphs for evaluation.

    Includes adversarial graphs designed to expose weaknesses in
    greedy algorithms like DSatur.

    Returns:
        list: List of (name, graph, known_chromatic_number_or_bound) tuples
    """
    from initial_program import Graph

    test_graphs = []

    # ============================================================
    # Standard test graphs (DSatur should do well on these)
    # ============================================================

    # 1. Petersen Graph - chromatic number = 3
    petersen = Graph(10)
    for i in range(5):
        petersen.add_edge(i, (i + 1) % 5)
    for i in range(5):
        petersen.add_edge(5 + i, 5 + (i + 2) % 5)
    for i in range(5):
        petersen.add_edge(i, 5 + i)
    test_graphs.append(("Petersen", petersen, 3))

    # 2. Complete graph K5 - chromatic number = 5
    k5 = Graph(5)
    for i in range(5):
        for j in range(i + 1, 5):
            k5.add_edge(i, j)
    test_graphs.append(("K5", k5, 5))

    # 3. Cycle graph C7 - chromatic number = 3 (odd cycle)
    cycle = Graph(7)
    for i in range(7):
        cycle.add_edge(i, (i + 1) % 7)
    test_graphs.append(("Cycle7", cycle, 3))

    # ============================================================
    # Adversarial graphs (DSatur often fails on these)
    # ============================================================

    # 4. Crown Graph (n=8) - 16 vertices, χ=2, DSatur may use up to 8 colors!
    # This is a classic adversarial example for greedy coloring
    crown8 = create_crown_graph(8)
    test_graphs.append(("Crown8", crown8, 2))

    # 5. Crown Graph (n=10) - 20 vertices, χ=2, DSatur may use up to 10 colors
    crown10 = create_crown_graph(10)
    test_graphs.append(("Crown10", crown10, 2))

    # 6. Mycielski M_4 (Grötzsch graph) - 11 vertices, χ=4, triangle-free
    # Triangle-free structure may mislead algorithms that use local clique detection
    mycielski4 = create_mycielski_graph(4)
    test_graphs.append(("Mycielski4", mycielski4, 4))

    # 7. Chvátal Graph - 12 vertices, χ=4, 4-regular, triangle-free
    chvatal = create_chvatal_graph()
    test_graphs.append(("Chvatal", chvatal, 4))

    # ============================================================
    # DSatur-adversarial graphs (where DSatur uses more colors than optimal)
    # ============================================================

    # 8. Random graph seed=19 - DSatur uses 6, optimal is 5
    adversarial19 = create_dsatur_adversarial_graph(19)
    chromatic19 = compute_chromatic_number(adversarial19)
    test_graphs.append(("DSaturAdversarial19", adversarial19, chromatic19))

    # 9. Random graph seed=70 - DSatur uses 6, optimal is 5
    adversarial70 = create_dsatur_adversarial_graph(70)
    chromatic70 = compute_chromatic_number(adversarial70)
    test_graphs.append(("DSaturAdversarial70", adversarial70, chromatic70))

    # ============================================================
    # Scale test graphs
    # ============================================================

    # 10. Crown Graph (n=15) - 30 vertices, χ=2
    # Larger crown to really stress-test algorithms
    crown15 = create_crown_graph(15)
    test_graphs.append(("Crown15", crown15, 2))

    return test_graphs


def evaluate(program_path: str) -> dict:
    """
    Evaluate an evolved graph coloring program.

    Args:
        program_path: Path to the evolved program file

    Returns:
        dict: Evaluation metrics including combined_score
    """
    try:
        # Load the evolved module
        module = load_program(program_path)

        # Get the coloring function and helper functions
        graph_coloring = module.graph_coloring
        is_valid_coloring = module.is_valid_coloring
        count_colors = module.count_colors

        # Create test graphs
        test_graphs = create_test_graphs()

        total_score = 0
        num_tests = len(test_graphs)
        all_valid = True
        total_colors = 0
        optimal_count = 0

        results_detail = []

        total_time = 0
        total_time_score = 0

        for name, graph, known_chromatic in test_graphs:
            # Calculate time budget for this graph based on vertices and edges
            num_edges = len(graph.get_edges())
            time_budget = get_time_budget(graph.num_vertices, num_edges)

            # Time the coloring
            start_time = time.time()
            coloring = graph_coloring(graph)
            elapsed_time = time.time() - start_time
            total_time += elapsed_time

            # Check validity
            is_valid, conflicts = is_valid_coloring(graph, coloring)
            num_colors = count_colors(coloring)

            # Calculate time efficiency score
            time_score = calculate_time_score(elapsed_time, time_budget)
            total_time_score += time_score

            if not is_valid:
                all_valid = False
                # Invalid coloring gets score of 0 for this graph
                color_score = 0
                graph_score = 0
            else:
                total_colors += num_colors

                # Score based on colors used
                # If we know optimal, score = optimal / used (max 1.0)
                # Otherwise, use heuristic based on graph size
                if known_chromatic is not None:
                    color_score = known_chromatic / num_colors
                    if num_colors == known_chromatic:
                        optimal_count += 1
                else:
                    # For unknown optimal, estimate lower bound as max_degree + 1
                    max_degree = max(graph.get_degree(v) for v in range(graph.num_vertices))
                    estimated_lower = max(2, max_degree // 2)
                    color_score = estimated_lower / num_colors

                color_score = min(1.0, color_score)  # Cap at 1.0

                # Combined score: color quality weighted with time efficiency
                # (1 - TIME_PENALTY_WEIGHT) for colors + TIME_PENALTY_WEIGHT for time
                graph_score = (1 - TIME_PENALTY_WEIGHT) * color_score + TIME_PENALTY_WEIGHT * time_score

            total_score += graph_score
            results_detail.append({
                'name': name,
                'colors': num_colors,
                'valid': is_valid,
                'conflicts': conflicts,
                'color_score': color_score if is_valid else 0,
                'time_score': time_score,
                'score': graph_score,
                'time': elapsed_time,
                'time_budget': time_budget,
                'over_budget': elapsed_time > time_budget
            })

        # Calculate final metrics
        avg_score = total_score / num_tests
        avg_time_score = total_time_score / num_tests

        # Combined score: 0 if any invalid, otherwise based on color efficiency + time
        if not all_valid:
            combined_score = 0.0
        else:
            combined_score = avg_score

        return {
            'combined_score': combined_score,
            'avg_color_score': avg_score,
            'avg_time_score': avg_time_score,
            'all_valid': all_valid,
            'optimal_count': optimal_count,
            'total_colors': total_colors,
            'total_time': total_time,
            'num_tests': num_tests,
            'time_penalty_weight': TIME_PENALTY_WEIGHT,
            'details': results_detail
        }

    except Exception as e:
        # Return minimal score on error
        return {
            'combined_score': 0.0,
            'error': str(e)
        }


# ============================================================
# Cascade Evaluation Stages
# ============================================================

def evaluate_stage1(program_path: str) -> dict:
    """
    Stage 1: Validity Gate - Quick validation check only.

    Tests the algorithm on 3 simple graphs and ONLY checks if colorings
    are valid. No scoring is done. This is the fastest possible rejection
    of broken algorithms.

    Args:
        program_path: Path to the evolved program file

    Returns:
        dict: combined_score is 1.0 if all valid, 0.0 if any invalid
    """
    try:
        # Load the evolved module
        module = load_program(program_path)

        # Get the coloring function and helper functions
        graph_coloring = module.graph_coloring
        is_valid_coloring = module.is_valid_coloring
        Graph = module.Graph

        # Quick test graphs - small and fast to validate
        quick_tests = []

        # 1. Triangle (K3) - simplest non-trivial graph
        triangle = Graph(3)
        triangle.add_edge(0, 1)
        triangle.add_edge(1, 2)
        triangle.add_edge(0, 2)
        quick_tests.append(("Triangle", triangle))

        # 2. Complete graph K4
        k4 = Graph(4)
        for i in range(4):
            for j in range(i + 1, 4):
                k4.add_edge(i, j)
        quick_tests.append(("K4", k4))

        # 3. Simple path (3 vertices)
        path = Graph(3)
        path.add_edge(0, 1)
        path.add_edge(1, 2)
        quick_tests.append(("Path3", path))

        # Run validity check only - no scoring
        for name, graph in quick_tests:
            coloring = graph_coloring(graph)
            is_valid, conflicts = is_valid_coloring(graph, coloring)

            if not is_valid:
                # Fail fast: any invalid coloring = immediate rejection
                return {
                    'combined_score': 0.0,
                    'stage1_passed': False,
                    'failed_on': name,
                    'conflicts': conflicts
                }

        # All graphs produced valid colorings
        return {
            'combined_score': 1.0,
            'stage1_passed': True
        }

    except Exception as e:
        return {
            'combined_score': 0.0,
            'stage1_passed': False,
            'error': str(e)
        }


def evaluate_stage2(program_path: str) -> dict:
    """
    Stage 2: Quick scoring on small graphs.

    Only runs if stage 1 passed. Computes quality scores on 3 small graphs
    to filter out algorithms that produce valid but inefficient colorings.

    Args:
        program_path: Path to the evolved program file

    Returns:
        dict: Quick scoring metrics
    """
    try:
        # Load the evolved module
        module = load_program(program_path)

        # Get the coloring function and helper functions
        graph_coloring = module.graph_coloring
        is_valid_coloring = module.is_valid_coloring
        count_colors = module.count_colors
        Graph = module.Graph

        # Same test graphs as stage 1, but now with expected chromatic numbers
        quick_tests = []

        # 1. Triangle (K3) - chromatic number = 3
        triangle = Graph(3)
        triangle.add_edge(0, 1)
        triangle.add_edge(1, 2)
        triangle.add_edge(0, 2)
        quick_tests.append(("Triangle", triangle, 3))

        # 2. Complete graph K4 - chromatic number = 4
        k4 = Graph(4)
        for i in range(4):
            for j in range(i + 1, 4):
                k4.add_edge(i, j)
        quick_tests.append(("K4", k4, 4))

        # 3. Simple path (3 vertices) - chromatic number = 2
        path = Graph(3)
        path.add_edge(0, 1)
        path.add_edge(1, 2)
        quick_tests.append(("Path3", path, 2))

        # Compute scores
        total_score = 0
        all_valid = True

        for name, graph, expected_chromatic in quick_tests:
            coloring = graph_coloring(graph)
            is_valid, conflicts = is_valid_coloring(graph, coloring)
            num_colors = count_colors(coloring)

            if not is_valid:
                all_valid = False
                graph_score = 0
            else:
                graph_score = min(1.0, expected_chromatic / num_colors)

            total_score += graph_score

        avg_score = total_score / len(quick_tests)

        # If any invalid, combined_score = 0
        if not all_valid:
            combined_score = 0.0
        else:
            combined_score = avg_score

        return {
            'combined_score': combined_score,
            'stage2_avg_score': avg_score,
            'stage2_all_valid': all_valid
        }

    except Exception as e:
        return {
            'combined_score': 0.0,
            'error': str(e)
        }


def evaluate_stage3(program_path: str) -> dict:
    """
    Stage 3: Full evaluation on all test graphs.

    Only runs if stage 2 passed threshold. Comprehensive evaluation
    on all 6 test graphs with detailed scoring.
    If any coloring is invalid, combined_score = 0.

    Args:
        program_path: Path to the evolved program file

    Returns:
        dict: Full evaluation metrics with combined_score = 0 if any invalid
    """
    result = evaluate(program_path)

    # Enforce: any invalid coloring = score of 0
    if not result.get('all_valid', False):
        result['combined_score'] = 0.0

    return result


if __name__ == "__main__":
    # Test the 3-stage cascade evaluation with the initial program
    print("=" * 50)
    print("Stage 1: Validity Gate (3 small graphs)")
    print("=" * 50)
    stage1_result = evaluate_stage1("initial_program.py")
    print(f"  Passed: {stage1_result.get('stage1_passed', False)}")
    print(f"  Combined Score: {stage1_result.get('combined_score', 0):.4f}")

    if not stage1_result.get('stage1_passed'):
        print("\n  FAILED - Skipping remaining stages")
        if 'failed_on' in stage1_result:
            print(f"  Failed on: {stage1_result['failed_on']}")
        exit(1)

    print("\n" + "=" * 50)
    print("Stage 2: Quick Scoring (3 small graphs)")
    print("=" * 50)
    stage2_result = evaluate_stage2("initial_program.py")
    print(f"  Combined Score: {stage2_result.get('combined_score', 0):.4f}")
    print(f"  Avg Score: {stage2_result.get('stage2_avg_score', 0):.4f}")
    print(f"  All Valid: {stage2_result.get('stage2_all_valid', False)}")

    print("\n" + "=" * 50)
    print("Stage 3: Full Evaluation (10 graphs)")
    print("=" * 50)
    stage3_result = evaluate_stage3("initial_program.py")
    print(f"  Combined Score: {stage3_result.get('combined_score', 0):.4f}")
    print(f"  Avg Color Score: {stage3_result.get('avg_color_score', 0):.4f}")
    print(f"  Avg Time Score: {stage3_result.get('avg_time_score', 0):.4f}")
    print(f"  All Valid: {stage3_result.get('all_valid', False)}")
    print(f"  Optimal Count: {stage3_result.get('optimal_count', 0)}/{stage3_result.get('num_tests', 0)}")
    print(f"  Total Time: {stage3_result.get('total_time', 0):.3f}s")
    print(f"  Time Penalty Weight: {stage3_result.get('time_penalty_weight', 0):.0%}")

    if 'details' in stage3_result:
        print("\nPer-graph results:")
        for detail in stage3_result['details']:
            over_budget_flag = " [SLOW]" if detail.get('over_budget', False) else ""
            print(f"  {detail['name']}: {detail['colors']} colors, "
                  f"score={detail['score']:.3f} (color={detail.get('color_score', 0):.2f}, "
                  f"time={detail.get('time_score', 0):.2f}), "
                  f"{detail['time']*1000:.1f}ms{over_budget_flag}")
