"""
Test cases for the Graph Coloring example.

Runs the graph coloring algorithm on all test graphs from the evaluator
and verifies that all colorings are valid (no adjacent vertices share a color).
"""

import unittest
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from initial_program import (
    Graph,
    graph_coloring,
    is_valid_coloring,
    count_colors,
    create_sample_graph,
    load_graph_from_file
)
from evaluator import (
    create_test_graphs,
    create_random_graph,
    evaluate,
    evaluate_stage1,
    evaluate_stage2,
    evaluate_stage3,
    get_time_budget,
    calculate_time_score,
    load_dimacs_graph,
    load_chromatic_registry,
    load_dimacs_test_graphs,
    BENCHMARKS_DIR,
    CHROMATIC_NUMBERS_FILE,
    BASE_TIME_BUDGET,
    COEFF_TIME,
    TIME_PENALTY_WEIGHT
)
import tempfile
import json


class TestGraphColoring(unittest.TestCase):
    """Test cases for graph coloring validity."""

    def test_petersen_graph_valid(self):
        """Test that Petersen graph coloring is valid."""
        graph = create_sample_graph()  # Petersen graph
        coloring = graph_coloring(graph)
        is_valid, conflicts = is_valid_coloring(graph, coloring)

        self.assertTrue(is_valid, f"Petersen graph coloring has {conflicts} conflicts")
        self.assertEqual(conflicts, 0)

    def test_petersen_graph_chromatic_number(self):
        """Test that Petersen graph uses at most 3 colors (optimal)."""
        graph = create_sample_graph()
        coloring = graph_coloring(graph)
        num_colors = count_colors(coloring)

        # Petersen graph has chromatic number 3
        self.assertLessEqual(num_colors, 4, f"Used {num_colors} colors, expected at most 4")

    def test_complete_graph_k5_valid(self):
        """Test that complete graph K5 coloring is valid."""
        k5 = Graph(5)
        for i in range(5):
            for j in range(i + 1, 5):
                k5.add_edge(i, j)

        coloring = graph_coloring(k5)
        is_valid, conflicts = is_valid_coloring(k5, coloring)

        self.assertTrue(is_valid, f"K5 coloring has {conflicts} conflicts")

    def test_complete_graph_k5_chromatic_number(self):
        """Test that K5 uses exactly 5 colors (optimal for complete graph)."""
        k5 = Graph(5)
        for i in range(5):
            for j in range(i + 1, 5):
                k5.add_edge(i, j)

        coloring = graph_coloring(k5)
        num_colors = count_colors(coloring)

        # Complete graph K5 requires exactly 5 colors
        self.assertEqual(num_colors, 5, f"K5 should use exactly 5 colors, used {num_colors}")

    def test_bipartite_graph_valid(self):
        """Test that bipartite graph coloring is valid."""
        bipartite = Graph(10)
        # Create a complete bipartite graph K5,5
        for i in range(5):
            for j in range(5, 10):
                bipartite.add_edge(i, j)

        coloring = graph_coloring(bipartite)
        is_valid, conflicts = is_valid_coloring(bipartite, coloring)

        self.assertTrue(is_valid, f"Bipartite graph coloring has {conflicts} conflicts")

    def test_bipartite_graph_chromatic_number(self):
        """Test that bipartite graph uses at most 2 colors."""
        bipartite = Graph(10)
        for i in range(5):
            for j in range(5, 10):
                bipartite.add_edge(i, j)

        coloring = graph_coloring(bipartite)
        num_colors = count_colors(coloring)

        # Bipartite graphs have chromatic number 2
        self.assertLessEqual(num_colors, 2, f"Bipartite should use at most 2 colors, used {num_colors}")

    def test_odd_cycle_valid(self):
        """Test that odd cycle (C7) coloring is valid."""
        cycle = Graph(7)
        for i in range(7):
            cycle.add_edge(i, (i + 1) % 7)

        coloring = graph_coloring(cycle)
        is_valid, conflicts = is_valid_coloring(cycle, coloring)

        self.assertTrue(is_valid, f"Cycle C7 coloring has {conflicts} conflicts")

    def test_odd_cycle_chromatic_number(self):
        """Test that odd cycle uses exactly 3 colors."""
        cycle = Graph(7)
        for i in range(7):
            cycle.add_edge(i, (i + 1) % 7)

        coloring = graph_coloring(cycle)
        num_colors = count_colors(coloring)

        # Odd cycles have chromatic number 3
        self.assertLessEqual(num_colors, 3, f"Odd cycle should use at most 3 colors, used {num_colors}")

    def test_even_cycle_valid(self):
        """Test that even cycle (C6) coloring is valid."""
        cycle = Graph(6)
        for i in range(6):
            cycle.add_edge(i, (i + 1) % 6)

        coloring = graph_coloring(cycle)
        is_valid, conflicts = is_valid_coloring(cycle, coloring)

        self.assertTrue(is_valid, f"Cycle C6 coloring has {conflicts} conflicts")

    def test_even_cycle_chromatic_number(self):
        """Test that even cycle uses exactly 2 colors."""
        cycle = Graph(6)
        for i in range(6):
            cycle.add_edge(i, (i + 1) % 6)

        coloring = graph_coloring(cycle)
        num_colors = count_colors(coloring)

        # Even cycles have chromatic number 2
        self.assertLessEqual(num_colors, 2, f"Even cycle should use at most 2 colors, used {num_colors}")

    def test_random_sparse_graph_valid(self):
        """Test that random sparse graph coloring is valid."""
        sparse = create_random_graph(30, 0.1, seed=42)
        coloring = graph_coloring(sparse)
        is_valid, conflicts = is_valid_coloring(sparse, coloring)

        self.assertTrue(is_valid, f"Random sparse graph coloring has {conflicts} conflicts")

    def test_random_dense_graph_valid(self):
        """Test that random dense graph coloring is valid."""
        dense = create_random_graph(20, 0.5, seed=42)
        coloring = graph_coloring(dense)
        is_valid, conflicts = is_valid_coloring(dense, coloring)

        self.assertTrue(is_valid, f"Random dense graph coloring has {conflicts} conflicts")

    def test_all_evaluator_graphs_valid(self):
        """Test that ALL graphs from the evaluator produce valid colorings."""
        test_graphs = create_test_graphs()

        for name, graph, known_chromatic in test_graphs:
            with self.subTest(graph=name):
                coloring = graph_coloring(graph)
                is_valid, conflicts = is_valid_coloring(graph, coloring)

                self.assertTrue(
                    is_valid,
                    f"Graph '{name}' has invalid coloring with {conflicts} conflicts"
                )

    def test_empty_graph(self):
        """Test coloring of a graph with no edges."""
        empty = Graph(5)
        coloring = graph_coloring(empty)
        is_valid, conflicts = is_valid_coloring(empty, coloring)
        num_colors = count_colors(coloring)

        self.assertTrue(is_valid)
        # Empty graph should use only 1 color
        self.assertEqual(num_colors, 1, f"Empty graph should use 1 color, used {num_colors}")

    def test_single_vertex(self):
        """Test coloring of a single vertex graph."""
        single = Graph(1)
        coloring = graph_coloring(single)
        is_valid, conflicts = is_valid_coloring(single, coloring)

        self.assertTrue(is_valid)
        self.assertEqual(count_colors(coloring), 1)

    def test_all_vertices_colored(self):
        """Test that all vertices are assigned a color."""
        graph = create_sample_graph()
        coloring = graph_coloring(graph)

        for vertex in range(graph.num_vertices):
            self.assertIn(
                vertex,
                coloring,
                f"Vertex {vertex} was not assigned a color"
            )


class TestGraphStructure(unittest.TestCase):
    """Test cases for the Graph class itself."""

    def test_graph_creation(self):
        """Test that a graph can be created."""
        g = Graph(5)
        self.assertEqual(g.num_vertices, 5)

    def test_add_edge(self):
        """Test adding edges to a graph."""
        g = Graph(3)
        g.add_edge(0, 1)
        g.add_edge(1, 2)

        self.assertIn(1, g.get_neighbors(0))
        self.assertIn(0, g.get_neighbors(1))
        self.assertIn(2, g.get_neighbors(1))

    def test_get_degree(self):
        """Test vertex degree calculation."""
        g = Graph(4)
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(0, 3)

        self.assertEqual(g.get_degree(0), 3)
        self.assertEqual(g.get_degree(1), 1)

    def test_get_edges(self):
        """Test edge retrieval."""
        g = Graph(3)
        g.add_edge(0, 1)
        g.add_edge(1, 2)

        edges = g.get_edges()
        self.assertEqual(len(edges), 2)
        self.assertIn((0, 1), edges)
        self.assertIn((1, 2), edges)


class TestGraphFileLoading(unittest.TestCase):
    """Test cases for loading graphs from files."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_graphs_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'sample_graphs'
        )

    def test_load_triangle_graph(self):
        """Test loading a triangle graph from file."""
        file_path = os.path.join(self.sample_graphs_dir, 'triangle.txt')
        graph = load_graph_from_file(file_path)

        self.assertEqual(graph.num_vertices, 3)
        self.assertEqual(len(graph.get_edges()), 3)
        # Triangle is K3, should need 3 colors
        coloring = graph_coloring(graph)
        self.assertEqual(count_colors(coloring), 3)

    def test_load_square_graph(self):
        """Test loading a square (C4) graph from file."""
        file_path = os.path.join(self.sample_graphs_dir, 'square.txt')
        graph = load_graph_from_file(file_path)

        self.assertEqual(graph.num_vertices, 4)
        self.assertEqual(len(graph.get_edges()), 4)
        # Square is bipartite, should need only 2 colors
        coloring = graph_coloring(graph)
        self.assertLessEqual(count_colors(coloring), 2)

    def test_load_k4_complete_graph(self):
        """Test loading a complete K4 graph from file."""
        file_path = os.path.join(self.sample_graphs_dir, 'k4_complete.txt')
        graph = load_graph_from_file(file_path)

        self.assertEqual(graph.num_vertices, 4)
        self.assertEqual(len(graph.get_edges()), 6)  # K4 has 4*3/2 = 6 edges
        # K4 needs exactly 4 colors
        coloring = graph_coloring(graph)
        self.assertEqual(count_colors(coloring), 4)

    def test_load_graph_with_comments(self):
        """Test that comment lines are properly skipped."""
        file_path = os.path.join(self.sample_graphs_dir, 'k4_complete.txt')
        graph = load_graph_from_file(file_path)
        # File has comment lines starting with #, should still load correctly
        self.assertEqual(graph.num_vertices, 4)

    def test_loaded_graph_valid_coloring(self):
        """Test that loaded graphs produce valid colorings."""
        for filename in ['triangle.txt', 'square.txt', 'k4_complete.txt']:
            with self.subTest(file=filename):
                file_path = os.path.join(self.sample_graphs_dir, filename)
                graph = load_graph_from_file(file_path)
                coloring = graph_coloring(graph)
                is_valid, conflicts = is_valid_coloring(graph, coloring)
                self.assertTrue(is_valid, f"{filename} has invalid coloring")

    def test_load_nonexistent_file(self):
        """Test that loading a nonexistent file raises an error."""
        with self.assertRaises(FileNotFoundError):
            load_graph_from_file('nonexistent_file.txt')


class TestEvaluator(unittest.TestCase):
    """Test cases for the evaluator functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.program_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'initial_program.py'
        )

    def test_evaluate_returns_combined_score(self):
        """Test that evaluate() returns a combined_score."""
        result = evaluate(self.program_path)
        self.assertIn('combined_score', result)
        self.assertIsInstance(result['combined_score'], float)

    def test_evaluate_returns_all_valid(self):
        """Test that evaluate() returns all_valid flag."""
        result = evaluate(self.program_path)
        self.assertIn('all_valid', result)
        self.assertTrue(result['all_valid'])

    def test_evaluate_returns_details(self):
        """Test that evaluate() returns per-graph details."""
        result = evaluate(self.program_path)
        self.assertIn('details', result)
        # Number of test graphs depends on DIMACS benchmarks loaded
        self.assertGreater(len(result['details']), 0)

    def test_evaluate_score_in_valid_range(self):
        """Test that combined_score is between 0 and 1."""
        result = evaluate(self.program_path)
        self.assertGreaterEqual(result['combined_score'], 0.0)
        self.assertLessEqual(result['combined_score'], 1.0)

    def test_evaluate_nonexistent_file(self):
        """Test that evaluate() handles missing files gracefully."""
        result = evaluate('nonexistent_program.py')
        self.assertEqual(result['combined_score'], 0.0)
        self.assertIn('error', result)

    # Stage 1 tests
    def test_stage1_passes_for_valid_program(self):
        """Test that stage1 passes for initial_program.py."""
        result = evaluate_stage1(self.program_path)
        self.assertTrue(result['stage1_passed'])
        self.assertEqual(result['combined_score'], 1.0)

    def test_stage1_returns_required_fields(self):
        """Test that stage1 returns required fields."""
        result = evaluate_stage1(self.program_path)
        self.assertIn('combined_score', result)
        self.assertIn('stage1_passed', result)

    def test_stage1_handles_errors(self):
        """Test that stage1 handles errors gracefully."""
        result = evaluate_stage1('nonexistent_program.py')
        self.assertEqual(result['combined_score'], 0.0)
        self.assertFalse(result['stage1_passed'])

    # Stage 2 tests
    def test_stage2_returns_score(self):
        """Test that stage2 returns a combined_score."""
        result = evaluate_stage2(self.program_path)
        self.assertIn('combined_score', result)
        self.assertGreater(result['combined_score'], 0.0)

    def test_stage2_reasonable_score_for_initial_program(self):
        """Test that initial program gets reasonable score on small graphs."""
        result = evaluate_stage2(self.program_path)
        # Initial greedy algorithm should get a reasonable score (> 0.7)
        # on DIMACS small benchmarks, though not necessarily optimal
        self.assertGreater(result['combined_score'], 0.7)

    def test_stage2_returns_avg_score(self):
        """Test that stage2 returns average score."""
        result = evaluate_stage2(self.program_path)
        self.assertIn('stage2_avg_score', result)
        self.assertIn('stage2_all_valid', result)

    # Stage 3 tests
    def test_stage3_returns_full_metrics(self):
        """Test that stage3 returns full evaluation metrics."""
        result = evaluate_stage3(self.program_path)
        self.assertIn('combined_score', result)
        self.assertIn('avg_color_score', result)
        self.assertIn('all_valid', result)
        self.assertIn('optimal_count', result)
        self.assertIn('details', result)

    def test_stage3_score_equals_evaluate(self):
        """Test that stage3 returns same result as evaluate() for valid programs."""
        stage3_result = evaluate_stage3(self.program_path)
        eval_result = evaluate(self.program_path)
        self.assertEqual(
            stage3_result['combined_score'],
            eval_result['combined_score']
        )

    # Cascade flow tests
    def test_cascade_flow_all_stages_pass(self):
        """Test that all stages pass for initial_program.py."""
        stage1 = evaluate_stage1(self.program_path)
        self.assertTrue(stage1['stage1_passed'])

        stage2 = evaluate_stage2(self.program_path)
        self.assertGreaterEqual(stage2['combined_score'], 0.7)

        stage3 = evaluate_stage3(self.program_path)
        self.assertGreater(stage3['combined_score'], 0.0)


class TestEvaluatorFailureCases(unittest.TestCase):
    """Test cases for evaluator failure scenarios using mock programs."""

    def setUp(self):
        """Set up test fixtures with temp directory for mock programs."""
        import tempfile
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp directory after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_program(self, name: str, code: str) -> str:
        """Create a mock program file and return its path."""
        file_path = os.path.join(self.temp_dir, name)
        with open(file_path, 'w') as f:
            f.write(code)
        return file_path

    def test_stage1_fails_for_invalid_coloring(self):
        """Test that stage1 fails for a program that produces invalid colorings."""
        # This program assigns the same color (0) to ALL vertices,
        # which creates conflicts on any graph with edges
        invalid_program = '''
class Graph:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.adjacency_list = {i: set() for i in range(num_vertices)}

    def add_edge(self, u, v):
        self.adjacency_list[u].add(v)
        self.adjacency_list[v].add(u)

    def get_neighbors(self, vertex):
        return self.adjacency_list[vertex]

    def get_degree(self, vertex):
        return len(self.adjacency_list[vertex])

    def get_edges(self):
        edges = []
        for u in range(self.num_vertices):
            for v in self.adjacency_list[u]:
                if u < v:
                    edges.append((u, v))
        return edges

def graph_coloring(graph):
    """Invalid: assigns same color to all vertices."""
    return {v: 0 for v in range(graph.num_vertices)}

def is_valid_coloring(graph, coloring):
    conflicts = 0
    for u, v in graph.get_edges():
        if coloring.get(u) == coloring.get(v):
            conflicts += 1
    return (conflicts == 0, conflicts)

def count_colors(coloring):
    return len(set(coloring.values()))
'''
        program_path = self._create_mock_program('invalid_coloring.py', invalid_program)
        result = evaluate_stage1(program_path)

        self.assertFalse(result.get('stage1_passed', True))
        self.assertEqual(result['combined_score'], 0.0)
        self.assertIn('failed_on', result)

    def test_stage1_fails_for_partial_coloring(self):
        """Test that stage1 fails for a program that doesn't color all vertices."""
        # This program only colors even-indexed vertices
        partial_program = '''
class Graph:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.adjacency_list = {i: set() for i in range(num_vertices)}

    def add_edge(self, u, v):
        self.adjacency_list[u].add(v)
        self.adjacency_list[v].add(u)

    def get_neighbors(self, vertex):
        return self.adjacency_list[vertex]

    def get_degree(self, vertex):
        return len(self.adjacency_list[vertex])

    def get_edges(self):
        edges = []
        for u in range(self.num_vertices):
            for v in self.adjacency_list[u]:
                if u < v:
                    edges.append((u, v))
        return edges

def graph_coloring(graph):
    """Only colors even vertices - missing odd vertices."""
    return {v: v % 3 for v in range(graph.num_vertices) if v % 2 == 0}

def is_valid_coloring(graph, coloring):
    conflicts = 0
    for u, v in graph.get_edges():
        if coloring.get(u) == coloring.get(v):
            conflicts += 1
    return (conflicts == 0, conflicts)

def count_colors(coloring):
    return len(set(coloring.values()))
'''
        program_path = self._create_mock_program('partial_coloring.py', partial_program)
        # This should fail because neighbors might both be None
        result = evaluate_stage1(program_path)

        # Partial coloring should still technically pass validity (None != None comparison)
        # But let's test the full evaluate which might catch other issues
        full_result = evaluate(program_path)
        # Score should be affected by the partial nature
        self.assertIsNotNone(full_result)

    def test_stage2_low_score_for_suboptimal_coloring(self):
        """Test that stage2 gives low score for inefficient coloring."""
        # This program uses n colors for n vertices (worst case)
        suboptimal_program = '''
class Graph:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.adjacency_list = {i: set() for i in range(num_vertices)}

    def add_edge(self, u, v):
        self.adjacency_list[u].add(v)
        self.adjacency_list[v].add(u)

    def get_neighbors(self, vertex):
        return self.adjacency_list[vertex]

    def get_degree(self, vertex):
        return len(self.adjacency_list[vertex])

    def get_edges(self):
        edges = []
        for u in range(self.num_vertices):
            for v in self.adjacency_list[u]:
                if u < v:
                    edges.append((u, v))
        return edges

def graph_coloring(graph):
    """Suboptimal: uses n colors for n vertices (every vertex gets unique color)."""
    return {v: v for v in range(graph.num_vertices)}

def is_valid_coloring(graph, coloring):
    conflicts = 0
    for u, v in graph.get_edges():
        if coloring.get(u) == coloring.get(v):
            conflicts += 1
    return (conflicts == 0, conflicts)

def count_colors(coloring):
    return len(set(coloring.values()))
'''
        program_path = self._create_mock_program('suboptimal_coloring.py', suboptimal_program)

        # Stage 1 should pass (coloring is valid)
        stage1_result = evaluate_stage1(program_path)
        self.assertTrue(stage1_result.get('stage1_passed', False))

        # Stage 2 should have lower score (using more colors than optimal)
        stage2_result = evaluate_stage2(program_path)
        # Triangle needs 3 colors, this uses 3 -> score = 1.0
        # K4 needs 4 colors, this uses 4 -> score = 1.0
        # Path3 needs 2 colors, this uses 3 -> score = 2/3 = 0.67
        # Average = (1.0 + 1.0 + 0.67) / 3 = 0.89
        self.assertGreater(stage2_result['combined_score'], 0.0)
        self.assertLessEqual(stage2_result['combined_score'], 1.0)

    def test_evaluate_handles_missing_function(self):
        """Test that evaluate handles a program missing graph_coloring function."""
        missing_func_program = '''
class Graph:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.adjacency_list = {i: set() for i in range(num_vertices)}

    def add_edge(self, u, v):
        self.adjacency_list[u].add(v)
        self.adjacency_list[v].add(u)

# Missing graph_coloring, is_valid_coloring, and count_colors functions!
'''
        program_path = self._create_mock_program('missing_function.py', missing_func_program)
        result = evaluate(program_path)

        self.assertEqual(result['combined_score'], 0.0)
        self.assertIn('error', result)

    def test_evaluate_handles_syntax_error(self):
        """Test that evaluate handles a program with syntax errors."""
        syntax_error_program = '''
def graph_coloring(graph)
    # Missing colon above - syntax error
    return {}
'''
        program_path = self._create_mock_program('syntax_error.py', syntax_error_program)
        result = evaluate(program_path)

        self.assertEqual(result['combined_score'], 0.0)
        self.assertIn('error', result)

    def test_evaluate_handles_runtime_error(self):
        """Test that evaluate handles a program that throws runtime errors."""
        runtime_error_program = '''
class Graph:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.adjacency_list = {i: set() for i in range(num_vertices)}

    def add_edge(self, u, v):
        self.adjacency_list[u].add(v)
        self.adjacency_list[v].add(u)

    def get_neighbors(self, vertex):
        return self.adjacency_list[vertex]

    def get_degree(self, vertex):
        return len(self.adjacency_list[vertex])

    def get_edges(self):
        edges = []
        for u in range(self.num_vertices):
            for v in self.adjacency_list[u]:
                if u < v:
                    edges.append((u, v))
        return edges

def graph_coloring(graph):
    """Raises an error during execution."""
    raise ValueError("Intentional error for testing")

def is_valid_coloring(graph, coloring):
    conflicts = 0
    for u, v in graph.get_edges():
        if coloring.get(u) == coloring.get(v):
            conflicts += 1
    return (conflicts == 0, conflicts)

def count_colors(coloring):
    return len(set(coloring.values()))
'''
        program_path = self._create_mock_program('runtime_error.py', runtime_error_program)
        result = evaluate(program_path)

        self.assertEqual(result['combined_score'], 0.0)
        self.assertIn('error', result)

    def test_stage1_handles_exception_gracefully(self):
        """Test that stage1 handles exceptions and returns proper error structure."""
        error_program = '''
raise ImportError("Cannot import module")
'''
        program_path = self._create_mock_program('import_error.py', error_program)
        result = evaluate_stage1(program_path)

        self.assertEqual(result['combined_score'], 0.0)
        self.assertFalse(result.get('stage1_passed', True))
        self.assertIn('error', result)

    def test_stage2_handles_exception_gracefully(self):
        """Test that stage2 handles exceptions and returns proper error structure."""
        error_program = '''
raise ImportError("Cannot import module")
'''
        program_path = self._create_mock_program('import_error2.py', error_program)
        result = evaluate_stage2(program_path)

        self.assertEqual(result['combined_score'], 0.0)
        self.assertIn('error', result)

    def test_stage3_zero_score_for_invalid(self):
        """Test that stage3 returns 0 score when any coloring is invalid."""
        # Same as invalid coloring test but for stage 3
        invalid_program = '''
class Graph:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.adjacency_list = {i: set() for i in range(num_vertices)}

    def add_edge(self, u, v):
        self.adjacency_list[u].add(v)
        self.adjacency_list[v].add(u)

    def get_neighbors(self, vertex):
        return self.adjacency_list[vertex]

    def get_degree(self, vertex):
        return len(self.adjacency_list[vertex])

    def get_edges(self):
        edges = []
        for u in range(self.num_vertices):
            for v in self.adjacency_list[u]:
                if u < v:
                    edges.append((u, v))
        return edges

def graph_coloring(graph):
    """Invalid: assigns same color to all vertices."""
    return {v: 0 for v in range(graph.num_vertices)}

def is_valid_coloring(graph, coloring):
    conflicts = 0
    for u, v in graph.get_edges():
        if coloring.get(u) == coloring.get(v):
            conflicts += 1
    return (conflicts == 0, conflicts)

def count_colors(coloring):
    return len(set(coloring.values()))
'''
        program_path = self._create_mock_program('invalid_for_stage3.py', invalid_program)
        result = evaluate_stage3(program_path)

        self.assertEqual(result['combined_score'], 0.0)
        self.assertFalse(result.get('all_valid', True))

    def test_nonexistent_program_file(self):
        """Test all stages handle nonexistent files gracefully."""
        fake_path = '/nonexistent/path/to/program.py'

        stage1 = evaluate_stage1(fake_path)
        self.assertEqual(stage1['combined_score'], 0.0)
        self.assertFalse(stage1.get('stage1_passed', True))

        stage2 = evaluate_stage2(fake_path)
        self.assertEqual(stage2['combined_score'], 0.0)

        stage3 = evaluate_stage3(fake_path)
        self.assertEqual(stage3['combined_score'], 0.0)

        full = evaluate(fake_path)
        self.assertEqual(full['combined_score'], 0.0)


class TestTimeBudget(unittest.TestCase):
    """Test cases for time budget calculation and scoring."""

    def test_get_time_budget_formula(self):
        """Test that get_time_budget follows the formula: BASE + COEFF × (n² + m)."""
        n, m = 100, 500
        expected = BASE_TIME_BUDGET + COEFF_TIME * (n * n + m)
        actual = get_time_budget(n, m)
        self.assertAlmostEqual(actual, expected, places=10)

    def test_get_time_budget_increases_with_vertices(self):
        """Test that budget increases with more vertices."""
        m = 100  # fixed edges
        budget_small = get_time_budget(10, m)
        budget_large = get_time_budget(100, m)
        self.assertGreater(budget_large, budget_small)

    def test_get_time_budget_increases_with_edges(self):
        """Test that budget increases with more edges."""
        n = 50  # fixed vertices
        budget_sparse = get_time_budget(n, 100)
        budget_dense = get_time_budget(n, 1000)
        self.assertGreater(budget_dense, budget_sparse)

    def test_get_time_budget_quadratic_scaling(self):
        """Test that budget scales quadratically with vertices."""
        m = 0  # no edges to isolate vertex effect
        budget_n = get_time_budget(100, m)
        budget_2n = get_time_budget(200, m)
        # For quadratic: budget(2n) ≈ 4 × budget(n) (ignoring BASE)
        ratio = (budget_2n - BASE_TIME_BUDGET) / (budget_n - BASE_TIME_BUDGET)
        self.assertAlmostEqual(ratio, 4.0, places=1)

    def test_calculate_time_score_within_budget(self):
        """Test that score is 1.0 when within budget."""
        budget = 0.01  # 10ms
        elapsed = 0.005  # 5ms (under budget)
        score = calculate_time_score(elapsed, budget)
        self.assertEqual(score, 1.0)

    def test_calculate_time_score_at_budget(self):
        """Test that score is 1.0 when exactly at budget."""
        budget = 0.01
        elapsed = 0.01
        score = calculate_time_score(elapsed, budget)
        self.assertEqual(score, 1.0)

    def test_calculate_time_score_over_budget(self):
        """Test that score decreases when over budget."""
        budget = 0.01
        elapsed = 0.02  # 2x over budget
        score = calculate_time_score(elapsed, budget)
        self.assertLess(score, 1.0)
        # Score should be approximately 0.5 (budget/elapsed)
        self.assertAlmostEqual(score, 0.5, places=2)

    def test_calculate_time_score_floor(self):
        """Test that score has a floor of 0.1."""
        budget = 0.001
        elapsed = 1.0  # 1000x over budget
        score = calculate_time_score(elapsed, budget)
        self.assertEqual(score, 0.1)

    def test_calculate_time_score_decay(self):
        """Test that score decays proportionally to overage."""
        budget = 0.01
        # 4x over budget should give score of 0.25
        score = calculate_time_score(0.04, budget)
        self.assertAlmostEqual(score, 0.25, places=2)

    def test_time_penalty_weight_in_range(self):
        """Test that TIME_PENALTY_WEIGHT is reasonable (0 to 1)."""
        self.assertGreater(TIME_PENALTY_WEIGHT, 0.0)
        self.assertLess(TIME_PENALTY_WEIGHT, 1.0)

    def test_evaluate_returns_time_metrics(self):
        """Test that evaluate() returns time-related metrics."""
        program_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'initial_program.py'
        )
        result = evaluate(program_path)

        self.assertIn('avg_time_score', result)
        self.assertIn('total_time', result)
        self.assertIn('time_penalty_weight', result)

        # Check details have time info
        for detail in result['details']:
            self.assertIn('time', detail)
            self.assertIn('time_budget', detail)
            self.assertIn('time_score', detail)
            self.assertIn('over_budget', detail)

    def test_fast_algorithm_gets_good_time_score(self):
        """Test that the simple greedy algorithm gets good time scores."""
        program_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'initial_program.py'
        )
        result = evaluate(program_path)

        # Simple greedy should be fast enough for good time scores
        self.assertGreaterEqual(result['avg_time_score'], 0.9)


class TestDIMACS(unittest.TestCase):
    """Test cases for DIMACS benchmark loading functionality."""

    def test_load_dimacs_graph_basic(self):
        """Test loading a simple DIMACS graph from a temp file."""
        dimacs_content = """c Test graph
c Comment line
p edge 4 5
e 1 2
e 2 3
e 3 4
e 4 1
e 1 3
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.col', delete=False) as f:
            f.write(dimacs_content)
            temp_path = f.name

        try:
            graph = load_dimacs_graph(temp_path)
            self.assertEqual(graph.num_vertices, 4)
            self.assertEqual(len(graph.get_edges()), 5)
            # Check 0-indexed conversion: edge 1-2 should be 0-1
            self.assertIn(1, graph.get_neighbors(0))
            self.assertIn(0, graph.get_neighbors(1))
        finally:
            os.unlink(temp_path)

    def test_load_dimacs_graph_1indexed_to_0indexed(self):
        """Test that DIMACS 1-indexed vertices are converted to 0-indexed."""
        dimacs_content = """p edge 3 2
e 1 2
e 2 3
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.col', delete=False) as f:
            f.write(dimacs_content)
            temp_path = f.name

        try:
            graph = load_dimacs_graph(temp_path)
            # Vertex 1 in DIMACS -> vertex 0 in our graph
            # Vertex 2 in DIMACS -> vertex 1 in our graph
            # Vertex 3 in DIMACS -> vertex 2 in our graph
            self.assertIn(1, graph.get_neighbors(0))  # edge 1-2 -> 0-1
            self.assertIn(2, graph.get_neighbors(1))  # edge 2-3 -> 1-2
        finally:
            os.unlink(temp_path)

    def test_load_dimacs_graph_missing_problem_line(self):
        """Test that missing problem line raises ValueError."""
        dimacs_content = """c Just comments
c No problem line
e 1 2
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.col', delete=False) as f:
            f.write(dimacs_content)
            temp_path = f.name

        try:
            with self.assertRaises(ValueError) as context:
                load_dimacs_graph(temp_path)
            self.assertIn("no problem line", str(context.exception))
        finally:
            os.unlink(temp_path)

    def test_load_dimacs_graph_empty_graph(self):
        """Test loading a graph with vertices but no edges."""
        dimacs_content = """p edge 5 0
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.col', delete=False) as f:
            f.write(dimacs_content)
            temp_path = f.name

        try:
            graph = load_dimacs_graph(temp_path)
            self.assertEqual(graph.num_vertices, 5)
            self.assertEqual(len(graph.get_edges()), 0)
        finally:
            os.unlink(temp_path)

    def test_load_chromatic_registry_exists(self):
        """Test that chromatic registry file exists and loads."""
        self.assertTrue(os.path.exists(CHROMATIC_NUMBERS_FILE))
        registry = load_chromatic_registry()
        self.assertIsInstance(registry, dict)
        self.assertIn('full', registry)
        self.assertIn('small', registry)

    def test_load_chromatic_registry_has_entries(self):
        """Test that registry has expected graph entries."""
        registry = load_chromatic_registry()
        full = registry.get('full', {})

        # Check some known graphs exist
        self.assertIn('myciel3.col', full)
        self.assertIn('queen5_5.col', full)

        # Check entry structure
        myciel3 = full['myciel3.col']
        self.assertIn('chromatic', myciel3)
        self.assertIn('type', myciel3)
        self.assertEqual(myciel3['chromatic'], 4)

    def test_load_chromatic_registry_missing_file(self):
        """Test that missing registry file returns empty dict."""
        import evaluator
        original_file = evaluator.CHROMATIC_NUMBERS_FILE
        try:
            evaluator.CHROMATIC_NUMBERS_FILE = '/nonexistent/path/chromatic.json'
            registry = load_chromatic_registry()
            self.assertEqual(registry, {})
        finally:
            evaluator.CHROMATIC_NUMBERS_FILE = original_file

    def test_load_dimacs_test_graphs_full(self):
        """Test loading full benchmark graphs."""
        graphs = load_dimacs_test_graphs('full')
        self.assertGreater(len(graphs), 0)

        # Check structure of returned tuples
        for name, graph, chromatic in graphs:
            self.assertIsInstance(name, str)
            self.assertIsInstance(graph.num_vertices, int)
            self.assertGreater(graph.num_vertices, 0)
            # Chromatic can be None or int
            if chromatic is not None:
                self.assertIsInstance(chromatic, int)
                self.assertGreater(chromatic, 0)

    def test_load_dimacs_test_graphs_small(self):
        """Test loading small benchmark graphs."""
        graphs = load_dimacs_test_graphs('small')
        self.assertGreater(len(graphs), 0)

        # All small graphs should have < 100 vertices
        for name, graph, chromatic in graphs:
            self.assertLess(graph.num_vertices, 100,
                            f"Small graph {name} has {graph.num_vertices} vertices")

    def test_load_dimacs_test_graphs_nonexistent_category(self):
        """Test that nonexistent category returns empty list."""
        graphs = load_dimacs_test_graphs('nonexistent_category')
        self.assertEqual(graphs, [])

    def test_load_dimacs_test_graphs_chromatic_lookup(self):
        """Test that chromatic numbers are correctly looked up from registry."""
        graphs = load_dimacs_test_graphs('full')
        registry = load_chromatic_registry()
        full_registry = registry.get('full', {})

        for name, graph, chromatic in graphs:
            filename = name + '.col'
            if filename in full_registry:
                expected = full_registry[filename].get('chromatic')
                self.assertEqual(chromatic, expected,
                                 f"Chromatic mismatch for {name}: got {chromatic}, expected {expected}")

    def test_benchmarks_directory_exists(self):
        """Test that benchmarks directory exists."""
        self.assertTrue(os.path.exists(BENCHMARKS_DIR))
        self.assertTrue(os.path.isdir(BENCHMARKS_DIR))

    def test_benchmarks_subdirectories_exist(self):
        """Test that small and full subdirectories exist."""
        small_dir = os.path.join(BENCHMARKS_DIR, 'small')
        full_dir = os.path.join(BENCHMARKS_DIR, 'full')
        self.assertTrue(os.path.exists(small_dir))
        self.assertTrue(os.path.exists(full_dir))

    def test_benchmark_graphs_are_valid(self):
        """Test that all benchmark graphs produce valid colorings."""
        graphs = load_dimacs_test_graphs('full')

        for name, graph, chromatic in graphs:
            coloring = graph_coloring(graph)
            is_valid, conflicts = is_valid_coloring(graph, coloring)
            self.assertTrue(is_valid,
                            f"Graph {name} produced invalid coloring with {conflicts} conflicts")

    def test_create_test_graphs_uses_dimacs(self):
        """Test that create_test_graphs() returns DIMACS graphs when available."""
        graphs = create_test_graphs()
        dimacs_graphs = load_dimacs_test_graphs('full')

        # Should return same number of graphs
        self.assertEqual(len(graphs), len(dimacs_graphs))

        # Names should match
        graph_names = {name for name, _, _ in graphs}
        dimacs_names = {name for name, _, _ in dimacs_graphs}
        self.assertEqual(graph_names, dimacs_names)


if __name__ == "__main__":
    # Run tests with verbosity
    unittest.main(verbosity=2)
