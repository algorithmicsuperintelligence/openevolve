#!/usr/bin/env python3
"""
Run graph coloring algorithm on all test graphs and produce a results table.

Usage:
    python run_benchmarks.py                    # Run on initial_program.py
    python run_benchmarks.py best_program.py   # Run on evolved program
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from initial_program import graph_coloring, is_valid_coloring, count_colors

# Try to import from evaluator for DIMACS support
try:
    from evaluator import load_dimacs_graph, load_chromatic_registry, BENCHMARKS_DIR
    HAS_DIMACS = True
except ImportError:
    HAS_DIMACS = False


def load_program(program_path):
    """Load graph_coloring function from a program file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("module", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.graph_coloring


def get_dimacs_graphs():
    """Get DIMACS benchmark graphs with chromatic numbers."""
    if not HAS_DIMACS:
        return []

    graphs = []
    registry = load_chromatic_registry()
    full_registry = registry.get('full', {})

    full_dir = os.path.join(BENCHMARKS_DIR, 'full')
    if not os.path.exists(full_dir):
        return []

    for filename in sorted(os.listdir(full_dir)):
        if not filename.endswith('.col'):
            continue

        filepath = os.path.join(full_dir, filename)
        try:
            graph = load_dimacs_graph(filepath)
            chromatic = full_registry.get(filename, {}).get('chromatic')
            name = filename.replace('.col', '')
            graphs.append((name, graph, chromatic))
        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}", file=sys.stderr)

    return graphs


def run_benchmarks(coloring_func):
    """Run benchmarks and return results."""
    results = []

    # DIMACS graphs only (matches evaluator Stage 3 test set)
    if HAS_DIMACS:
        print("Loading DIMACS benchmarks...", file=sys.stderr)
        for name, graph, chromatic in get_dimacs_graphs():
            coloring = coloring_func(graph)
            is_valid, conflicts = is_valid_coloring(graph, coloring)
            num_colors = count_colors(coloring)
            results.append({
                'name': name,
                'vertices': graph.num_vertices,
                'edges': len(graph.get_edges()),
                'colors': num_colors,
                'chromatic': chromatic,
                'valid': is_valid,
                'optimal': num_colors == chromatic if chromatic else None
            })
    else:
        print("Warning: DIMACS support not available. No benchmarks to run.", file=sys.stderr)
        print("Make sure evaluator.py is in the same directory.", file=sys.stderr)

    return results


def print_results(results):
    """Print results as a formatted table."""
    print()
    print("=" * 85)
    print(f"{'Graph':<20} {'V':>6} {'E':>7} {'Colors':>7} {'χ':>5} {'Valid':>7} {'Optimal':>8}")
    print("=" * 85)

    total_colors = 0
    total_chromatic = 0
    optimal_count = 0
    valid_count = 0

    for r in results:
        chromatic_str = str(r['chromatic']) if r['chromatic'] else '?'
        valid_str = '✓' if r['valid'] else '✗'

        if r['optimal'] is None:
            optimal_str = '?'
        elif r['optimal']:
            optimal_str = '✓'
            optimal_count += 1
        else:
            optimal_str = f"+{r['colors'] - r['chromatic']}"

        if r['valid']:
            valid_count += 1

        total_colors += r['colors']
        if r['chromatic']:
            total_chromatic += r['chromatic']

        print(f"{r['name']:<20} {r['vertices']:>6} {r['edges']:>7} {r['colors']:>7} {chromatic_str:>5} {valid_str:>7} {optimal_str:>8}")

    print("=" * 85)
    print(f"{'TOTAL':<20} {'':<6} {'':<7} {total_colors:>7} {total_chromatic:>5} {valid_count:>7} {optimal_count:>8}")
    print("=" * 85)
    print()
    print(f"Summary: {optimal_count}/{len(results)} optimal, {valid_count}/{len(results)} valid")
    print(f"Total colors used: {total_colors} (optimal: {total_chromatic}, excess: {total_colors - total_chromatic})")


def main():
    # Determine which program to test
    if len(sys.argv) > 1:
        program_path = sys.argv[1]
        if not os.path.isabs(program_path):
            program_path = os.path.join(os.path.dirname(__file__), program_path)
        print(f"Loading coloring function from: {program_path}", file=sys.stderr)
        coloring_func = load_program(program_path)
    else:
        print("Using built-in graph_coloring function", file=sys.stderr)
        coloring_func = graph_coloring

    results = run_benchmarks(coloring_func)
    print_results(results)


if __name__ == '__main__':
    main()
