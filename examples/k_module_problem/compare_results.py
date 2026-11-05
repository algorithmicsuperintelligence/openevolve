#!/usr/bin/env python3
"""
Compare results from OpenEvolve and Iterative Agent on K-Module Problem.

This script analyzes the outputs from both approaches and generates
comparison plots showing:
1. Convergence speed (iterations to solution)
2. Best score achieved over iterations
3. Total LLM calls made

Usage:
    python compare_results.py [--openevolve-dir DIR] [--iterative-dir DIR]
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from openevolve.checkpoint import load_checkpoint


def load_openevolve_results(output_dir: str) -> dict:
    """Load results from OpenEvolve checkpoint."""
    results = {
        "iterations": [],
        "scores": [],
        "best_scores": [],
        "solution_found_at": None,
    }

    # Find the latest checkpoint
    checkpoint_dir = Path(output_dir) / "checkpoints"
    if not checkpoint_dir.exists():
        print(f"Warning: No checkpoints found in {output_dir}")
        return results

    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*"))
    if not checkpoints:
        return results

    latest_checkpoint = checkpoints[-1]
    programs_dir = latest_checkpoint / "programs"

    if not programs_dir.exists():
        return results

    # Load all program results
    programs = []
    for data in load_checkpoint(str(latest_checkpoint)).programs:
        if "iteration_found" in data and "metrics" in data:
            programs.append({
                "iteration": data["iteration_found"],
                "score": data["metrics"].get("combined_score", 0),
                "correct_modules": data["metrics"].get("correct_modules", 0),
                "timestamp": data.get("timestamp", 0),
            })

    # Sort by timestamp
    programs.sort(key=lambda x: x["timestamp"])

    # Build iteration-by-iteration results
    best_so_far = 0
    for i, prog in enumerate(programs):
        results["iterations"].append(i)
        results["scores"].append(prog["score"])
        best_so_far = max(best_so_far, prog["score"])
        results["best_scores"].append(best_so_far)

        # Check if solution found (score == 1.0 means 4/4 correct)
        if prog["score"] >= 1.0 and results["solution_found_at"] is None:
            results["solution_found_at"] = i

    return results


def load_iterative_results(output_dir: str) -> dict:
    """Load results from iterative agent output."""
    results = {
        "iterations": [],
        "scores": [],
        "best_scores": [],
        "solution_found_at": None,
    }

    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"Warning: No output found in {output_dir}")
        return results

    # Look for metrics files (the iterative agent saves metrics per iteration)
    metrics_files = sorted(output_path.glob("**/metrics*.json"))

    if not metrics_files:
        # Try loading from a single results file
        results_file = output_path / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
                if "iterations" in data:
                    return data

    best_so_far = 0
    for i, mf in enumerate(metrics_files):
        with open(mf) as f:
            data = json.load(f)
            score = data.get("combined_score", data.get("score", 0))
            results["iterations"].append(i)
            results["scores"].append(score)
            best_so_far = max(best_so_far, score)
            results["best_scores"].append(best_so_far)

            if score >= 1.0 and results["solution_found_at"] is None:
                results["solution_found_at"] = i

    return results


def plot_comparison(openevolve_results: dict, iterative_results: dict, output_file: str = None):
    """Generate comparison plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Score progression
    ax1 = axes[0]

    if openevolve_results["iterations"]:
        ax1.plot(
            openevolve_results["iterations"],
            openevolve_results["scores"],
            'g-s', alpha=0.5, markersize=4, label='OpenEvolve (each program)'
        )
        ax1.plot(
            openevolve_results["iterations"],
            openevolve_results["best_scores"],
            'g--', linewidth=2, label='OpenEvolve (best so far)'
        )

    if iterative_results["iterations"]:
        ax1.plot(
            iterative_results["iterations"],
            iterative_results["scores"],
            'b-o', alpha=0.5, markersize=4, label='Iterative Agent (each iteration)'
        )
        ax1.plot(
            iterative_results["iterations"],
            iterative_results["best_scores"],
            'b--', linewidth=2, label='Iterative Agent (best so far)'
        )

    ax1.axhline(y=1.0, color='r', linestyle=':', linewidth=2, label='Solution (4/4 correct)')
    ax1.set_xlabel('Program Version / Iteration', fontsize=12)
    ax1.set_ylabel('Score (fraction of correct modules)', fontsize=12)
    ax1.set_title('K-Module Problem: Convergence Comparison', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.1)

    # Plot 2: Summary statistics
    ax2 = axes[1]

    categories = ['Programs/Iterations\nto Solution', 'Final Best Score']
    openevolve_values = [
        openevolve_results["solution_found_at"] if openevolve_results["solution_found_at"] else len(openevolve_results["iterations"]),
        max(openevolve_results["best_scores"]) if openevolve_results["best_scores"] else 0
    ]
    iterative_values = [
        iterative_results["solution_found_at"] if iterative_results["solution_found_at"] else len(iterative_results["iterations"]),
        max(iterative_results["best_scores"]) if iterative_results["best_scores"] else 0
    ]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax2.bar(x - width/2, openevolve_values, width, label='OpenEvolve', color='green', alpha=0.7)
    bars2 = ax2.bar(x + width/2, iterative_values, width, label='Iterative Agent', color='blue', alpha=0.7)

    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('Summary Comparison', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=10)

    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Comparison plot saved to: {output_file}")
    else:
        plt.show()


def print_summary(openevolve_results: dict, iterative_results: dict):
    """Print summary comparison."""
    print("\n" + "=" * 60)
    print("K-MODULE PROBLEM: COMPARISON SUMMARY")
    print("=" * 60)

    print("\n### OpenEvolve (Evolutionary Search)")
    print(f"  Total programs evaluated: {len(openevolve_results['iterations'])}")
    if openevolve_results['solution_found_at'] is not None:
        print(f"  Solution found at program: #{openevolve_results['solution_found_at']}")
    else:
        print(f"  Solution NOT found")
    if openevolve_results['best_scores']:
        print(f"  Final best score: {max(openevolve_results['best_scores']):.4f}")

    print("\n### Iterative Agent (Iterative Refinement)")
    print(f"  Total iterations: {len(iterative_results['iterations'])}")
    if iterative_results['solution_found_at'] is not None:
        print(f"  Solution found at iteration: #{iterative_results['solution_found_at']}")
    else:
        print(f"  Solution NOT found")
    if iterative_results['best_scores']:
        print(f"  Final best score: {max(iterative_results['best_scores']):.4f}")

    print("\n### Analysis")
    if openevolve_results['solution_found_at'] and iterative_results['solution_found_at']:
        speedup = iterative_results['solution_found_at'] / openevolve_results['solution_found_at']
        print(f"  OpenEvolve found solution {speedup:.1f}x faster")
    elif openevolve_results['solution_found_at'] and not iterative_results['solution_found_at']:
        print(f"  OpenEvolve found solution, Iterative did not")
    elif iterative_results['solution_found_at'] and not openevolve_results['solution_found_at']:
        print(f"  Iterative found solution, OpenEvolve did not")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Compare K-Module problem results")
    parser.add_argument(
        "--openevolve-dir",
        default="openevolve_output",
        help="OpenEvolve output directory"
    )
    parser.add_argument(
        "--iterative-dir",
        default="iterative_output",
        help="Iterative agent output directory"
    )
    parser.add_argument(
        "--output",
        default="comparison_plot.png",
        help="Output plot filename"
    )
    args = parser.parse_args()

    # Load results
    print("Loading OpenEvolve results...")
    openevolve_results = load_openevolve_results(args.openevolve_dir)

    print("Loading Iterative Agent results...")
    iterative_results = load_iterative_results(args.iterative_dir)

    # Print summary
    print_summary(openevolve_results, iterative_results)

    # Generate plot
    if openevolve_results["iterations"] or iterative_results["iterations"]:
        plot_comparison(openevolve_results, iterative_results, args.output)
    else:
        print("No results to plot. Run both approaches first.")


if __name__ == "__main__":
    main()
