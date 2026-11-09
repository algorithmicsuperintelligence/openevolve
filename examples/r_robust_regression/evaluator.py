"""
Evaluator for R robust regression example
"""

import asyncio
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np

from openevolve.evaluation_result import EvaluationResult


def _r_string(path: str) -> str:
    """Quote a path for an R double-quoted string.

    Backslashes are turned into forward slashes, which R accepts on Windows,
    so a path such as C:\\Users\\... is not treated as an escape sequence.
    """
    text = os.fspath(path).replace("\\", "/")
    return '"' + text.replace('"', '\\"') + '"'


def _r_runner_script(program_path: str, x_path: str, y_path: str) -> str:
    """Build the R script that loads the candidate and the CSV test data."""
    return f"""
# Source the program
source({_r_string(program_path)})

# Load test data
X <- as.matrix(read.csv({_r_string(x_path)}, header=FALSE))
y <- as.vector(as.matrix(read.csv({_r_string(y_path)}, header=FALSE)))

# Time the execution
start_time <- Sys.time()
metrics <- main()
end_time <- Sys.time()
exec_time <- as.numeric(end_time - start_time, units="secs")

# Add execution time
metrics$execution_time <- exec_time

# Save results
write(jsonlite::toJSON(metrics, auto_unbox=TRUE), "results.json")
"""


async def evaluate(program_path: str) -> EvaluationResult:
    """
    Evaluate an R program implementing robust regression.

    Tests the program on synthetic data with outliers to measure:
    - Accuracy (MSE, MAE, R-squared)
    - Robustness to outliers
    - Computational efficiency
    """
    try:
        # Generate test datasets with different outlier levels
        test_cases = [
            generate_regression_data(n_samples=100, n_features=3, outlier_fraction=0.0, noise=0.1),
            generate_regression_data(n_samples=100, n_features=3, outlier_fraction=0.1, noise=0.1),
            generate_regression_data(n_samples=100, n_features=3, outlier_fraction=0.2, noise=0.1),
            generate_regression_data(n_samples=200, n_features=5, outlier_fraction=0.15, noise=0.2),
        ]

        total_score = 0
        total_mse = 0
        total_mae = 0
        total_medae = 0
        total_r_squared = 0
        total_outlier_robustness = 0
        total_time = 0

        artifacts = {"test_results": []}

        for i, (X, y, true_coeffs) in enumerate(test_cases):
            # Save test data first so the R script can read the CSV paths.
            X_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
            y_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
            np.savetxt(X_file.name, X, delimiter=",", fmt="%.6f")
            np.savetxt(y_file.name, y, delimiter=",", fmt="%.6f")
            X_file.close()
            y_file.close()

            # Create a temporary R script that sources the program and runs it
            with tempfile.NamedTemporaryFile(mode="w", suffix=".r", delete=False) as f:
                f.write(_r_runner_script(program_path, X_file.name, y_file.name))
                test_script = f.name

            # Run the R script
            try:
                result = subprocess.run(
                    ["Rscript", test_script],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=os.path.dirname(test_script),
                )

                if result.returncode != 0:
                    artifacts["test_results"].append(
                        {"test_case": i, "error": "R execution failed", "stderr": result.stderr}
                    )
                    continue

                # Read results
                results_path = os.path.join(os.path.dirname(test_script), "results.json")
                if not os.path.exists(results_path):
                    artifacts["test_results"].append(
                        {"test_case": i, "error": "No results file produced"}
                    )
                    continue

                with open(results_path, "r") as f:
                    metrics = json.load(f)

                # Calculate case score (emphasize robustness for cases with outliers)
                outlier_fraction = [0.0, 0.1, 0.2, 0.15][i]
                if outlier_fraction > 0:
                    # For cases with outliers, prioritize robust metrics
                    case_score = (
                        0.2 * (1 - min(metrics.get("mse", 1), 1))
                        + 0.3 * (1 - min(metrics.get("medae", 1), 1))
                        + 0.4 * metrics.get("outlier_robustness", 0)
                        + 0.1 * max(0, metrics.get("r_squared", 0))
                    )
                else:
                    # For clean data, prioritize accuracy
                    case_score = (
                        0.4 * (1 - min(metrics.get("mse", 1), 1))
                        + 0.3 * (1 - min(metrics.get("mae", 1), 1))
                        + 0.2 * max(0, metrics.get("r_squared", 0))
                        + 0.1 * metrics.get("outlier_robustness", 0)
                    )

                total_score += case_score
                total_mse += metrics.get("mse", 1)
                total_mae += metrics.get("mae", 1)
                total_medae += metrics.get("medae", 1)
                total_r_squared += max(0, metrics.get("r_squared", 0))
                total_outlier_robustness += metrics.get("outlier_robustness", 0)
                total_time += metrics.get("execution_time", 1)

                artifacts["test_results"].append(
                    {
                        "test_case": i,
                        "outlier_fraction": outlier_fraction,
                        "metrics": metrics,
                        "case_score": case_score,
                    }
                )

            except subprocess.TimeoutExpired:
                artifacts["test_results"].append({"test_case": i, "error": "Timeout"})
            except Exception as e:
                artifacts["test_results"].append({"test_case": i, "error": str(e)})
            finally:
                # Cleanup
                os.unlink(test_script)
                os.unlink(X_file.name)
                os.unlink(y_file.name)
                if os.path.exists(os.path.join(os.path.dirname(test_script), "results.json")):
                    os.unlink(os.path.join(os.path.dirname(test_script), "results.json"))

        # Calculate average metrics
        n_successful = len([r for r in artifacts["test_results"] if "error" not in r])
        if n_successful == 0:
            return EvaluationResult(
                metrics={
                    "score": 0.0,
                    "mse": float("inf"),
                    "mae": float("inf"),
                    "medae": float("inf"),
                    "r_squared": 0.0,
                    "outlier_robustness": 0.0,
                    "execution_time": float("inf"),
                },
                artifacts=artifacts,
            )

        avg_score = total_score / n_successful
        avg_mse = total_mse / n_successful
        avg_mae = total_mae / n_successful
        avg_medae = total_medae / n_successful
        avg_r_squared = total_r_squared / n_successful
        avg_outlier_robustness = total_outlier_robustness / n_successful
        avg_time = total_time / n_successful

        # Add efficiency bonus for faster execution
        efficiency_bonus = max(0, 1 - avg_time) * 0.1
        final_score = min(1.0, avg_score + efficiency_bonus)

        return EvaluationResult(
            metrics={
                "score": final_score,
                "mse": avg_mse,
                "mae": avg_mae,
                "medae": avg_medae,
                "r_squared": avg_r_squared,
                "outlier_robustness": avg_outlier_robustness,
                "execution_time": avg_time,
            },
            artifacts=artifacts,
        )

    except Exception as e:
        return EvaluationResult(
            metrics={
                "score": 0.0,
                "mse": float("inf"),
                "mae": float("inf"),
                "medae": float("inf"),
                "r_squared": 0.0,
                "outlier_robustness": 0.0,
                "execution_time": float("inf"),
            },
            artifacts={"error": str(e), "type": "evaluation_error"},
        )


def generate_regression_data(n_samples=100, n_features=3, outlier_fraction=0.1, noise=0.1):
    """Generate synthetic regression data with outliers."""
    np.random.seed(42)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # True coefficients
    true_coeffs = np.random.randn(n_features + 1)  # +1 for intercept

    # Generate target values
    y = true_coeffs[0] + X @ true_coeffs[1:] + noise * np.random.randn(n_samples)

    # Add outliers
    n_outliers = int(n_samples * outlier_fraction)
    if n_outliers > 0:
        outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
        # Make outliers by adding large errors
        y[outlier_indices] += np.random.choice([-1, 1], n_outliers) * np.random.uniform(
            3, 10, n_outliers
        )

    return X, y, true_coeffs


# For testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        result = asyncio.run(evaluate(sys.argv[1]))
        print(f"Score: {result.metrics['score']:.4f}")
        print(f"MSE: {result.metrics['mse']:.4f}")
        print(f"Outlier Robustness: {result.metrics['outlier_robustness']:.4f}")
