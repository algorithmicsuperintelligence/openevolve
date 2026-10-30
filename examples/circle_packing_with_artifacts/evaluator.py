"""
Evaluator for circle packing example (n=26) with improved timeout handling
Enhanced with artifacts to demonstrate execution feedback
"""

import importlib.util
import numpy as np
import time
import os
import signal
import subprocess
import tempfile
import traceback
import sys
import pickle

# Import EvaluationResult for artifacts support
from openevolve.evaluation_result import EvaluationResult


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")


def validate_packing(centers, radii):
    """
    Validate finite circles with non-negative radii inside the unit square, without overlaps

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle

    Returns:
        Tuple of (is_valid: bool, validation_details: dict)
    """
    n = centers.shape[0]
    finite_radii = np.all(np.isfinite(radii))
    validation_details = {
        "total_circles": n,
        "boundary_violations": [],
        "overlaps": [],
        "invalid_values": [],
        "min_radius": float(np.min(radii)) if finite_radii else None,
        "max_radius": float(np.max(radii)) if finite_radii else None,
        "avg_radius": float(np.mean(radii)) if finite_radii else None,
    }

    # NaN can bypass geometric comparisons, so check numeric values first.
    if not np.all(np.isfinite(centers)):
        validation_details["invalid_values"].append(
            "Circle centers must contain only finite values"
        )
    if not finite_radii:
        validation_details["invalid_values"].append("Circle radii must contain only finite values")
    if np.any(radii < 0):
        validation_details["invalid_values"].append("Circle radii must be non-negative")
    if validation_details["invalid_values"]:
        validation_details["is_valid"] = False
        return False, validation_details

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-6 or x + r > 1 + 1e-6 or y - r < -1e-6 or y + r > 1 + 1e-6:
            violation = (
                f"Circle {i} at ({x:.6f}, {y:.6f}) with radius {r:.6f} is outside unit square"
            )
            validation_details["boundary_violations"].append(violation)
            print(violation)

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-6:  # Allow for tiny numerical errors
                overlap = (
                    f"Circles {i} and {j} overlap: dist={dist:.6f}, r1+r2={radii[i]+radii[j]:.6f}"
                )
                validation_details["overlaps"].append(overlap)
                print(overlap)

    is_valid = (
        len(validation_details["boundary_violations"]) == 0
        and len(validation_details["overlaps"]) == 0
    )
    validation_details["is_valid"] = is_valid

    return is_valid, validation_details


def run_with_timeout(program_path, timeout_seconds=20):
    """
    Run the program in a separate process with timeout
    using a simple subprocess approach

    Args:
        program_path: Path to the program file
        timeout_seconds: Maximum execution time in seconds

    Returns:
        centers, radii, sum_radii tuple from the program
    """
    # Create a temporary file to execute
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        # Write a script that executes the program and saves results
        script = f"""
import sys
import numpy as np
import os
import pickle
import traceback

# Add the directory to sys.path
sys.path.insert(0, os.path.dirname('{program_path}'))

# Debugging info
print(f"Running in subprocess, Python version: {{sys.version}}")
print(f"Program path: {program_path}")

try:
    # Import the program
    spec = __import__('importlib.util').util.spec_from_file_location("program", '{program_path}')
    program = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(program)
    
    # Run the packing function
    print("Calling run_packing()...")
    centers, radii, sum_radii = program.run_packing()
    print(f"run_packing() returned successfully: sum_radii = {{sum_radii}}")

    # Save results to a file
    results = {{
        'centers': centers,
        'radii': radii,
        'sum_radii': sum_radii
    }}

    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {temp_file.name}.results")
    
except Exception as e:
    # If an error occurs, save the error instead
    print(f"Error in subprocess: {{str(e)}}")
    traceback.print_exc()
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{'error': str(e)}}, f)
    print(f"Error saved to {temp_file.name}.results")
"""
        temp_file.write(script.encode())
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"

    try:
        # Run the script with timeout
        process = subprocess.Popen(
            [sys.executable, temp_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode

            # Always print output for debugging purposes
            print(f"Subprocess stdout: {stdout.decode()}")
            if stderr:
                print(f"Subprocess stderr: {stderr.decode()}")

            # Still raise an error for non-zero exit codes, but only after printing the output
            if exit_code != 0:
                raise RuntimeError(f"Process exited with code {exit_code}")

            # Load the results
            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    results = pickle.load(f)

                # Check if an error was returned
                if "error" in results:
                    raise RuntimeError(f"Program execution failed: {results['error']}")

                return results["centers"], results["radii"], results["sum_radii"]
            else:
                raise RuntimeError("Results file not found")

        except subprocess.TimeoutExpired:
            # Kill the process if it times out
            process.kill()
            process.wait()
            raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")

    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(results_path):
            os.unlink(results_path)


def evaluate(program_path):
    """
    Evaluate the program by running it once and checking the sum of radii

    Args:
        program_path: Path to the program file

    Returns:
        EvaluationResult with metrics and artifacts
    """
    # Target value from the paper
    TARGET_VALUE = 2.635  # AlphaEvolve result for n=26

    try:
        # For constructor-based approaches, a single evaluation is sufficient
        # since the result is deterministic
        start_time = time.time()

        # Use subprocess to run with timeout
        centers, radii, reported_sum = run_with_timeout(
            program_path, timeout_seconds=600  # Single timeout
        )

        end_time = time.time()
        eval_time = end_time - start_time

        # Ensure centers and radii are numpy arrays
        if not isinstance(centers, np.ndarray):
            centers = np.array(centers)
        if not isinstance(radii, np.ndarray):
            radii = np.array(radii)

        # Validate solution
        valid, validation_details = validate_packing(centers, radii)

        # Check shape and size
        shape_valid = centers.shape == (26, 2) and radii.shape == (26,)
        if not shape_valid:
            shape_error = f"Invalid shapes: centers={centers.shape}, radii={radii.shape}, expected (26, 2) and (26,)"
            print(shape_error)

            return EvaluationResult(
                metrics={
                    "sum_radii": 0.0,
                    "target_ratio": 0.0,
                    "validity": 0.0,
                    "eval_time": float(eval_time),
                    "combined_score": 0.0,
                    "radius_variance": 0.0,
                    "spatial_spread": 0.0,
                },
                artifacts={
                    "stderr": shape_error,
                    "failure_stage": "shape_validation",
                    "expected_shapes": "centers: (26, 2), radii: (26,)",
                    "actual_shapes": f"centers: {centers.shape}, radii: {radii.shape}",
                    "execution_time": f"{eval_time:.2f}s",
                },
            )

        # Calculate sum
        sum_radii = np.sum(radii) if valid else 0.0

        # Calculate feature metrics for MAP-Elites diversity
        # radius_variance: normalized variance of radii (0-1)
        # Max theoretical variance for radii in [0, 0.5] is ~0.0625
        radius_variance = float(np.var(radii) / 0.0625) if valid else 0.0
        radius_variance = min(1.0, max(0.0, radius_variance))  # Clamp to [0, 1]

        # spatial_spread: how spread out centers are (0-1)
        # Based on std of distances from centroid, normalized by max possible (0.5 * sqrt(2))
        centroid = np.mean(centers, axis=0)
        distances_from_centroid = np.sqrt(np.sum((centers - centroid) ** 2, axis=1))
        max_spread = 0.5 * np.sqrt(2)  # Max distance from center to corner
        spatial_spread = float(np.std(distances_from_centroid) / max_spread) if valid else 0.0
        spatial_spread = min(1.0, max(0.0, spatial_spread))  # Clamp to [0, 1]

        # Make sure reported_sum matches the calculated sum
        sum_mismatch = abs(sum_radii - reported_sum) > 1e-6
        if sum_mismatch:
            mismatch_warning = (
                f"Warning: Reported sum {reported_sum} doesn't match calculated sum {sum_radii}"
            )
            print(mismatch_warning)

        # Target ratio (how close we are to the target)
        target_ratio = sum_radii / TARGET_VALUE if valid else 0.0

        # Validity score
        validity = 1.0 if valid else 0.0

        # Combined score - higher is better
        combined_score = target_ratio * validity

        print(
            f"Evaluation: valid={valid}, sum_radii={sum_radii:.6f}, target={TARGET_VALUE}, ratio={target_ratio:.6f}, time={eval_time:.2f}s"
        )

        # Prepare artifacts with packing details
        artifacts = {
            "execution_time": f"{eval_time:.2f}s",
            "packing_summary": f"Sum of radii: {sum_radii:.6f}/{TARGET_VALUE} = {target_ratio:.4f}",
            "validation_report": (
                f"Valid: {valid}, Violations: "
                f"{len(validation_details.get('boundary_violations', []))} boundary, "
                f"{len(validation_details.get('overlaps', []))} overlaps, "
                f"{len(validation_details.get('invalid_values', []))} invalid values"
            ),
        }

        # Add validation details if there are issues
        if not valid:
            if validation_details.get("invalid_values"):
                artifacts["invalid_values"] = "\n".join(validation_details["invalid_values"])
            if validation_details.get("boundary_violations"):
                artifacts["boundary_violations"] = "\n".join(
                    validation_details["boundary_violations"]
                )
            if validation_details.get("overlaps"):
                artifacts["overlap_violations"] = "\n".join(validation_details["overlaps"])
            artifacts["failure_stage"] = "geometric_validation"

        # Add sum mismatch warning if present
        if sum_mismatch:
            artifacts["sum_mismatch"] = f"Reported: {reported_sum:.6f}, Calculated: {sum_radii:.6f}"

        # Add successful packing stats for good solutions
        if valid and target_ratio > 0.95:  # Near-optimal solutions
            artifacts["stdout"] = f"Excellent packing! Achieved {target_ratio:.1%} of target value"
            artifacts["radius_stats"] = (
                f"Min: {validation_details['min_radius']:.6f}, Max: {validation_details['max_radius']:.6f}, Avg: {validation_details['avg_radius']:.6f}"
            )

        return EvaluationResult(
            metrics={
                "sum_radii": float(sum_radii),
                "target_ratio": float(target_ratio),
                "validity": float(validity),
                "eval_time": float(eval_time),
                "combined_score": float(combined_score),
                "radius_variance": radius_variance,
                "spatial_spread": spatial_spread,
            },
            artifacts=artifacts,
        )

    except TimeoutError as e:
        error_msg = f"Evaluation timed out: {str(e)}"
        print(error_msg)
        return EvaluationResult(
            metrics={
                "sum_radii": 0.0,
                "target_ratio": 0.0,
                "validity": 0.0,
                "eval_time": 600.0,  # Timeout duration
                "combined_score": 0.0,
                "radius_variance": 0.0,
                "spatial_spread": 0.0,
            },
            artifacts={
                "stderr": error_msg,
                "failure_stage": "execution_timeout",
                "timeout_duration": "600s",
                "suggestion": "Consider optimizing the packing algorithm for faster convergence",
            },
        )
    except Exception as e:
        error_msg = f"Evaluation failed completely: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return EvaluationResult(
            metrics={
                "sum_radii": 0.0,
                "target_ratio": 0.0,
                "validity": 0.0,
                "eval_time": 0.0,
                "combined_score": 0.0,
                "radius_variance": 0.0,
                "spatial_spread": 0.0,
            },
            artifacts={
                "stderr": error_msg,
                "traceback": traceback.format_exc(),
                "failure_stage": "program_execution",
                "suggestion": "Check for syntax errors, import issues, or runtime exceptions",
            },
        )


# Stage-based evaluation for cascade evaluation
def evaluate_stage1(program_path):
    """
    First stage evaluation - quick validation check
    Enhanced with artifacts for debugging
    """
    try:
        # Use the simplified subprocess approach
        try:
            start_time = time.time()
            centers, radii, sum_radii = run_with_timeout(program_path, timeout_seconds=600)
            eval_time = time.time() - start_time

            # Ensure centers and radii are numpy arrays
            if not isinstance(centers, np.ndarray):
                centers = np.array(centers)
            if not isinstance(radii, np.ndarray):
                radii = np.array(radii)

            # Validate solution (shapes and constraints)
            shape_valid = centers.shape == (26, 2) and radii.shape == (26,)
            if not shape_valid:
                shape_error = f"Invalid shapes: centers={centers.shape}, radii={radii.shape}"
                print(shape_error)
                return EvaluationResult(
                    metrics={"validity": 0.0, "combined_score": 0.0, "radius_variance": 0.0, "spatial_spread": 0.0},
                    artifacts={
                        "stderr": shape_error,
                        "failure_stage": "stage1_shape_validation",
                        "expected_shapes": "centers: (26, 2), radii: (26,)",
                        "actual_shapes": f"centers: {centers.shape}, radii: {radii.shape}",
                        "execution_time": f"{eval_time:.2f}s",
                    },
                )

            valid, validation_details = validate_packing(centers, radii)

            # Calculate sum
            actual_sum = np.sum(radii) if valid else 0.0

            # Calculate feature metrics for MAP-Elites diversity
            radius_variance = float(np.var(radii) / 0.0625) if valid else 0.0
            radius_variance = min(1.0, max(0.0, radius_variance))
            centroid = np.mean(centers, axis=0)
            distances_from_centroid = np.sqrt(np.sum((centers - centroid) ** 2, axis=1))
            spatial_spread = float(np.std(distances_from_centroid) / (0.5 * np.sqrt(2))) if valid else 0.0
            spatial_spread = min(1.0, max(0.0, spatial_spread))

            # Target from paper
            target = 2.635

            # Simple combined score for stage 1
            combined_score = (actual_sum / target) if valid else 0.0

            # Prepare artifacts for stage 1
            artifacts = {
                "execution_time": f"{eval_time:.2f}s",
                "stage": "quick_validation",
                "packing_summary": f"Sum: {actual_sum:.6f}, Ratio: {actual_sum/target:.4f}",
            }

            # Add validation issues if any
            if not valid:
                artifacts["stderr"] = (
                    "Validation failed: "
                    f"{len(validation_details.get('boundary_violations', []))} boundary violations, "
                    f"{len(validation_details.get('overlaps', []))} overlaps, "
                    f"{len(validation_details.get('invalid_values', []))} invalid values"
                )
                artifacts["failure_stage"] = "stage1_geometric_validation"
                if validation_details.get("invalid_values"):
                    artifacts["invalid_values"] = "\n".join(validation_details["invalid_values"])
                if validation_details.get("boundary_violations"):
                    artifacts["boundary_issues"] = validation_details["boundary_violations"][
                        0
                    ]  # Just first issue
                if validation_details.get("overlaps"):
                    artifacts["overlap_issues"] = validation_details["overlaps"][
                        0
                    ]  # Just first issue

            # Return evaluation metrics
            return EvaluationResult(
                metrics={
                    "validity": 1.0 if valid else 0.0,
                    "sum_radii": float(actual_sum),
                    "target_ratio": float(actual_sum / target if valid else 0.0),
                    "combined_score": float(combined_score),
                    "radius_variance": radius_variance,
                    "spatial_spread": spatial_spread,
                },
                artifacts=artifacts,
            )

        except TimeoutError as e:
            error_msg = f"Stage 1 evaluation timed out: {e}"
            print(error_msg)
            return EvaluationResult(
                metrics={"validity": 0.0, "combined_score": 0.0, "radius_variance": 0.0, "spatial_spread": 0.0},
                artifacts={
                    "stderr": error_msg,
                    "failure_stage": "stage1_timeout",
                    "timeout_duration": "600s",
                    "suggestion": "Algorithm may be too slow for stage 1 - consider simpler heuristics",
                },
            )
        except Exception as e:
            error_msg = f"Stage 1 evaluation failed: {e}"
            print(error_msg)
            print(traceback.format_exc())
            return EvaluationResult(
                metrics={"validity": 0.0, "combined_score": 0.0, "radius_variance": 0.0, "spatial_spread": 0.0},
                artifacts={
                    "stderr": error_msg,
                    "traceback": traceback.format_exc(),
                    "failure_stage": "stage1_execution",
                    "suggestion": "Check basic syntax and imports before attempting full evaluation",
                },
            )

    except Exception as e:
        error_msg = f"Stage 1 evaluation failed completely: {e}"
        print(error_msg)
        print(traceback.format_exc())
        return EvaluationResult(
            metrics={"validity": 0.0, "combined_score": 0.0, "radius_variance": 0.0, "spatial_spread": 0.0},
            artifacts={
                "stderr": error_msg,
                "traceback": traceback.format_exc(),
                "failure_stage": "stage1_critical_failure",
                "suggestion": "Major issues detected - check program structure and dependencies",
            },
        )


def evaluate_stage2(program_path):
    """
    Second stage evaluation - full evaluation
    """
    # Full evaluation as in the main evaluate function
    return evaluate(program_path)
