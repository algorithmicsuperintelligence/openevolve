"""
Evaluator for the function minimization example
"""

import importlib.util
import numpy as np
import time
import concurrent.futures
import traceback
import signal
from openevolve.evaluation_result import EvaluationResult


def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=5):
    """
    Run a function with a timeout using concurrent.futures

    Args:
        func: Function to run
        args: Arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        timeout_seconds: Timeout in seconds

    Returns:
        Result of the function or raises TimeoutError
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
            return result
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")


def safe_float(value):
    """Convert a value to float safely"""
    try:
        return float(value)
    except (TypeError, ValueError):
        print(f"Warning: Could not convert {value} of type {type(value)} to float")
        return 0.0


def evaluate(program_path):
    """
    Evaluate the program by running it multiple times and checking how close
    it gets to the known global minimum.

    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics
    """
    # Known global minimum (approximate)
    GLOBAL_MIN_X = -1.704
    GLOBAL_MIN_Y = 0.678
    GLOBAL_MIN_VALUE = -1.519

    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        # Check if the required function exists
        if not hasattr(program, "run_search"):
            print(f"Error: program does not have 'run_search' function")
            
            error_artifacts = {
                "error_type": "MissingFunction",
                "error_message": "Program is missing required 'run_search' function",
                "suggestion": "Make sure your program includes a function named 'run_search' that returns (x, y, value) or (x, y)"
            }
            
            return EvaluationResult(
                metrics={
                    "value_score": 0.0,
                    "distance_score": 0.0,
                    "reliability_score": 0.0,
                    "combined_score": 0.0,
                    "error": "Missing run_search function",
                },
                artifacts=error_artifacts
            )

        # Run multiple trials
        num_trials = 10
        x_values = []
        y_values = []
        values = []
        distances = []
        times = []
        success_count = 0

        for trial in range(num_trials):
            try:
                start_time = time.time()

                # Run with timeout
                result = run_with_timeout(program.run_search, timeout_seconds=5)

                # Handle different result formats
                if isinstance(result, tuple):
                    if len(result) == 3:
                        x, y, value = result
                    elif len(result) == 2:
                        # Assume it's (x, y) and calculate value
                        x, y = result
                        # Calculate the function value since it wasn't returned
                        value = np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20
                        print(f"Trial {trial}: Got 2 values, calculated function value: {value}")
                    else:
                        print(
                            f"Trial {trial}: Invalid result format, expected tuple of 2 or 3 values but got {len(result)}"
                        )
                        continue
                else:
                    print(
                        f"Trial {trial}: Invalid result format, expected tuple but got {type(result)}"
                    )
                    continue

                end_time = time.time()

                # Ensure all values are float
                x = safe_float(x)
                y = safe_float(y)
                value = safe_float(value)

                # Check if the result is valid (not NaN or infinite)
                if (
                    np.isnan(x)
                    or np.isnan(y)
                    or np.isnan(value)
                    or np.isinf(x)
                    or np.isinf(y)
                    or np.isinf(value)
                ):
                    print(f"Trial {trial}: Invalid result, got x={x}, y={y}, value={value}")
                    continue

                # Calculate metrics
                x_diff = x - GLOBAL_MIN_X
                y_diff = y - GLOBAL_MIN_Y
                distance_to_global = np.sqrt(x_diff**2 + y_diff**2)

                x_values.append(x)
                y_values.append(y)
                values.append(value)
                distances.append(distance_to_global)
                times.append(end_time - start_time)
                success_count += 1

            except TimeoutError as e:
                print(f"Trial {trial}: {str(e)}")
                continue
            except IndexError as e:
                # Specifically handle IndexError which often happens with early termination checks
                print(f"Trial {trial}: IndexError - {str(e)}")
                print(
                    "This is likely due to a list index check before the list is fully populated."
                )
                continue
            except Exception as e:
                print(f"Trial {trial}: Error - {str(e)}")
                print(traceback.format_exc())
                continue

        # If all trials failed, return zero scores
        if success_count == 0:
            error_artifacts = {
                "error_type": "AllTrialsFailed",
                "error_message": f"All {num_trials} trials failed - common issues: timeouts, crashes, or invalid return values",
                "suggestion": "Check for infinite loops, ensure function returns (x, y) or (x, y, value), and verify algorithm terminates within time limit"
            }
            
            return EvaluationResult(
                metrics={
                    "value_score": 0.0,
                    "distance_score": 0.0,
                    "reliability_score": 0.0,
                    "combined_score": 0.0,
                    "error": "All trials failed",
                },
                artifacts=error_artifacts
            )

        # Calculate metrics
        avg_value = float(np.mean(values))
        avg_distance = float(np.mean(distances))
        avg_time = float(np.mean(times)) if times else 1.0

        # Convert to scores (higher is better)
        value_score = float(1.0 / (1.0 + abs(avg_value - GLOBAL_MIN_VALUE)))
        distance_score = float(1.0 / (1.0 + avg_distance))
        
        # Add reliability score based on success rate
        reliability_score = float(success_count / num_trials)

        # Calculate solution quality based on distance to global minimum
        if avg_distance < 0.5:  # Very close to the correct solution
            solution_quality_multiplier = 1.5  # 50% bonus
        elif avg_distance < 1.5:  # In the right region
            solution_quality_multiplier = 1.2  # 20% bonus
        elif avg_distance < 3.0:  # Getting closer
            solution_quality_multiplier = 1.0  # No adjustment
        else:  # Not finding the right region
            solution_quality_multiplier = 0.7  # 30% penalty

        # Calculate combined score that prioritizes finding the global minimum
        # Base score from value and distance, then apply solution quality multiplier
        base_score = 0.5 * value_score + 0.3 * distance_score + 0.2 * reliability_score
        combined_score = float(base_score * solution_quality_multiplier)

        # Add artifacts for successful runs
        artifacts = {
            "convergence_info": f"Converged in {num_trials} trials with {success_count} successes",
            "best_position": f"Final position: x={x_values[-1]:.4f}, y={y_values[-1]:.4f}" if x_values else "No successful trials",
            "average_distance_to_global": f"{avg_distance:.4f}",
            "search_efficiency": f"Success rate: {reliability_score:.2%}"
        }

        return EvaluationResult(
            metrics={
                "value_score": value_score,
                "distance_score": distance_score,
                "reliability_score": reliability_score,
                "combined_score": combined_score,
            },
            artifacts=artifacts
        )
    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        print(traceback.format_exc())
        
        # Create error artifacts
        error_artifacts = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "full_traceback": traceback.format_exc(),
            "suggestion": "Check for syntax errors or missing imports in the generated code"
        }
        
        return EvaluationResult(
            metrics={
                "value_score": 0.0,
                "distance_score": 0.0,
                "reliability_score": 0.0,
                "combined_score": 0.0,
                "error": str(e),
            },
            artifacts=error_artifacts
        )


# Stage-based evaluation for cascade evaluation
def evaluate_stage1(program_path):
    """First stage evaluation with fewer trials"""
    # Known global minimum (approximate)
    GLOBAL_MIN_X = float(-1.704)
    GLOBAL_MIN_Y = float(0.678)
    GLOBAL_MIN_VALUE = float(-1.519)

    # Quick check to see if the program runs without errors
    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        # Check if the required function exists
        if not hasattr(program, "run_search"):
            print(f"Stage 1 validation: Program does not have 'run_search' function")
            
            error_artifacts = {
                "error_type": "MissingFunction",
                "error_message": "Stage 1: Program is missing required 'run_search' function",
                "suggestion": "Make sure your program includes a function named 'run_search' that returns (x, y, value) or (x, y)"
            }
            
            return EvaluationResult(
                metrics={
                    "runs_successfully": 0.0, 
                    "combined_score": 0.0,
                    "error": "Missing run_search function"
                },
                artifacts=error_artifacts
            )

        try:
            # Run a single trial with timeout
            result = run_with_timeout(program.run_search, timeout_seconds=5)

            # Handle different result formats
            if isinstance(result, tuple):
                if len(result) == 3:
                    x, y, value = result
                elif len(result) == 2:
                    # Assume it's (x, y) and calculate value
                    x, y = result
                    # Calculate the function value since it wasn't returned
                    value = np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20
                    print(f"Stage 1: Got 2 values, calculated function value: {value}")
                else:
                    print(
                        f"Stage 1: Invalid result format, expected tuple of 2 or 3 values but got {len(result)}"
                    )
                    
                    error_artifacts = {
                        "error_type": "InvalidReturnFormat",
                        "error_message": f"Stage 1: Function returned tuple with {len(result)} values, expected 2 or 3",
                        "suggestion": "run_search() must return (x, y) or (x, y, value) - check your return statement"
                    }
                    
                    return EvaluationResult(
                        metrics={
                            "runs_successfully": 0.0, 
                            "combined_score": 0.0,
                            "error": "Invalid result format"
                        },
                        artifacts=error_artifacts
                    )
            else:
                print(f"Stage 1: Invalid result format, expected tuple but got {type(result)}")
                
                error_artifacts = {
                    "error_type": "InvalidReturnType",
                    "error_message": f"Stage 1: Function returned {type(result)}, expected tuple",
                    "suggestion": "run_search() must return a tuple like (x, y) or (x, y, value), not a single value or other type"
                }
                
                return EvaluationResult(
                    metrics={
                        "runs_successfully": 0.0, 
                        "combined_score": 0.0,
                        "error": "Invalid result format"
                    },
                    artifacts=error_artifacts
                )

            # Ensure all values are float
            x = safe_float(x)
            y = safe_float(y)
            value = safe_float(value)

            # Check if the result is valid
            if (
                np.isnan(x)
                or np.isnan(y)
                or np.isnan(value)
                or np.isinf(x)
                or np.isinf(y)
                or np.isinf(value)
            ):
                print(f"Stage 1 validation: Invalid result, got x={x}, y={y}, value={value}")
                
                error_artifacts = {
                    "error_type": "InvalidResultValues",
                    "error_message": f"Stage 1: Got invalid values - x={x}, y={y}, value={value}",
                    "suggestion": "Function returned NaN or infinite values. Check for division by zero, invalid math operations, or uninitialized variables"
                }
                
                return EvaluationResult(
                    metrics={
                        "runs_successfully": 0.5, 
                        "combined_score": 0.0,
                        "error": "Invalid result values"
                    },
                    artifacts=error_artifacts
                )

            # Calculate distance safely
            x_diff = float(x) - GLOBAL_MIN_X
            y_diff = float(y) - GLOBAL_MIN_Y
            distance = float(np.sqrt(x_diff**2 + y_diff**2))

            # Calculate value-based score
            value_score = float(1.0 / (1.0 + abs(value - GLOBAL_MIN_VALUE)))
            distance_score = float(1.0 / (1.0 + distance))

            # Calculate solution quality based on distance to global minimum
            if distance < 0.5:  # Very close to the correct solution
                solution_quality_multiplier = 1.4  # 40% bonus
            elif distance < 1.5:  # In the right region
                solution_quality_multiplier = 1.15  # 15% bonus
            elif distance < 3.0:  # Getting closer
                solution_quality_multiplier = 1.0  # No adjustment
            else:  # Not finding the right region
                solution_quality_multiplier = 0.8  # 20% penalty

            # Calculate combined score for stage 1
            base_score = 0.6 * value_score + 0.4 * distance_score
            combined_score = float(base_score * solution_quality_multiplier)

            # Add artifacts for successful stage 1
            stage1_artifacts = {
                "stage1_result": f"Found solution at x={x:.4f}, y={y:.4f} with value={value:.4f}",
                "distance_to_global": f"{distance:.4f}",
                "solution_quality": f"Distance < 0.5: Very close" if distance < 0.5 else f"Distance < 1.5: Good region" if distance < 1.5 else "Could be improved"
            }

            return EvaluationResult(
                metrics={
                    "runs_successfully": 1.0,
                    "value_score": value_score,
                    "distance_score": distance_score,
                    "combined_score": combined_score,
                },
                artifacts=stage1_artifacts
            )
        except TimeoutError as e:
            print(f"Stage 1 evaluation timed out: {e}")
            
            error_artifacts = {
                "error_type": "TimeoutError",
                "error_message": "Stage 1: Function execution exceeded 5 second timeout",
                "suggestion": "Function is likely stuck in infinite loop or doing too much computation. Try reducing iterations or adding early termination conditions"
            }
            
            return EvaluationResult(
                metrics={
                    "runs_successfully": 0.0, 
                    "combined_score": 0.0,
                    "error": "Timeout"
                },
                artifacts=error_artifacts
            )
        except IndexError as e:
            # Specifically handle IndexError which often happens with early termination checks
            print(f"Stage 1 evaluation failed with IndexError: {e}")
            print("This is likely due to a list index check before the list is fully populated.")
            
            error_artifacts = {
                "error_type": "IndexError",
                "error_message": f"Stage 1: {str(e)}",
                "suggestion": "List index out of range - likely accessing empty list or wrong index. Check list initialization and bounds"
            }
            
            return EvaluationResult(
                metrics={
                    "runs_successfully": 0.0, 
                    "combined_score": 0.0,
                    "error": f"IndexError: {str(e)}"
                },
                artifacts=error_artifacts
            )
        except Exception as e:
            print(f"Stage 1 evaluation failed: {e}")
            print(traceback.format_exc())
            
            error_artifacts = {
                "error_type": type(e).__name__,
                "error_message": f"Stage 1: {str(e)}",
                "full_traceback": traceback.format_exc(),
                "suggestion": "Unexpected error occurred. Check the traceback for specific issue"
            }
            
            return EvaluationResult(
                metrics={
                    "runs_successfully": 0.0, 
                    "combined_score": 0.0,
                    "error": str(e)
                },
                artifacts=error_artifacts
            )

    except Exception as e:
        print(f"Stage 1 evaluation failed: {e}")
        print(traceback.format_exc())
        
        error_artifacts = {
            "error_type": type(e).__name__,
            "error_message": f"Stage 1 outer exception: {str(e)}",
            "full_traceback": traceback.format_exc(),
            "suggestion": "Critical error during stage 1 evaluation. Check program syntax and imports"
        }
        
        return EvaluationResult(
            metrics={
                "runs_successfully": 0.0, 
                "combined_score": 0.0,
                "error": str(e)
            },
            artifacts=error_artifacts
        )


def evaluate_stage2(program_path):
    """Second stage evaluation with more thorough testing"""
    # Full evaluation as in the main evaluate function
    return evaluate(program_path)
