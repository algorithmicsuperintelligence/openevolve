"""
Evaluator for the TSP tour minimization example
"""

import sys
import uuid
import math
import pathlib
import traceback
import typing as tp
from datetime import datetime

BASE_DIR = pathlib.Path(__file__).parent
sys.path.append(str(BASE_DIR))

# numpy & related imports
import numpy as np

# openevolve & related imports
from openevolve.evaluation_result import EvaluationResult

# local imports
from utils.utils import *
from utils.runner import *
from utils.load_data import *
from utils.code_to_query import *


PROJECT_TEMP_DIRECTORY = BASE_DIR / "temp"  # configure it as you want
SOLUTIONS_DIRECTORY = PROJECT_TEMP_DIRECTORY / "solutions"
N = 1000
INSTANCES_COUNT = 8

HEAT_MAP_TRAIN_TIMEOUT = 480.0
HEAT_MAP_INFERENCE_TIMEOUT = 60.0
TSP_COMPILATION_TIMEOUT = 10.0
TSP_RUN_TIMEOUT = 160.0


def calc_average_elapsed_time(time_elapsed: list[float | int | None]) -> float | None:
    """
    Calculate the average of non-None times
    If more than 5% of the elements are None, return None
    """

    if not time_elapsed:
        return None

    n_total = len(time_elapsed)
    n_none = sum(t is None for t in time_elapsed)

    if n_none / n_total > 0.05:
        return None

    valid_values = [t for t in time_elapsed if t is not None]
    if not valid_values:
        return None

    return float(sum(valid_values) / len(valid_values))


def build_artifacts_from_saver(artifacts: dict, metrics: dict, output_saver: dict) -> dict:
    if "heat_map_train_data" in output_saver:
        artifacts["heat_map_train_stdout"] = output_saver["heat_map_train_data"]["stdout"]
        artifacts["heat_map_train_stderr"] = output_saver["heat_map_train_data"]["stderr"]

        artifacts["heat_map_train_time_elapsed"] = output_saver["heat_map_train_data"]["time_elapsed"]
        metrics["heat_map_train_execution_time_score"] = HEAT_MAP_TRAIN_TIMEOUT - output_saver["heat_map_train_data"]["time_elapsed"]
    
    if "heat_map_inference_data" in output_saver:
        artifacts["heat_map_inference_first_test_sample_stdout"] = output_saver["heat_map_inference_data"]["instance_stdout"]
        artifacts["heat_map_inference_first_test_sample_stderr"] = output_saver["heat_map_inference_data"]["instance_stderr"]

        average_heat_map_inference_time_elapsed = calc_average_elapsed_time(output_saver["heat_map_inference_data"]["time_elapsed"])
        artifacts["average_heat_map_inference_time_elapsed"] = average_heat_map_inference_time_elapsed
        metrics["average_heat_map_inference_execution_time_score"] = HEAT_MAP_INFERENCE_TIMEOUT - average_heat_map_inference_time_elapsed
    
    if "tsp_run_data" in output_saver:
        artifacts["tsp_run_first_test_sample_stdout"] = output_saver["tsp_run_data"]["instance_stdout"]
        artifacts["tsp_run_first_test_sample_stderr"] = output_saver["tsp_run_data"]["instance_stderr"]

        average_tsp_run_time_elapsed = calc_average_elapsed_time(output_saver["tsp_run_data"]["time_elapsed"])
        artifacts["average_tsp_run_time_elapsed"] = average_tsp_run_time_elapsed
        metrics["average_tsp_run_execution_time_score"] = TSP_RUN_TIMEOUT - average_tsp_run_time_elapsed
    
    return artifacts


def calc_combined_score(
    n: int,
    distances: np.ndarray,
    time_elapsed: tp.Sequence[float | int | None],
    h: float = 1.0,
    w: float = 1.0,
    *,
    alpha: float = 30.0,  # distance weight, larger makes distance more dominant
    g_cut: float | None = 0.15,  # hard cutoff for very bad tours, relative gap above this becomes zero score
    time_limit: float | None = 160.0,  # per-instance time budget in seconds
    time_weight: float = 0.10,  # how strongly time perturbs distance score, 0 ignores time
    time_beta: float = 1.0,  # shape parameter for time curve, >= 1 keeps it gentle
) -> float:
    """
    Distance-first combined score for TSP on [0, 1] x [0, 1] with n cities.

    For each instance i with tour length d_i and runtime t_i:

      1) BHH baseline:
           L_baseline = c_BHH * sqrt(n * h * w)

      2) Relative gap (clipped at zero):
           g_i = max(0, d_i / L_baseline - 1)

      3) Distance factor (dominant):
           D_i = exp(-alpha * g_i)      (if g_i <= g_cut)
                 0                      (otherwise)

      4) Normalized time (only if `time_limit` is not None):
           τ_i = clip(t_i, 0, time_limit) / time_limit      (in [0, 1])
           T_i = (1 - τ_i) ** time_beta                     (in [0, 1], higher is better)

         Missing or invalid times are treated as worst case (τ_i = 1, T_i = 0).

      5) Time as a small modifier (tie-breaker):
           M_i = 1 - time_weight * (1 - T_i)

         So M_i is in [1 - time_weight, 1].
         - If time_weight is 0.10 then time can change D_i by at most 10 percent.
         - Fast runs (T_i close to 1) give M_i close to 1.
         - Slow runs near the limit (T_i close to 0) give M_i close to 1 - time_weight.

      6) Per-instance score and final score:
           S_i = D_i * M_i
           Score = mean_i S_i

    Larger score is better. Mean tour length dominates and time is only a gentle tie-breaker.
    """
    bhh_reference_length = approximation_using_BHH_constant(n, h=h, w=w)

    distances_flat = np.asarray(distances, dtype=np.float64).ravel()
    num_instances = distances_flat.shape[0]

    if len(time_elapsed) != num_instances:
        raise ValueError("len(time_elapsed) must match distances.shape[0]")

    valid_distance_mask = np.isfinite(distances_flat) & (distances_flat > 0.0)
    if not np.any(valid_distance_mask):
        return 0.0

    relative_gaps = np.empty_like(distances_flat)
    relative_gaps[~valid_distance_mask] = np.inf
    relative_gaps[valid_distance_mask] = np.maximum(
        0.0,
        distances_flat[valid_distance_mask] / bhh_reference_length - 1.0,
    )

    good_gap_mask = valid_distance_mask.copy()
    if g_cut is not None:
        good_gap_mask &= (relative_gaps <= g_cut)

    distance_factor = np.zeros_like(distances_flat)
    distance_factor[good_gap_mask] = np.exp(-alpha * relative_gaps[good_gap_mask])

    if time_limit is None or time_weight <= 0.0:
        per_instance_scores = distance_factor
    else:
        time_budget = max(float(time_limit), 1e-3)

        time_fraction = np.ones(num_instances, dtype=np.float64)
        for index, elapsed in enumerate(time_elapsed):
            if elapsed is None:
                continue
            try:
                elapsed_value = float(elapsed)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(elapsed_value) or elapsed_value <= 0.0:
                continue

            effective_time = min(elapsed_value, time_budget)
            time_fraction[index] = effective_time / time_budget

        time_score = (1.0 - time_fraction) ** time_beta
        time_modifier = 1.0 - time_weight * (1.0 - time_score)

        per_instance_scores = distance_factor * time_modifier

    mean_score = float(per_instance_scores.mean())
    if not math.isfinite(mean_score):
        return 0.0

    return float(max(0.0, min(1.0, mean_score)))


def evaluate(program_path: str) -> EvaluationResult:
    """Main stage evaluation with thorough testing on test dataset"""

    # creating a new solution directory id
    solution_dir_id = datetime.now().strftime("%Y_%m_%d-%H_%M_%S") + '-' + str(uuid.uuid4())
    solution_dir_path = f"{str(SOLUTIONS_DIRECTORY)}/{solution_dir_id}"

    # parsing solutions code and generating all files
    save_parsed_output_code(parse_output_code(read_file(program_path)), solution_dir_path)

    error_metrics = {
        "heat_map_train_execution_time_score": 0.0,
        "average_heat_map_inference_execution_time_score": 0.0,
        "average_tsp_run_execution_time_score": 0.0,
        "path_length_score": 0.0,
        "path_length_variance_score": 0.0,
        "combined_score": 0.0,
    }

    output_saver = {}

    # testing pipeline
    try:
        # loading cities
        cities = load_points(str(PROJECT_TEMP_DIRECTORY), N, instances_count=INSTANCES_COUNT)

        # running
        try:
            run_data = run(
                solution_dir_path,
                cities,
                heat_map_train_timeout=HEAT_MAP_TRAIN_TIMEOUT,
                heat_map_inference_timeout=HEAT_MAP_INFERENCE_TIMEOUT,
                tsp_compilation_timeout=TSP_COMPILATION_TIMEOUT,
                tsp_run_timeout=TSP_RUN_TIMEOUT,
                verbose=False,
                output_saver=output_saver,
            )
            remove_dir(solution_dir_path)
        except Exception as e:
            error_artifacts = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "full_traceback": traceback.format_exc(),
            }
            build_artifacts_from_saver(error_artifacts, error_metrics, output_saver)            
            remove_dir(solution_dir_path)

            return EvaluationResult(
                metrics=error_metrics | {"error": str(e)},
                artifacts=error_artifacts
            )

        # checking solutions
        try:
            total_distances = calc_total_cycle_distance(cities, run_data["solutions"])
        except Exception as e:
            error_artifacts = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "full_traceback": traceback.format_exc(),
            }
            build_artifacts_from_saver(error_artifacts, error_metrics, output_saver)
            return EvaluationResult(
                metrics=error_metrics | {"error": str(e)},
                artifacts=error_artifacts
            )

        # final metrics and artifacts
        artifacts, metrics = {}, {}
        build_artifacts_from_saver(artifacts, metrics, output_saver)

        artifacts["average_path_length"] = float(total_distances.mean())
        artifacts["path_length_variance"] = float(np.var(total_distances))

        path_length_lower_bound = approximation_using_BHH_constant(cities.shape[1], h=1.0, w=1.0)

        metrics["path_length_score"] = 10 / max(artifacts["average_path_length"] - path_length_lower_bound, 0.01)
        metrics["path_length_variance_score"] = 0.5 / artifacts["path_length_variance"]
        metrics["combined_score"] = calc_combined_score(
            cities.shape[1],
            total_distances,
            output_saver["tsp_run_data"]["time_elapsed"],
            h=1.0, w=1.0,
        )

        # returning results
        return EvaluationResult(metrics=metrics, artifacts=artifacts)
    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        print(traceback.format_exc())

        # create error artifacts
        error_artifacts, error_metrics = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "full_traceback": traceback.format_exc(),
            "suggestion": "Check for syntax errors or missing imports in the generated code"
        }, {}
        build_artifacts_from_saver(error_artifacts, error_metrics, output_saver)

        return EvaluationResult(
            metrics=error_metrics | {"error": str(e)},
            artifacts=error_artifacts
        )
