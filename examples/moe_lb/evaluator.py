"""
Evaluator for MoE Load Balancing case.
Adopted from LoongFlow's original implementation:
https://github.com/baidu-baige/LoongFlow/tree/main/agents/math_agent/examples/moe_lb
"""

import inspect
import json
import os
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass, fields, is_dataclass
from functools import wraps
from typing import Any

import numpy as np

TARGET_IMBALANCE_RATIO = 1.01
TIMEOUT_SECONDS = 60
N_EVAL_EPOCHS = 100  # Runs 100 epochs for evaluation
N_EVAL_RUNS = 5  # Repeat the entire process 5 times to get an average speedup


class TimeoutError(Exception):
    pass


@dataclass
class SingleRunResult:
    score: float
    imbalance_ratio: float
    verified_max_load: float


@dataclass
class FailureArtifacts:
    failure_stage: str
    error_message: str


@dataclass
class SuccessArtifacts:
    test_case_name: str


@dataclass
class Metrics:
    mean: float
    std: float
    min: float | None = None
    max: float | None = None


@dataclass
class EvalMetrics:
    score: Metrics | None = None
    imbalance_ratio: Metrics | None = None
    runtime: Metrics | None = None
    completed_runs: int = 0
    completed_epochs: int = 0


@dataclass
class EvalResult:
    validity: int
    combined_score: tuple[float, float]  # ? (mean_score, speedup)
    summary: str
    metrics: EvalMetrics
    artifacts: dict[str, Any]


def to_dict_safe(obj):
    if is_dataclass(obj):
        return {f.name: to_dict_safe(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_dict_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_dict_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_dict_safe(v) for v in obj)
    return obj


def return_asdict(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        if is_dataclass(res):
            return to_dict_safe(res)
        return res

    return wrapper


def redundant_policy(
    n_logical_experts: int,
    ep_size: int,
    n_redundants_per_rank: int,
    workload_history: np.ndarray,
) -> np.ndarray:
    """
    Determines expert placement across GPUs.
    Each GPU hosts its primary experts plus replicas of high-load experts from others.
    """
    n_original_per_rank = n_logical_experts // ep_size
    n_slots_per_rank = n_original_per_rank + n_redundants_per_rank
    placement = np.zeros((ep_size, n_slots_per_rank), dtype=int)

    # 1. Map primary experts to their native ranks
    primary_experts = np.arange(n_logical_experts).reshape(ep_size, n_original_per_rank)
    placement[:, :n_original_per_rank] = primary_experts

    # 2. Select top-N "hot" experts per rank using partition (faster than full sort)
    rank_workloads = workload_history[primary_experts]
    # np.argpartition puts the largest N elements at the end, then we sort only those
    top_n_idx = np.argpartition(rank_workloads, -n_redundants_per_rank, axis=1)[
        :, -n_redundants_per_rank:
    ]
    # Refine: sort the top-N slice to ensure deterministic replica ordering
    top_w = np.take_along_axis(rank_workloads, top_n_idx, axis=1)
    refined_order = np.argsort(-top_w, axis=1)
    top_indices = np.take_along_axis(top_n_idx, refined_order, axis=1)

    top_expert_ids = np.take_along_axis(primary_experts, top_indices, axis=1)

    # 3. Spread replicas to other ranks using a ring-shift pattern
    for i in range(n_redundants_per_rank):
        target_ranks = (np.arange(ep_size) + i + 1) % ep_size
        placement[target_ranks, n_original_per_rank + i] = top_expert_ids[:, i]

    return placement


def _sandbox_exec():
    import importlib.util
    import pickle
    import sys
    import traceback

    try:
        args = pickle.load(sys.stdin.buffer)
        program_path = args["program_path"]
        inputs = args["inputs"]

        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        alloc, max_load = program.solve_lplb_policy(
            inputs["initial_workloads"], inputs["physical_expert_placement"]
        )

        result = {"validity": 1, "allocation_matrix": alloc, "minimized_max_load": max_load}
        sys.stdout.buffer.write(pickle.dumps(result))

    except Exception as e:
        error_result = {"validity": 0, "error": str(e), "traceback": traceback.format_exc()}
        sys.stdout.buffer.write(pickle.dumps(error_result))

    sys.stdout.buffer.flush()


def run_in_sandbox(program_path: os.PathLike, input_data: dict) -> dict:
    executor_source = inspect.getsource(_sandbox_exec)

    full_script = f"{executor_source}\nif __name__ == '__main__': _sandbox_exec()"

    process = subprocess.Popen(
        [sys.executable, "-c", full_script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        payload = {"program_path": str(program_path), "inputs": input_data}
        stdout_data, stderr_data = process.communicate(
            input=pickle.dumps(payload), timeout=TIMEOUT_SECONDS
        )

        if process.returncode != 0:
            raise RuntimeError(
                f"Sandbox process crashed with exit code {process.returncode}. Stderr: {stderr_data.decode()}"
            )
        if not stdout_data:
            raise RuntimeError("Sandbox produced no output. It might have been killed.")

        result = pickle.loads(stdout_data)

        if result["validity"] == 0:
            raise RuntimeError(f"Sandbox Error: {result['error']}\n{result['traceback']}")

        return result

    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise TimeoutError(f"User program timed out after {TIMEOUT_SECONDS}s")
    except Exception as e:
        if process.poll() is None:
            process.kill()
        raise e


def run_single_evaluation(program_path: str) -> SingleRunResult:
    # --- 1. Scenario Synthesis ---
    n_logical_experts, ep_size, n_redundants_per_rank = 256, 16, 4
    workload_history = np.random.rand(n_logical_experts) * (2**20)
    placement = redundant_policy(
        n_logical_experts, ep_size, n_redundants_per_rank, workload_history
    )
    initial_workloads = np.random.rand(n_logical_experts) * (2**12)

    # --- 2. Sandbox Execution ---
    results: dict[str, Any] = run_in_sandbox(
        program_path,
        {
            "initial_workloads": initial_workloads,
            "physical_expert_placement": placement,
        },
    )

    alloc_matrix: np.ndarray = results["allocation_matrix"]

    # --- 3. Verification & Metrics ---
    # Constraint A: Flow conservation (Workload must be fully processed)
    if not np.allclose(alloc_matrix.sum(axis=1), initial_workloads, atol=1e-7):
        raise ValueError("Conservation failed: Load assigned does not match workload.")

    # Constraint B: Non-negativity
    if (alloc_matrix < -1e-9).any():
        raise ValueError("Non-negativity failed: Found negative allocations.")

    # Constraint C: Placement (Expert replica must exist on the GPU slot)
    # Optimized: Use sparse-style check to avoid massive O(N*M) mask creation
    active_rows, active_cols = np.nonzero(alloc_matrix)
    flat_placement = placement.flatten()
    if not np.all(active_rows == flat_placement[active_cols]):
        raise ValueError("Placement failed: Expert assigned to a GPU where it isn't stored.")

    # --- 4. Performance Scoring ---
    slot_loads = alloc_matrix.sum(axis=0)
    gpu_loads = slot_loads.reshape(ep_size, -1).sum(axis=1)
    verified_max_load = np.max(gpu_loads)

    avg_load = initial_workloads.sum() / ep_size
    imbalance_ratio = verified_max_load / avg_load if avg_load > 0 else 0.0

    # Score capped at 1.0, decreases as imbalance exceeds target
    score = min(1.0, TARGET_IMBALANCE_RATIO / imbalance_ratio) if imbalance_ratio > 0 else 0.0

    return SingleRunResult(
        score=score, imbalance_ratio=imbalance_ratio, verified_max_load=verified_max_load
    )


def summarize(data_list: list[float], include_minmax: bool = True) -> Metrics:
    return Metrics(
        mean=float(np.mean(data_list)),
        std=float(np.std(data_list)),
        min=float(np.min(data_list)) if include_minmax else None,
        max=float(np.max(data_list)) if include_minmax else None,
    )


@return_asdict
def evaluate(path_user_py: str) -> dict[str, Any]:
    score_list, imbalance_ratio_list, time_list = [], [], []

    for i in range(N_EVAL_RUNS):
        tstart = time.time()
        for j in range(N_EVAL_EPOCHS):
            try:
                result = run_single_evaluation(path_user_py)

                score_list.append(result.score)
                imbalance_ratio_list.append(result.imbalance_ratio)

            except (TimeoutError, RuntimeError, TypeError, ValueError) as e:
                eval_time = time.time() - tstart
                return EvalResult(
                    validity=0,
                    combined_score=(0.0, 0.0),
                    summary=f"[Run {i + 1}] Evaluation failed on epoch {j + 1}/{N_EVAL_EPOCHS}: {str(e)}",
                    metrics=EvalMetrics(
                        runtime=Metrics(mean=eval_time, std=0.0, min=eval_time, max=eval_time),
                        completed_runs=i,
                    ),
                    artifacts=FailureArtifacts(
                        failure_stage="single_run_execution",
                        error_message=str(e),
                    ),
                )
        tend = time.time()
        time_list.append(tend - tstart)

    score = summarize(score_list)
    imbalance_ratio = summarize(imbalance_ratio_list, False)
    runtime = summarize(time_list)

    speedup = round(TIMEOUT_SECONDS / runtime.mean, 3)

    return EvalResult(
        validity=1,
        combined_score=(round(score.mean, 3), round(speedup, 3)),
        summary=(
            f"Evaluation successful across {N_EVAL_RUNS} * {N_EVAL_EPOCHS} executions in average {runtime.mean:.2f} seconds. "
            f"Mean score: {score.mean:.4f}, "
            f"Mean imbalance: {imbalance_ratio.mean:.4f}"
        ),
        metrics=EvalMetrics(
            score=score,
            imbalance_ratio=imbalance_ratio,
            runtime=runtime,
            completed_runs=N_EVAL_RUNS,
            completed_epochs=N_EVAL_RUNS * N_EVAL_EPOCHS,
        ),
        artifacts=SuccessArtifacts(test_case_name=f"Dynamic_Avg_{N_EVAL_RUNS}_Runs"),
    )


if __name__ == "__main__":
    program_file = "initial_program.py"
    # program_file = "archives/loongflow_best_lp.py"
    # program_file = "archives/loongflow_best_common.py"
    if not os.path.exists(program_file):
        print(f"Error: File not found at {program_file}")
    else:
        print(
            f"--- Evaluating {program_file} ({N_EVAL_RUNS} runs, {N_EVAL_EPOCHS} epochs per run) ---"
        )
        report = evaluate(program_file)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        combined_score: tuple[float, float] = report.get("combined_score", (0.0, 0.0))
        print(f"\nFinal Mean Score: {combined_score[0]:.3f}")
        print(f"Speedup Factor: {combined_score[1]:.3f}x")
