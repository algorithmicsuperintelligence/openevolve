import numpy as np
from typing import List, Tuple, Dict, Any
import json
import os

from openevolve.evaluation_result import EvaluationResult
import importlib.util

TASK_FILE = os.getenv("ARC_TASK_FILE", "training")
TASK_NUM = os.getenv("TASK_NUM", 0)
DATA_ROOT = os.getenv("DATA_ROOT", "/workspaces/ARC-Evolve/data/arc-prize-2025")


def pass_at_2_accuracy_single(
    attempts: List[np.ndarray],
    gt: np.ndarray
) -> Tuple[int, Dict[int, Any]]:
    """
    Compute pass@2 accuracy for a single ARC test case.

    Args:
        attempts: List of 2 numpy arrays representing model attempts.
        gt: Ground-truth output as a 2D numpy array.

    Returns:
        pass_at_2: int (1 if any attempt is perfectly correct, else 0)
        diagnostics: dict mapping attempt index -> diagnostic info.
                     If sizes match, includes indices of incorrect cells.
    """
    assert len(attempts) == 2, "Expected exactly 2 attempts for pass@2 evaluation."

    diagnostics = {}
    passed = False

    for i, pred in enumerate(attempts):
        attempt_info = {}

        # Size check
        if pred.shape != gt.shape:
            attempt_info["size_match"] = False
            attempt_info["pred_shape"] = pred.shape
            attempt_info["gt_shape"] = gt.shape
            attempt_info["incorrect_indices"] = None
            attempt_passed = False
        else:
            attempt_info["size_match"] = True

            # Find incorrect cells
            incorrect_mask = pred != gt
            incorrect_indices = np.argwhere(incorrect_mask)

            attempt_info["incorrect_indices"] = incorrect_indices.tolist()
            attempt_info["num_incorrect"] = int(incorrect_mask.sum())

            # Perfect match
            if incorrect_mask.sum() == 0:
                attempt_passed = True
            else:
                attempt_passed = False
                
        attempt_info["perfect_match"] = attempt_passed
        passed = attempt_passed or passed

        diagnostics[i] = attempt_info

    pass_at_2 = 1 if passed else 0
    
    return pass_at_2, diagnostics

def pass_at_2_accuracy_multi_test(
    all_attempts: List[List[np.ndarray]],
    all_gt: List[np.ndarray]
) -> Tuple[List[int], List[Dict[int, Any]]]:
    """
    Compute pass@2 accuracy across multiple ARC test cases.

    Args:
        all_attempts: List of lists of 2 numpy arrays for each test case.
        all_gt: List of ground-truth outputs as 2D numpy arrays.
    """
    assert len(all_attempts) == len(all_gt), "Mismatched number of test cases."
    
    all_diagnostics = []
    all_pass = []

    for attempts, gt in zip(all_attempts, all_gt):
        pass_at_2, diagnostics = pass_at_2_accuracy_single(attempts, gt)
        all_pass.append(pass_at_2)
        all_diagnostics.append(diagnostics)

    return all_pass, all_diagnostics

def extract_failure_artifacts(diagnostics):
    """
    Extract failure artifacts from diagnostics for a given example.

    Args:
        diagnostics: Diagnostics dictionary from pass_at_2_accuracy_single.
        ex_name: Name of the example (for artifact labeling).
    """
    artifacts = {}
    if not diagnostics["size_match"]:
        artifacts["error_type"] = "SizeMismatch"
        artifacts["error_message"] = f"Size mismatch found in attempt output."
        artifacts["suggestion"] = "Review your output size determination."
    else:
        artifacts["error_type"] = "IncorrectCells"
        artifacts["error_message"] = f"{diagnostics['num_incorrect']} incorrect cells found at indices {diagnostics['incorrect_indices']}."
        artifacts["suggestion"] = "Review your logic to ensure correct cell values."

    return artifacts

def evaluate(program_path):
    """
    Evaluate the program by running it multiple times and checking how close
    it gets to the known global minimum.

    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics
    """
    spec = importlib.util.spec_from_file_location("program_module", program_path)
    program_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program_module)
    
    if not hasattr(program_module, 'transform_grid_attempt_1') or not hasattr(program_module, 'transform_grid_attempt_2'):
        print(f"Stage 1 validation failed: Program must define 'transform_grid_attempt_1' and 'transform_grid_attempt_2' functions.")
        
        error_artifacts = {
                "error_type": "MissingFunction",
                "error_message": "Stage 1: Program is missing required 'transform_grid_attempt_1' and 'transform_grid_attempt_2' functions.",
                "suggestion": "Make sure your program includes a functions named 'transform_grid_attempt_1' and 'transform_grid_attempt_2' that take as an argument a 2D numpy array and return a 2D numpy array."
            }
        
        return EvaluationResult(
                metrics={
                    "runs_successfully": 0.0, 
                    "combined_score": 0.0,
                    "error": "Missing transform_grid_attempt_1 and transform_grid_attempt_2 functions"
                },
                artifacts=error_artifacts
            )
        
    # Load ARC tasks
    challenge_path = os.path.join(DATA_ROOT, f"arc-agi_{TASK_FILE}_challenges.json")

    with open(challenge_path, 'r') as f:
        tasks = json.load(f)
        
    task_id = list(tasks.keys())[int(TASK_NUM)]
    task = tasks[task_id]
    
    train_inputs = [np.array(inp["input"]) for inp in task['train']]
    train_gts = [np.array(gt["output"]) for gt in task['train']]

    train_attempts = []
    
    # Generate attempts for training data
    for inp in train_inputs:
        attempt_1 = program_module.transform_grid_attempt_1(inp)
        if not isinstance(attempt_1, np.ndarray):
            print(f"transform_grid_attempt_1 did not return a numpy array")
            
            error_artifacts = {
                "error_type": "InvalidReturnType",
                "error_message": "Stage 1: transform_grid_attempt_1 did not return a numpy array.",
                "suggestion": "Make sure your transform_grid_attempt_1 function returns a 2D numpy array."
            }
            
            return EvaluationResult(
                metrics={
                    "runs_successfully": 0.0, 
                    "combined_score": 0.0,
                    "error": "transform_grid_attempt_1 did not return a numpy array"
                },
                artifacts=error_artifacts
            )

        attempt_2 = program_module.transform_grid_attempt_2(inp)
        if not isinstance(attempt_2, np.ndarray):
            print(f"transform_grid_attempt_2 did not return a numpy array")
            
            error_artifacts = {
                "error_type": "InvalidReturnType",
                "error_message": "Stage 1: transform_grid_attempt_2 did not return a numpy array.",
                "suggestion": "Make sure your transform_grid_attempt_2 function returns a 2D numpy array."
            }
            
            return EvaluationResult(
                metrics={
                    "runs_successfully": 0.0, 
                    "combined_score": 0.0,
                    "error": "transform_grid_attempt_2 did not return a numpy array"
                },
                artifacts=error_artifacts
            )
        train_attempts.append([attempt_1, attempt_2])

    pass_at_2_train, train_diagnostics_list = pass_at_2_accuracy_multi_test(train_attempts, train_gts)
    
    metrics = {
        "runs_successfully": 1.0,
        "combined_score": sum(pass_at_2_train) / len(pass_at_2_train),
    }
    error_artifacts = {}
    for i, (train_pass, train_diagnostics) in enumerate(zip(pass_at_2_train, train_diagnostics_list)):
        example_name = f"train_example_{i}"
        metrics[f"{example_name}_pass_at_2"] = train_pass
        for attempt in train_diagnostics:
            attempt_pass = train_diagnostics[attempt]["perfect_match"]
            metrics[f"{example_name}_attempt_{attempt}"] = attempt_pass
            if not attempt_pass:
                error_artifacts[f"{example_name}_attempt_{attempt}_diagnostics"] = extract_failure_artifacts(train_diagnostics[attempt])

    return EvaluationResult(
        metrics=metrics,
        artifacts=error_artifacts
    )