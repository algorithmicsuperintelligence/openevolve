import importlib.util
import os
import json
import numpy as np
from evaluator import pass_at_2_accuracy_multi_test, extract_failure_artifacts

TASK_FILE = os.getenv("ARC_TASK_FILE", "training")
TASK_NUM = os.getenv("TASK_NUM", 0)
OUTS_DIR = os.getenv("OUTS_DIR", "")


def load_program_module():
    """Dynamically load the best_program.py module from the specified directory."""
    spec = importlib.util.spec_from_file_location("program_module", os.path.join(OUTS_DIR, "best/best_program.py"))
    program_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program_module)
    
    return program_module

def evaluate():
    """Evaluate the program module located in the specified directory."""
    program_module = load_program_module()
    if not hasattr(program_module, 'transform_grid_attempt_1') or not hasattr(program_module, 'transform_grid_attempt_2'):
        print(f"Stage 1 validation failed: Program must define 'transform_grid_attempt_1' and 'transform_grid_attempt_2' functions.")
        
        error_artifacts = {
                "error_type": "MissingFunction",
                "error_message": "Stage 1: Program is missing required 'transform_grid_attempt_1' and 'transform_grid_attempt_2' functions.",
                "suggestion": "Make sure your program includes a functions named 'transform_grid_attempt_1' and 'transform_grid_attempt_2' that take as an argument a 2D numpy array and return a 2D numpy array."
            }
        
        return dict(
                metrics={
                    "runs_successfully": 0.0, 
                    "combined_score": 0.0,
                    "error": "Missing transform_grid_attempt_1 and transform_grid_attempt_2 functions"
                },
                artifacts=error_artifacts
            )
    # Load ARC tasks
    challenge_path = f"/workspaces/ARC-Evolve/data/arc-prize-2025/arc-agi_{TASK_FILE}_challenges.json"
    solution_path = f"/workspaces/ARC-Evolve/data/arc-prize-2025/arc-agi_{TASK_FILE}_solutions.json"

    with open(challenge_path, 'r') as f:
        tasks = json.load(f)
    with open(solution_path, 'r') as f:
        solutions = json.load(f)
        
    task_id = list(tasks.keys())[int(TASK_NUM)]
    solution = solutions[task_id]
    task = tasks[task_id]
    
    test_inputs = [np.array(inp["input"]) for inp in task['test']]
    test_gts = [np.array(gt) for gt in solution]
    
    test_attempts = []
    for inp in test_inputs:
        attempt_1 = program_module.transform_grid_attempt_1(inp)
        if not isinstance(attempt_1, np.ndarray):
            print(f"transform_grid_attempt_1 did not return a numpy array")
            
            error_artifacts = {
                "error_type": "InvalidReturnType",
                "error_message": "Stage 1: transform_grid_attempt_1 did not return a numpy array.",
                "suggestion": "Make sure your transform_grid_attempt_1 function returns a 2D numpy array."
            }
            
            return dict(
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
            
            return dict(
                metrics={
                    "runs_successfully": 0.0, 
                    "combined_score": 0.0,
                    "error": "transform_grid_attempt_2 did not return a numpy array"
                },
                artifacts=error_artifacts
            )
        test_attempts.append([attempt_1, attempt_2])
        
    pass_at_2_test, test_diagnostics_list = pass_at_2_accuracy_multi_test(test_attempts, test_gts)
    metrics = {
        "runs_successfully": 1.0,
        "combined_score": sum(pass_at_2_test) / len(pass_at_2_test),
    }
    error_artifacts = {}
    for i, (test_pass, test_diagnostics) in enumerate(zip(pass_at_2_test, test_diagnostics_list)):
        example_name = f"test_example_{i}"
        metrics[f"{example_name}_pass_at_2"] = test_pass
        for attempt in test_diagnostics:
            metrics[f"{example_name}_attempt_{attempt}"] = test_diagnostics[attempt]["perfect_match"]
        if test_pass == 0:
            error_artifacts = extract_failure_artifacts(test_diagnostics)
    
    return dict(
        metrics=metrics,
        artifacts=error_artifacts
    )
    
if __name__ == "__main__":
    evaluation_result = evaluate()
    with open(os.path.join(OUTS_DIR, "best", "post_evolution_evaluation_result.json"), 'w') as f:
        json.dump(evaluation_result, f, indent=4)