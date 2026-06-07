"""Shared evaluator core for OpenEvolve Triton backward benchmark tasks.

Each task supplies a small ``task_spec.py`` module with operator-specific input
generation, PyTorch autograd oracle, candidate API name, and output names. This
core handles candidate loading, correctness gating, per-gradient reporting, and
latency comparison against PyTorch autograd by default.

Set ``task_spec.BASELINE_PROGRAM_PATH`` to compare speedup against a fixed Triton
program instead (legacy / ablation only).
"""

from __future__ import annotations

import importlib.util
import json
import os
import statistics
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence, Tuple

try:
    from openevolve.evaluation_result import EvaluationResult
except Exception:  # pragma: no cover - allows direct example smoke tests without installing openevolve deps
    @dataclass
    class EvaluationResult:
        metrics: Dict[str, float]
        artifacts: Dict[str, str | bytes] = field(default_factory=dict)


COMPILE_ERROR_SCORE = -1e9
CORRECTNESS_ERROR_SCORE = -1e6


def _json_artifact(data: Dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def _result(metrics: Dict[str, float], artifacts: Dict[str, Any]) -> EvaluationResult:
    return EvaluationResult(
        metrics=metrics,
        artifacts={key: _json_artifact(value) for key, value in artifacts.items()},
    )


def _output_metric_names(output_names: Sequence[str]) -> Dict[str, float]:
    return {f"{name}_correct": 0.0 for name in output_names}


def _failure(
    task_spec,
    score: float,
    error_type: str,
    error_message: str,
    details: Dict[str, Any] | None = None,
) -> EvaluationResult:
    metrics = {
        "combined_score": float(score),
        "correct": 0.0,
        "partial_correctness": 0.0,
        "speedup": 0.0,
        "candidate_latency_ms": 0.0,
        "baseline_latency_ms": 0.0,
    }
    metrics.update(_output_metric_names(task_spec.OUTPUT_NAMES))
    return _result(
        metrics,
        {
            "failure": {
                "error_type": error_type,
                "error_message": error_message,
                "details": details or {},
            }
        },
    )


def _load_module(program_path: str, prefix: str):
    module_name = f"{prefix}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_runtime():
    try:
        import torch
        import triton  # noqa: F401
    except Exception as exc:
        return None, f"Failed to import torch/triton: {exc}"

    if not torch.cuda.is_available():
        return torch, "CUDA is not available; run this evaluator on a GPU node"

    return torch, None


def validate_api(module, function_name: str) -> Callable:
    fn = getattr(module, function_name, None)
    if fn is None or not callable(fn):
        raise AttributeError(f"candidate must define callable {function_name}")
    return fn


def _normalize_outputs(output: Any, output_names: Sequence[str]) -> Tuple[Any, ...]:
    if len(output_names) == 1:
        return (output,)
    if not isinstance(output, tuple) or len(output) != len(output_names):
        expected = ", ".join(output_names)
        raise TypeError(f"candidate must return a tuple ({expected})")
    return output


def _max_errors(torch_module, candidate, reference) -> Tuple[float, float]:
    diff = (candidate.float() - reference.float()).abs()
    max_abs = float(torch_module.max(diff).item())
    denom = torch_module.clamp(reference.float().abs(), min=1e-8)
    max_rel = float(torch_module.max(diff / denom).item())
    return max_abs, max_rel


def _case_report(task_spec, case) -> Dict[str, Any]:
    if hasattr(task_spec, "case_metadata"):
        return dict(task_spec.case_metadata(case))
    return {"case": repr(case)}


def _run_correctness(torch_module, task_spec, fn: Callable, cases: Sequence[Any]):
    passed_cases = 0
    passed_by_output = {name: 0 for name in task_spec.OUTPUT_NAMES}
    reports = []

    for case in cases:
        case_data = _case_report(task_spec, case)
        try:
            if hasattr(task_spec, "seed_for_case"):
                torch_module.manual_seed(task_spec.seed_for_case(case))
            inputs = task_spec.make_inputs(torch_module, case)
            expected = _normalize_outputs(
                task_spec.torch_oracle(torch_module, *inputs),
                task_spec.OUTPUT_NAMES,
            )
            actual = _normalize_outputs(fn(*inputs), task_spec.OUTPUT_NAMES)
            torch_module.cuda.synchronize()

            shape_errors = []
            for name, actual_tensor, expected_tensor in zip(task_spec.OUTPUT_NAMES, actual, expected):
                if actual_tensor.shape != expected_tensor.shape:
                    shape_errors.append(
                        f"{name} shape {tuple(actual_tensor.shape)} != {tuple(expected_tensor.shape)}"
                    )
            if shape_errors:
                reports.append(
                    {
                        **case_data,
                        "correct": False,
                        **{f"{name}_correct": False for name in task_spec.OUTPUT_NAMES},
                        "error_type": "ShapeMismatch",
                        "error_message": "; ".join(shape_errors),
                    }
                )
                continue

            report = {**case_data, "correct": True}
            for name, actual_tensor, expected_tensor in zip(task_spec.OUTPUT_NAMES, actual, expected):
                atol = task_spec.atol(case, name)
                rtol = task_spec.rtol(case, name)
                max_abs, max_rel = _max_errors(torch_module, actual_tensor, expected_tensor)
                is_correct = bool(torch_module.allclose(actual_tensor, expected_tensor, atol=atol, rtol=rtol))
                passed_by_output[name] += int(is_correct)
                report[f"{name}_correct"] = is_correct
                report[f"{name}_max_abs_error"] = max_abs
                report[f"{name}_max_rel_error"] = max_rel
                report[f"{name}_atol"] = atol
                report[f"{name}_rtol"] = rtol
                report["correct"] = report["correct"] and is_correct

            if hasattr(task_spec, "correctness_hint"):
                report["hint"] = task_spec.correctness_hint()

            passed_cases += int(report["correct"])
            reports.append(report)
        except Exception as exc:
            reports.append(
                {
                    **case_data,
                    "correct": False,
                    **{f"{name}_correct": False for name in task_spec.OUTPUT_NAMES},
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(limit=6),
                }
            )

    total = max(1, len(cases))
    return {
        "passed": passed_cases,
        "total": len(cases),
        "partial_correctness": passed_cases / total,
        **{f"{name}_correctness": passed_by_output[name] / total for name in task_spec.OUTPUT_NAMES},
        "reports": reports,
    }


def _benchmark_ms(
    torch_module,
    task_spec,
    fn: Callable,
    case: Any,
    warmup: int = 10,
    reps: int = 50,
) -> float:
    inputs = task_spec.make_inputs(torch_module, case)

    for _ in range(warmup):
        fn(*inputs)
    torch_module.cuda.synchronize()

    start = torch_module.cuda.Event(enable_timing=True)
    end = torch_module.cuda.Event(enable_timing=True)
    timings = []
    for _ in range(reps):
        start.record()
        fn(*inputs)
        end.record()
        torch_module.cuda.synchronize()
        timings.append(float(start.elapsed_time(end)))

    return float(statistics.median(timings))


def _run_benchmarks(torch_module, task_spec, candidate_fn: Callable, baseline_fn: Callable):
    case_reports = []
    candidate_total = 0.0
    baseline_total = 0.0

    for case in task_spec.BENCHMARK_CASES:
        candidate_ms = _benchmark_ms(torch_module, task_spec, candidate_fn, case)
        baseline_ms = _benchmark_ms(torch_module, task_spec, baseline_fn, case)
        speedup = baseline_ms / max(candidate_ms, 1e-9)

        candidate_total += candidate_ms
        baseline_total += baseline_ms
        case_reports.append(
            {
                **_case_report(task_spec, case),
                "candidate_latency_ms": candidate_ms,
                "baseline_latency_ms": baseline_ms,
                "speedup": speedup,
            }
        )

    aggregate_speedup = baseline_total / max(candidate_total, 1e-9)
    return {
        "candidate_latency_ms": candidate_total,
        "baseline_latency_ms": baseline_total,
        "speedup": aggregate_speedup,
        "cases": case_reports,
    }


def _baseline_program_path(task_spec) -> str:
    if hasattr(task_spec, "BASELINE_PROGRAM_PATH"):
        return task_spec.BASELINE_PROGRAM_PATH
    return os.path.join(os.path.dirname(os.path.abspath(task_spec.__file__)), "initial_program.py")


def _resolve_benchmark_baseline(
    torch_module,
    task_spec,
) -> tuple[Callable, str]:
    """Return (baseline_fn, baseline_kind) for performance benchmarking."""
    if hasattr(task_spec, "BASELINE_PROGRAM_PATH"):
        baseline_module = _load_module(_baseline_program_path(task_spec), "baseline")
        baseline_fn = validate_api(baseline_module, task_spec.CANDIDATE_FN_NAME)
        return baseline_fn, "triton_program"

    def baseline_fn(*inputs):
        return task_spec.torch_oracle(torch_module, *inputs)

    return baseline_fn, "pytorch_autograd"


def _correctness_metrics(task_spec, score: float, correctness: Dict[str, Any]) -> Dict[str, float]:
    metrics = {
        "combined_score": float(score),
        "correct": 0.0,
        "partial_correctness": float(correctness["partial_correctness"]),
        "speedup": 0.0,
        "candidate_latency_ms": 0.0,
        "baseline_latency_ms": 0.0,
    }
    for name in task_spec.OUTPUT_NAMES:
        metrics[f"{name}_correct"] = float(correctness[f"{name}_correctness"])
    return metrics


def evaluate_program(program_path: str, task_spec) -> EvaluationResult:
    torch_module, runtime_error = check_runtime()
    if runtime_error:
        return _failure(task_spec, COMPILE_ERROR_SCORE, "RuntimeUnavailable", runtime_error)

    try:
        candidate_module = _load_module(program_path, "candidate")
        candidate_fn = validate_api(candidate_module, task_spec.CANDIDATE_FN_NAME)
    except Exception as exc:
        return _failure(
            task_spec,
            COMPILE_ERROR_SCORE,
            "ImportOrApiError",
            str(exc),
            {"traceback": traceback.format_exc(limit=8)},
        )

    try:
        baseline_fn, baseline_kind = _resolve_benchmark_baseline(torch_module, task_spec)
    except Exception as exc:
        return _failure(
            task_spec,
            COMPILE_ERROR_SCORE,
            "BaselineImportError",
            str(exc),
            {"traceback": traceback.format_exc(limit=8)},
        )

    correctness = _run_correctness(torch_module, task_spec, candidate_fn, task_spec.CORRECTNESS_CASES)
    if correctness["passed"] != correctness["total"]:
        score = CORRECTNESS_ERROR_SCORE + correctness["partial_correctness"]
        return _result(_correctness_metrics(task_spec, score, correctness), {"correctness": correctness})

    try:
        benchmark = _run_benchmarks(torch_module, task_spec, candidate_fn, baseline_fn)
    except Exception as exc:
        return _failure(
            task_spec,
            COMPILE_ERROR_SCORE,
            "BenchmarkError",
            str(exc),
            {"traceback": traceback.format_exc(limit=8), "correctness": correctness},
        )

    metrics = {
        "combined_score": float(benchmark["speedup"]),
        "correct": 1.0,
        "partial_correctness": 1.0,
        "speedup": float(benchmark["speedup"]),
        "candidate_latency_ms": float(benchmark["candidate_latency_ms"]),
        "baseline_latency_ms": float(benchmark["baseline_latency_ms"]),
    }
    metrics.update({f"{name}_correct": 1.0 for name in task_spec.OUTPUT_NAMES})
    benchmark["baseline_kind"] = baseline_kind
    return _result(metrics, {"correctness": correctness, "benchmark": benchmark})


def evaluate_stage1_program(program_path: str, task_spec) -> EvaluationResult:
    torch_module, runtime_error = check_runtime()
    if runtime_error:
        return _failure(task_spec, COMPILE_ERROR_SCORE, "RuntimeUnavailable", runtime_error)

    try:
        candidate_module = _load_module(program_path, "candidate_stage1")
        candidate_fn = validate_api(candidate_module, task_spec.CANDIDATE_FN_NAME)
    except Exception as exc:
        return _failure(
            task_spec,
            COMPILE_ERROR_SCORE,
            "ImportOrApiError",
            str(exc),
            {"traceback": traceback.format_exc(limit=8)},
        )

    quick_cases = task_spec.CORRECTNESS_CASES[:2]
    correctness = _run_correctness(torch_module, task_spec, candidate_fn, quick_cases)
    if correctness["passed"] != correctness["total"]:
        score = CORRECTNESS_ERROR_SCORE + correctness["partial_correctness"]
        return _result(_correctness_metrics(task_spec, score, correctness), {"correctness": correctness})

    metrics = {
        "combined_score": 1.0,
        "correct": 1.0,
        "partial_correctness": 1.0,
    }
    metrics.update({f"{name}_correct": 1.0 for name in task_spec.OUTPUT_NAMES})
    return _result(metrics, {"correctness": correctness})


def main(argv: List[str], task_spec) -> int:
    if len(argv) != 2:
        print(f"Usage: {argv[0]} PROGRAM_PATH")
        return 2

    result = evaluate_program(argv[1], task_spec)
    print(json.dumps({"metrics": result.metrics, "artifacts": result.artifacts}, indent=2))
    return 0 if result.metrics.get("correct", 0.0) == 1.0 else 1
