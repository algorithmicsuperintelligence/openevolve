"""Evaluator for Triton softmax backward optimization.

Correctness is a hard gate. Candidates only receive a positive score after
passing randomized gradient checks against the PyTorch formula.
"""

import importlib.util
import json
import os
import statistics
import sys
import traceback
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from openevolve.evaluation_result import EvaluationResult


COMPILE_ERROR_SCORE = -1e9
CORRECTNESS_ERROR_SCORE = -1e6


@dataclass(frozen=True)
class TestCase:
    rows: int
    cols: int
    dtype_name: str
    atol: float
    rtol: float


CORRECTNESS_CASES = [
    TestCase(128, 1024, "float32", 1e-5, 1e-5),
    TestCase(256, 2048, "float32", 1e-5, 1e-5),
    TestCase(64, 4096, "float32", 1e-5, 1e-5),
    TestCase(128, 1000, "float16", 1e-3, 1e-3),
    TestCase(128, 2048, "float16", 1e-3, 1e-3),
]

BENCHMARK_CASES = [
    TestCase(512, 1024, "float16", 1e-3, 1e-3),
    TestCase(256, 2048, "float16", 1e-3, 1e-3),
    TestCase(128, 4096, "float16", 1e-3, 1e-3),
]


def _json_artifact(data: Dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def _result(metrics: Dict[str, float], artifacts: Dict[str, Any]) -> EvaluationResult:
    return EvaluationResult(
        metrics=metrics,
        artifacts={key: _json_artifact(value) for key, value in artifacts.items()},
    )


def _failure(
    score: float,
    error_type: str,
    error_message: str,
    details: Dict[str, Any] | None = None,
) -> EvaluationResult:
    artifacts = {
        "failure": {
            "error_type": error_type,
            "error_message": error_message,
            "details": details or {},
        }
    }
    return _result(
        {
            "combined_score": float(score),
            "correct": 0.0,
            "speedup": 0.0,
            "candidate_latency_ms": 0.0,
            "baseline_latency_ms": 0.0,
        },
        artifacts,
    )


def _load_module(program_path: str, prefix: str):
    module_name = f"{prefix}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_runtime():
    try:
        import torch
        import triton  # noqa: F401
    except Exception as exc:
        return None, f"Failed to import torch/triton: {exc}"

    if not torch.cuda.is_available():
        return torch, "CUDA is not available; run this evaluator on a GPU node"

    return torch, None


def _dtype(torch_module, dtype_name: str):
    if dtype_name == "float32":
        return torch_module.float32
    if dtype_name == "float16":
        return torch_module.float16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _make_inputs(torch_module, case: TestCase):
    dtype = _dtype(torch_module, case.dtype_name)
    logits = torch_module.randn((case.rows, case.cols), device="cuda", dtype=torch_module.float32)
    y = torch_module.softmax(logits, dim=-1).to(dtype)
    dy = torch_module.randn((case.rows, case.cols), device="cuda", dtype=dtype)
    return dy, y


def _torch_oracle(torch_module, dy, y):
    dy_f32 = dy.float()
    y_f32 = y.float()
    dx = y_f32 * (dy_f32 - torch_module.sum(dy_f32 * y_f32, dim=-1, keepdim=True))
    return dx.to(dy.dtype)


def _max_errors(torch_module, candidate, reference) -> Tuple[float, float]:
    diff = (candidate.float() - reference.float()).abs()
    max_abs = float(torch_module.max(diff).item())
    denom = torch_module.clamp(reference.float().abs(), min=1e-8)
    max_rel = float(torch_module.max(diff / denom).item())
    return max_abs, max_rel


def _validate_api(module) -> Callable:
    fn = getattr(module, "softmax_backward_triton", None)
    if fn is None or not callable(fn):
        raise AttributeError("candidate must define callable softmax_backward_triton(dy, y)")
    return fn


def _run_correctness(torch_module, fn: Callable, cases: List[TestCase]):
    passed = 0
    reports = []

    for case in cases:
        try:
            torch_module.manual_seed(case.rows * 100000 + case.cols)
            dy, y = _make_inputs(torch_module, case)
            expected = _torch_oracle(torch_module, dy, y)
            actual = fn(dy, y)
            torch_module.cuda.synchronize()

            if actual.shape != expected.shape:
                reports.append(
                    {
                        "correct": False,
                        "shape": [case.rows, case.cols],
                        "dtype": case.dtype_name,
                        "error_type": "ShapeMismatch",
                        "expected_shape": list(expected.shape),
                        "actual_shape": list(actual.shape),
                    }
                )
                continue

            max_abs, max_rel = _max_errors(torch_module, actual, expected)
            correct = bool(torch_module.allclose(actual, expected, atol=case.atol, rtol=case.rtol))
            if correct:
                passed += 1

            reports.append(
                {
                    "correct": correct,
                    "shape": [case.rows, case.cols],
                    "dtype": case.dtype_name,
                    "max_abs_error": max_abs,
                    "max_rel_error": max_rel,
                    "atol": case.atol,
                    "rtol": case.rtol,
                    "hint": (
                        "softmax backward must compute "
                        "y * (dy - sum(dy * y, axis=-1, keepdim=True)) with fp32 accumulation"
                    ),
                }
            )
        except Exception as exc:
            reports.append(
                {
                    "correct": False,
                    "shape": [case.rows, case.cols],
                    "dtype": case.dtype_name,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(limit=6),
                }
            )

    return {
        "passed": passed,
        "total": len(cases),
        "partial_correctness": passed / max(1, len(cases)),
        "reports": reports,
    }


def _benchmark_ms(torch_module, fn: Callable, case: TestCase, warmup: int = 10, reps: int = 50) -> float:
    dy, y = _make_inputs(torch_module, case)

    for _ in range(warmup):
        fn(dy, y)
    torch_module.cuda.synchronize()

    start = torch_module.cuda.Event(enable_timing=True)
    end = torch_module.cuda.Event(enable_timing=True)
    timings = []
    for _ in range(reps):
        start.record()
        fn(dy, y)
        end.record()
        torch_module.cuda.synchronize()
        timings.append(float(start.elapsed_time(end)))

    return float(statistics.median(timings))


def _run_benchmarks(torch_module, candidate_fn: Callable, baseline_fn: Callable):
    case_reports = []
    candidate_total = 0.0
    baseline_total = 0.0

    for case in BENCHMARK_CASES:
        candidate_ms = _benchmark_ms(torch_module, candidate_fn, case)
        baseline_ms = _benchmark_ms(torch_module, baseline_fn, case)
        speedup = baseline_ms / max(candidate_ms, 1e-9)

        candidate_total += candidate_ms
        baseline_total += baseline_ms
        case_reports.append(
            {
                "shape": [case.rows, case.cols],
                "dtype": case.dtype_name,
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


def _baseline_program_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "initial_program.py")


def evaluate(program_path: str) -> EvaluationResult:
    torch_module, runtime_error = _check_runtime()
    if runtime_error:
        return _failure(COMPILE_ERROR_SCORE, "RuntimeUnavailable", runtime_error)

    try:
        candidate_module = _load_module(program_path, "candidate")
        candidate_fn = _validate_api(candidate_module)
    except Exception as exc:
        return _failure(
            COMPILE_ERROR_SCORE,
            "ImportOrApiError",
            str(exc),
            {"traceback": traceback.format_exc(limit=8)},
        )

    try:
        baseline_module = _load_module(_baseline_program_path(), "baseline")
        baseline_fn = _validate_api(baseline_module)
    except Exception as exc:
        return _failure(
            COMPILE_ERROR_SCORE,
            "BaselineImportError",
            str(exc),
            {"traceback": traceback.format_exc(limit=8)},
        )

    correctness = _run_correctness(torch_module, candidate_fn, CORRECTNESS_CASES)
    if correctness["passed"] != correctness["total"]:
        score = CORRECTNESS_ERROR_SCORE + correctness["partial_correctness"]
        return _result(
            {
                "combined_score": float(score),
                "correct": 0.0,
                "partial_correctness": float(correctness["partial_correctness"]),
                "speedup": 0.0,
                "candidate_latency_ms": 0.0,
                "baseline_latency_ms": 0.0,
            },
            {"correctness": correctness},
        )

    try:
        benchmark = _run_benchmarks(torch_module, candidate_fn, baseline_fn)
    except Exception as exc:
        return _failure(
            COMPILE_ERROR_SCORE,
            "BenchmarkError",
            str(exc),
            {"traceback": traceback.format_exc(limit=8), "correctness": correctness},
        )

    return _result(
        {
            "combined_score": float(benchmark["speedup"]),
            "correct": 1.0,
            "partial_correctness": 1.0,
            "speedup": float(benchmark["speedup"]),
            "candidate_latency_ms": float(benchmark["candidate_latency_ms"]),
            "baseline_latency_ms": float(benchmark["baseline_latency_ms"]),
        },
        {"correctness": correctness, "benchmark": benchmark},
    )


def evaluate_stage1(program_path: str) -> EvaluationResult:
    """Fast correctness-only cascade stage."""
    torch_module, runtime_error = _check_runtime()
    if runtime_error:
        return _failure(COMPILE_ERROR_SCORE, "RuntimeUnavailable", runtime_error)

    try:
        candidate_module = _load_module(program_path, "candidate_stage1")
        candidate_fn = _validate_api(candidate_module)
    except Exception as exc:
        return _failure(
            COMPILE_ERROR_SCORE,
            "ImportOrApiError",
            str(exc),
            {"traceback": traceback.format_exc(limit=8)},
        )

    quick_cases = CORRECTNESS_CASES[:2]
    correctness = _run_correctness(torch_module, candidate_fn, quick_cases)
    if correctness["passed"] != correctness["total"]:
        score = CORRECTNESS_ERROR_SCORE + correctness["partial_correctness"]
        return _result(
            {
                "combined_score": float(score),
                "correct": 0.0,
                "partial_correctness": float(correctness["partial_correctness"]),
            },
            {"correctness": correctness},
        )

    return _result(
        {
            "combined_score": 1.0,
            "correct": 1.0,
            "partial_correctness": 1.0,
        },
        {"correctness": correctness},
    )


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print(f"Usage: {argv[0]} PROGRAM_PATH")
        return 2

    result = evaluate(argv[1])
    print(json.dumps({"metrics": result.metrics, "artifacts": result.artifacts}, indent=2))
    return 0 if result.metrics.get("correct", 0.0) == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
