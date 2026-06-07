"""Generic correctness runner for backward kernels.

Example:

    python -m tests.atenir_correctness.run_correctness \
      --forward benchmark.triton_layernorm_backward_bench.forward_ref:layernorm_forward_ref \
      --backward benchmark.triton_layernorm_backward_bench.backward_naive_triton:layernorm_backward_naive_triton \
      --shape 17,127 --shape 127 --shape 127 \
      --scalar 1e-5 \
      --dtype float32 \
      --atol 2e-5 --rtol 2e-5

Or use a built-in suite:

    python -m tests.atenir_correctness.run_correctness \
      --forward benchmark.triton_layernorm_backward_bench.forward_ref:layernorm_forward_ref \
      --backward benchmark.triton_layernorm_backward_bench.backward_atenir:layernorm_backward_triton \
      --mode dynamic \
      --scalar 1e-5

The runner treats the forward function as the semantic reference, computes
PyTorch-autograd gradients, calls the backward function as ``backward(dy,
*inputs, *scalars)``, and compares every returned gradient. If shapes are not
provided explicitly, it infers a shape preset from the forward/backward/task
name and applies the selected ``--mode``.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any

import torch


DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


SHAPE_PRESETS = {
    "square_sum": {
        "static": [[(16, 17)]],
        "dynamic": [[(16, 17)], [(7, 31)], [(32, 64)]],
        "nontile": [[(7, 31)]],
    },
    "rmsnorm": {
        "static": [[(16, 64), (64,)]],
        "dynamic": [[(1, 768), (768,)], [(8, 1024), (1024,)], [(32, 1536), (1536,)]],
        "nontile": [[(17, 127), (127,)], [(17, 513), (513,)]],
    },
    "attention_block": {
        "static": [[(2, 8, 16), (2, 8, 16), (2, 8, 16)]],
        "dynamic": [[(1, 4, 16), (1, 4, 16), (1, 4, 16)], [(2, 8, 32), (2, 8, 32), (2, 8, 32)]],
        "nontile": [[(1, 7, 15), (1, 7, 15), (1, 7, 15)]],
    },
    "topk_gather": {
        "static": [[(16, 32)]],
        "dynamic": [[(8, 32)], [(16, 64)], [(32, 128)]],
        "nontile": [[(7, 31)]],
    },
    "layernorm": {
        "static": [[(8, 64), (64,), (64,)]],
        "dynamic": [
            [(1, 768), (768,), (768,)],
            [(8, 1024), (1024,), (1024,)],
            [(32, 1536), (1536,), (1536,)],
            [(8, 4096), (4096,), (4096,)],
            [(1, 8192), (8192,), (8192,)],
        ],
        "nontile": [
            [(17, 127), (127,), (127,)],
            [(17, 513), (513,), (513,)],
            [(17, 1000), (1000,), (1000,)],
        ],
    },
}


def _load_callable(spec: str):
    module_name, fn_name = spec.split(":", 1) if ":" in spec else spec.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), fn_name)


def _load_callable_from_file(path: str, fn_name: str):
    module_name = f"correctness_candidate_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, Path(path).resolve())
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Python file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, fn_name)


def _load_module(spec: str):
    return importlib.import_module(spec)


def _parse_shape(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.replace("x", ",").split(",") if part)


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if lowered in DTYPES:
        return DTYPES[lowered]
    return value


def _make_inputs(shapes: list[tuple[int, ...]], dtype: torch.dtype, seed: int):
    torch.manual_seed(seed)
    return [torch.randn(shape, device="cuda", dtype=dtype) for shape in shapes]


def _normalize_outputs(output):
    if isinstance(output, tuple):
        return output
    if isinstance(output, list):
        return tuple(output)
    return (output,)


def _max_errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    max_abs = float(torch.max(diff).item())
    denom = torch.clamp(expected.float().abs(), min=1e-8)
    max_rel = float(torch.max(diff / denom).item())
    return max_abs, max_rel


def _compare_one_case(
    *,
    forward,
    backward,
    inputs: list[torch.Tensor],
    scalars: list[Any],
    atol: float,
    rtol: float,
    case_name: str,
) -> dict[str, Any]:
    ref_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in inputs]
    y = forward(*ref_inputs, *scalars)
    if isinstance(y, (tuple, list)):
        y = y[0]
    dy = torch.randn_like(y)
    expected = torch.autograd.grad(y, ref_inputs, grad_outputs=dy)

    actual = _normalize_outputs(backward(dy, *[tensor.detach() for tensor in inputs], *scalars))
    if len(actual) != len(expected):
        return {
            "passed": False,
            "case": case_name,
            "error": f"output count mismatch: got {len(actual)}, expected {len(expected)}",
        }

    reports = []
    passed = True
    for index, (actual_tensor, expected_tensor) in enumerate(zip(actual, expected)):
        if actual_tensor.shape != expected_tensor.shape:
            reports.append(
                {
                    "index": index,
                    "passed": False,
                    "error": "shape mismatch",
                    "actual_shape": list(actual_tensor.shape),
                    "expected_shape": list(expected_tensor.shape),
                }
            )
            passed = False
            continue
        max_abs, max_rel = _max_errors(actual_tensor, expected_tensor)
        ok = bool(torch.allclose(actual_tensor, expected_tensor, atol=atol, rtol=rtol))
        reports.append(
            {
                "case": case_name,
                "index": index,
                "passed": ok,
                "shape": list(actual_tensor.shape),
                "max_abs": max_abs,
                "max_rel": max_rel,
                "atol": atol,
                "rtol": rtol,
            }
        )
        passed = passed and ok

    return {"passed": passed, "case": case_name, "reports": reports}


def _compare_atenir_compose_case(
    *,
    forward_spec: str,
    inputs: list[torch.Tensor],
    atol: float,
    rtol: float,
    case_name: str,
) -> dict[str, Any]:
    from tests.atenir_correctness.harness import AtenIRComposeBackend, _load_callable, autograd_oracle

    forward = _load_callable(forward_spec)
    grad_out, expected = autograd_oracle(forward, inputs)
    backend = AtenIRComposeBackend(f"atenir_compose_{forward_spec}", forward_spec)
    actual = backend.run(grad_out, inputs)
    if len(actual) != len(expected):
        return {
            "passed": False,
            "case": case_name,
            "error": f"output count mismatch: got {len(actual)}, expected {len(expected)}",
            "fallback_report": backend.fallback_report,
        }

    reports = []
    passed = True
    for index, (actual_tensor, expected_tensor) in enumerate(zip(actual, expected)):
        if actual_tensor.shape != expected_tensor.shape:
            reports.append(
                {
                    "case": case_name,
                    "index": index,
                    "passed": False,
                    "error": "shape mismatch",
                    "actual_shape": list(actual_tensor.shape),
                    "expected_shape": list(expected_tensor.shape),
                }
            )
            passed = False
            continue
        max_abs, max_rel = _max_errors(actual_tensor, expected_tensor)
        ok = bool(torch.allclose(actual_tensor, expected_tensor, atol=atol, rtol=rtol))
        reports.append(
            {
                "case": case_name,
                "index": index,
                "passed": ok,
                "shape": list(actual_tensor.shape),
                "max_abs": max_abs,
                "max_rel": max_rel,
                "atol": atol,
                "rtol": rtol,
            }
        )
        passed = passed and ok
    return {
        "passed": passed,
        "case": case_name,
        "reports": reports,
        "fallback_report": backend.fallback_report,
    }


def _infer_op_from_text(value: str | None) -> str | None:
    if not value:
        return None
    lowered = value.lower()
    for op_name in SHAPE_PRESETS:
        if op_name in lowered:
            return op_name
    return None


def _shape_sets_from_args(args: argparse.Namespace) -> list[list[tuple[int, ...]]]:
    if args.shape:
        return [[_parse_shape(value) for value in args.shape]]

    op_name = (
        args.op
        or _infer_op_from_text(args.task_spec)
        or _infer_op_from_text(args.forward)
        or _infer_op_from_text(args.backward)
    )
    if not op_name:
        raise ValueError(
            "No shapes provided and no known preset could be inferred. "
            "Pass --shape, or use a forward/task-spec name containing a registered op."
        )
    if op_name not in SHAPE_PRESETS:
        raise ValueError(f"No shape presets registered for op {op_name!r}")
    if args.mode not in SHAPE_PRESETS[op_name]:
        raise ValueError(
            f"No mode {args.mode!r} for op {op_name!r}; available: {sorted(SHAPE_PRESETS[op_name])}"
        )
    return SHAPE_PRESETS[op_name][args.mode]


def _run_explicit_check_for_dtype(args: argparse.Namespace, dtype_name: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"passed": False, "error": "CUDA is not available"}

    forward = _load_callable(args.forward)
    if args.backward_file:
        backward = _load_callable_from_file(args.backward_file, args.backward_fn)
    else:
        backward = _load_callable(args.backward) if args.backward else None
    dtype = DTYPES[dtype_name]
    scalars = [_parse_scalar(value) for value in args.scalar]
    reports = []
    passed = True

    for case_index, shapes in enumerate(_shape_sets_from_args(args)):
        try:
            inputs = _make_inputs(shapes, dtype, args.seed + case_index)
            if args.backend == "atenir_compose":
                case_report = _compare_atenir_compose_case(
                    forward_spec=args.forward,
                    inputs=inputs,
                    atol=args.atol,
                    rtol=args.rtol,
                    case_name=f"{args.mode}_{case_index}",
                )
            else:
                case_report = _compare_one_case(
                    forward=forward,
                    backward=backward,
                    inputs=inputs,
                    scalars=scalars,
                    atol=args.atol,
                    rtol=args.rtol,
                    case_name=f"{args.mode}_{case_index}",
                )
        except Exception as exc:
            case_report = {
                "passed": False,
                "case": f"{args.mode}_{case_index}",
                "shapes": [list(shape) for shape in shapes],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(limit=8),
            }
        reports.append(case_report)
        passed = passed and bool(case_report.get("passed"))

    passed_cases = sum(1 for report in reports if report.get("passed"))
    total_cases = len(reports)
    return {
        "passed": passed,
        "passed_cases": passed_cases,
        "failed_cases": total_cases - passed_cases,
        "total_cases": total_cases,
        "forward": args.forward,
        "backward": args.backward,
        "backend": args.backend,
        "dtype": dtype_name,
        "mode": args.mode,
        "scalars": args.scalar,
        "reports": reports,
    }


def _run_explicit_check(args: argparse.Namespace) -> dict[str, Any]:
    reports = [_run_explicit_check_for_dtype(args, dtype_name) for dtype_name in args.dtype]
    passed_cases = sum(report.get("passed_cases", 0) for report in reports)
    total_cases = sum(report.get("total_cases", 0) for report in reports)
    passed = all(report.get("passed") for report in reports)
    return {
        "passed": passed,
        "passed_cases": passed_cases,
        "failed_cases": total_cases - passed_cases,
        "total_cases": total_cases,
        "forward": args.forward,
        "backward": args.backward,
        "backend": args.backend,
        "dtypes": args.dtype,
        "mode": args.mode,
        "scalars": args.scalar,
        "reports": reports,
    }


def _run_task_spec_check_for_dtype(args: argparse.Namespace, dtype_name: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"passed": False, "error": "CUDA is not available"}

    task_spec = _load_module(args.task_spec)
    backward = _load_callable(args.backward)
    dtype = DTYPES[dtype_name]
    reports = []
    passed = True

    for case_index, shapes in enumerate(_shape_sets_from_args(args)):
        rows, cols = shapes[0]
        case = task_spec.TestCase(rows, cols, dtype_name, args.atol, args.rtol)
        try:
            if hasattr(task_spec, "seed_for_case"):
                torch.manual_seed(task_spec.seed_for_case(case))
            inputs = task_spec.make_inputs(torch, case)
            # Respect suite dtype even if task_spec defaults differ.
            converted_inputs = []
            for value in inputs:
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    converted_inputs.append(value.to(dtype=dtype))
                else:
                    converted_inputs.append(value)

            expected = task_spec.torch_oracle(torch, *converted_inputs)
            actual = _normalize_outputs(backward(*converted_inputs))
        except Exception as exc:
            reports.append(
                {
                    "case": [rows, cols],
                    "case_index": case_index,
                    "passed": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
            )
            passed = False
            continue

        if len(actual) != len(expected):
            reports.append(
                {
                    "case": [rows, cols],
                    "passed": False,
                    "error": f"output count mismatch: got {len(actual)}, expected {len(expected)}",
                }
            )
            passed = False
            continue

        for output_index, (actual_tensor, expected_tensor) in enumerate(zip(actual, expected)):
            if actual_tensor.shape != expected_tensor.shape:
                reports.append(
                    {
                        "case": [rows, cols],
                        "output_index": output_index,
                        "passed": False,
                        "error": "shape mismatch",
                        "actual_shape": list(actual_tensor.shape),
                        "expected_shape": list(expected_tensor.shape),
                    }
                )
                passed = False
                continue
            max_abs, max_rel = _max_errors(actual_tensor, expected_tensor)
            ok = bool(torch.allclose(actual_tensor, expected_tensor, atol=args.atol, rtol=args.rtol))
            reports.append(
                {
                    "case": [rows, cols],
                    "case_index": case_index,
                    "output_index": output_index,
                    "passed": ok,
                    "shape": list(actual_tensor.shape),
                    "max_abs": max_abs,
                    "max_rel": max_rel,
                    "atol": args.atol,
                    "rtol": args.rtol,
                }
            )
            passed = passed and ok

    passed_cases = sum(1 for report in reports if report.get("passed"))
    total_cases = len(reports)
    return {
        "passed": passed,
        "passed_cases": passed_cases,
        "failed_cases": total_cases - passed_cases,
        "total_cases": total_cases,
        "task_spec": args.task_spec,
        "backward": args.backward,
        "mode": args.mode,
        "dtype": dtype_name,
        "reports": reports,
    }


def _run_task_spec_check(args: argparse.Namespace) -> dict[str, Any]:
    reports = [_run_task_spec_check_for_dtype(args, dtype_name) for dtype_name in args.dtype]
    passed_cases = sum(report.get("passed_cases", 0) for report in reports)
    total_cases = sum(report.get("total_cases", 0) for report in reports)
    passed = all(report.get("passed") for report in reports)
    return {
        "passed": passed,
        "passed_cases": passed_cases,
        "failed_cases": total_cases - passed_cases,
        "total_cases": total_cases,
        "task_spec": args.task_spec,
        "backward": args.backward,
        "mode": args.mode,
        "dtypes": args.dtype,
        "reports": reports,
    }


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    if args.task_spec and not args.forward:
        return _run_task_spec_check(args)
    return _run_explicit_check(args)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic backward correctness runner")
    parser.add_argument("--forward", help="Forward reference, e.g. module:function")
    parser.add_argument("--backward", help="Backward candidate, e.g. module:function")
    parser.add_argument("--backward-file", help="Path to Python file containing backward candidate")
    parser.add_argument("--backward-fn", default="layernorm_backward_triton")
    parser.add_argument("--backend", choices=["callable", "atenir_compose"], default="callable")
    parser.add_argument("--shape", action="append", default=[], help="Input shape, e.g. 17,127")
    parser.add_argument("--scalar", action="append", default=[], help="Scalar forward/backward arg, e.g. 1e-5")
    parser.add_argument("--dtype", action="append", choices=sorted(DTYPES), default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--task-spec", default=None, help="Task spec module, e.g. examples.foo.task_spec")
    parser.add_argument("--op", default=None, help="Operation preset name, e.g. layernorm")
    parser.add_argument("--mode", default="static", choices=["static", "dynamic", "nontile"])
    args = parser.parse_args(argv)
    if args.dtype is None:
        args.dtype = ["float32"]
    if args.forward is None and args.task_spec is None:
        parser.error("pass --forward for generic mode, or --task-spec for task-spec oracle mode")
    if args.backend == "callable" and args.task_spec is None and args.backward is None and args.backward_file is None:
        parser.error("--backend callable requires --backward or --backward-file")
    return args


def main(argv: list[str] | None = None) -> int:
    report = run_check(parse_args(argv or sys.argv[1:]))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
