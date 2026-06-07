"""Correctness verifier for the primitive AtenIR lowering agent.

Loads a graph JSON and a kernels.py file (produced by the lowering agent),
builds a kernel registry via make_kernel_registry(), runs the graph with
atenir.compose.run_graph, and compares the output against a PyTorch autograd
reference computed from the original forward function.

The graph must have been extracted in --fn (autograd) mode so that its
placeholders are ordered as [grad_out, *forward_inputs].

Outputs a JSON report to stdout.

Example:
    python -m pipeline.run_lowering_correctness \\
        --forward benchmark.triton_layernorm_backward_bench.forward_ref:layernorm_forward_ref \\
        --graph-json /path/to/atenir_graph.json \\
        --kernels-file /path/to/kernels.py \\
        --dtype float32 \\
        --atol 2e-5 --rtol 2e-5
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


def _load_callable(spec: str):
    module_name, fn_name = spec.split(":", 1) if ":" in spec else spec.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), fn_name)


def _import_registry(kernels_file: str) -> dict:
    """Dynamically import kernels.py and return the kernel registry dict."""
    module_name = f"atenir_lowering_kernels_{uuid.uuid4().hex}"
    path = Path(kernels_file).resolve()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load kernels file: {kernels_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "make_kernel_registry"):
        raise AttributeError(f"kernels file has no make_kernel_registry(): {kernels_file}")
    return module.make_kernel_registry()


def _run_for_dtype(
    *,
    forward_spec: str,
    graph_json: str,
    kernels_file: str,
    dtype_name: str,
    atol: float,
    rtol: float,
    seed: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"passed": False, "error": "CUDA not available", "dtype": dtype_name}

    # Load the graph
    from atenir.compose import run_graph

    graph = json.loads(Path(graph_json).read_text())
    placeholders = [n for n in graph["nodes"] if n["op"] == "placeholder"]
    call_nodes = [n for n in graph["nodes"] if n["op"] == "call_function"]

    # Import the registry
    try:
        registry = _import_registry(kernels_file)
    except Exception as exc:
        return {
            "passed": False,
            "dtype": dtype_name,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }

    # Check all nodes have kernels
    missing = [n["name"] for n in call_nodes if n["name"] not in registry]
    if missing:
        return {
            "passed": False,
            "dtype": dtype_name,
            "error_type": "MissingKernels",
            "error": f"No kernel registered for nodes: {missing}",
        }

    dtype = DTYPES[dtype_name]
    forward_fn = _load_callable(forward_spec)

    # The graph was extracted with --fn (autograd) mode.
    # Placeholders: [grad_out, *forward_inputs]
    fwd_shapes = [tuple(ph["shape"]) for ph in placeholders[1:]]

    torch.manual_seed(seed)
    try:
        inputs = [torch.randn(shape, dtype=dtype, device="cuda") for shape in fwd_shapes]

        # Compute autograd reference
        ref_inputs = [t.detach().clone().requires_grad_(True) for t in inputs]
        out = forward_fn(*ref_inputs)
        if isinstance(out, (tuple, list)):
            out = out[0]
        grad_out = torch.randn_like(out)
        expected_grads = tuple(
            torch.autograd.grad(out, ref_inputs, grad_outputs=grad_out)
        )

        # Build environment for run_graph
        env_tensors = [grad_out.detach().contiguous()] + [
            t.detach().contiguous() for t in inputs
        ]
        if len(placeholders) != len(env_tensors):
            return {
                "passed": False,
                "dtype": dtype_name,
                "error_type": "PlaceholderMismatch",
                "error": (
                    f"Graph has {len(placeholders)} placeholders but "
                    f"we built {len(env_tensors)} env tensors. "
                    "The graph may have extra saved-tensor placeholders."
                ),
            }
        env = {ph["name"]: t for ph, t in zip(placeholders, env_tensors)}

        # Run graph with LLM kernels
        actual_grads = run_graph(graph_json, env, registry)

    except Exception as exc:
        return {
            "passed": False,
            "dtype": dtype_name,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }

    # Compare outputs
    if len(actual_grads) != len(expected_grads):
        return {
            "passed": False,
            "dtype": dtype_name,
            "error_type": "OutputCountMismatch",
            "error": f"got {len(actual_grads)} outputs, expected {len(expected_grads)}",
        }

    grad_reports = []
    passed = True
    for i, (actual, expected) in enumerate(zip(actual_grads, expected_grads)):
        if actual.shape != expected.shape:
            grad_reports.append(
                {
                    "index": i,
                    "passed": False,
                    "error": "shape mismatch",
                    "actual_shape": list(actual.shape),
                    "expected_shape": list(expected.shape),
                }
            )
            passed = False
            continue
        diff = (actual.float() - expected.float()).abs()
        max_abs = float(diff.max().item())
        denom = expected.float().abs().clamp(min=1e-8)
        max_rel = float((diff / denom).max().item())
        ok = bool(torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol))
        grad_reports.append(
            {
                "index": i,
                "passed": ok,
                "shape": list(actual.shape),
                "max_abs": max_abs,
                "max_rel": max_rel,
                "atol": atol,
                "rtol": rtol,
            }
        )
        passed = passed and ok

    passed_cases = sum(1 for r in grad_reports if r.get("passed"))
    total_cases = len(grad_reports)
    return {
        "passed": passed,
        "passed_cases": passed_cases,
        "failed_cases": total_cases - passed_cases,
        "total_cases": total_cases,
        "dtype": dtype_name,
        "grad_reports": grad_reports,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AtenIR lowering correctness verifier")
    parser.add_argument("--forward", required=True, help="Forward callable spec, e.g. pkg.mod:fn")
    parser.add_argument("--graph-json", required=True, help="Path to extracted AtenIR graph JSON")
    parser.add_argument("--kernels-file", required=True, help="Path to assembled kernels.py")
    parser.add_argument(
        "--dtype",
        action="append",
        choices=sorted(DTYPES),
        default=None,
        help="Data type(s) to test (repeatable)",
    )
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv or sys.argv[1:])
    if args.dtype is None:
        args.dtype = ["float32"]

    reports = [
        _run_for_dtype(
            forward_spec=args.forward,
            graph_json=args.graph_json,
            kernels_file=args.kernels_file,
            dtype_name=dtype_name,
            atol=args.atol,
            rtol=args.rtol,
            seed=args.seed,
        )
        for dtype_name in args.dtype
    ]

    passed_cases = sum(r.get("passed_cases", 0) for r in reports)
    total_cases = sum(r.get("total_cases", 0) for r in reports)
    passed = all(r.get("passed") for r in reports)
    result = {
        "passed": passed,
        "passed_cases": passed_cases,
        "failed_cases": total_cases - passed_cases,
        "total_cases": total_cases,
        "dtypes": args.dtype,
        "reports": reports,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
