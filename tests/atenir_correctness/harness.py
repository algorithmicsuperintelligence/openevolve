"""Small unified correctness harness for AtenIR and callable backends.

The harness intentionally focuses on correctness, not timing.  It compares every
backend output against a PyTorch-autograd oracle, per gradient.
"""

from __future__ import annotations

import importlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch


@dataclass(frozen=True)
class ShapeCase:
    name: str
    shapes: tuple[tuple[int, ...], ...]
    dtype: torch.dtype = torch.float32
    atol: float = 1e-4
    rtol: float = 1e-4
    seed: int = 0


@dataclass(frozen=True)
class CompareReport:
    backend: str
    case: str
    output_index: int
    max_abs: float
    max_rel: float
    passed: bool


def require_cuda_or_skip():
    try:
        import triton  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("triton is not installed")
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA is not available")


def _load_callable(fn_spec: str) -> Callable:
    module_name, fn_name = fn_spec.split(":", 1) if ":" in fn_spec else fn_spec.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), fn_name)


def make_float_inputs(case: ShapeCase, device: str = "cuda") -> list[torch.Tensor]:
    torch.manual_seed(case.seed)
    return [
        torch.randn(shape, device=device, dtype=case.dtype, requires_grad=True)
        for shape in case.shapes
    ]


def autograd_oracle(fn: Callable, inputs: Sequence[torch.Tensor]):
    refs = [tensor.detach().clone().requires_grad_(True) for tensor in inputs]
    out = fn(*refs)
    if isinstance(out, (tuple, list)):
        out = out[0]
    grad_out = torch.randn_like(out)
    grads = torch.autograd.grad(out, refs, grad_outputs=grad_out)
    return grad_out.detach(), tuple(grad.detach() for grad in grads)


def compare_outputs(
    backend_name: str,
    case: ShapeCase,
    actual: Sequence[torch.Tensor],
    expected: Sequence[torch.Tensor],
) -> list[CompareReport]:
    if len(actual) != len(expected):
        raise AssertionError(f"{backend_name}: got {len(actual)} outputs, expected {len(expected)}")

    reports = []
    for idx, (got, ref) in enumerate(zip(actual, expected)):
        if got.shape != ref.shape:
            raise AssertionError(
                f"{backend_name} {case.name}: output {idx} shape {got.shape} != {ref.shape}"
            )
        diff = (got.float() - ref.float()).abs()
        max_abs = float(torch.max(diff).item())
        denom = torch.clamp(ref.float().abs(), min=1e-8)
        max_rel = float(torch.max(diff / denom).item())
        passed = bool(torch.allclose(got.float(), ref.float(), atol=case.atol, rtol=case.rtol))
        reports.append(
            CompareReport(
                backend=backend_name,
                case=case.name,
                output_index=idx,
                max_abs=max_abs,
                max_rel=max_rel,
                passed=passed,
            )
        )
    return reports


class CallableBackend:
    def __init__(self, name: str, fn: Callable):
        self.name = name
        self.fn = fn

    def run(self, grad_out: torch.Tensor, inputs: Sequence[torch.Tensor]):
        return tuple(self.fn(grad_out, *[tensor.detach().contiguous() for tensor in inputs]))


class AtenIRComposeBackend:
    """Haochen-style extract -> make_registry -> run_graph backend."""

    def __init__(
        self,
        name: str,
        fn_spec: str,
        extract_shapes: tuple[tuple[int, ...], ...] | None = None,
    ):
        self.name = name
        self.fn_spec = fn_spec
        self.extract_shapes = extract_shapes
        self.fallback_report: dict[str, int] = {}

    def _extract_graph(self, runtime_shapes: tuple[tuple[int, ...], ...]):
        from atenir.extract import _parse_spec, _serialise, extract_autograd

        shapes = self.extract_shapes or runtime_shapes
        spec_str = "[" + ", ".join(
            "(" + ",".join(str(dim) for dim in shape) + ") f32" for shape in shapes
        ) + "]"
        graph_module = extract_autograd(self.fn_spec, _parse_spec(spec_str), device="cpu")
        return _serialise(graph_module)

    def _fallback_counts(self, graph: dict) -> dict[str, int]:
        total = 0
        fallback = 0
        triton_like = 0
        fallback_targets = (
            "aten.mm",
            "aten.bmm",
            "aten.matmul",
            "aten.scatter",
            "aten.index",
            "aten.log_softmax",
            "aten.select",
            "aten.slice",
            "aten.topk",
        )
        for node in graph["nodes"]:
            if node.get("op") != "call_function":
                continue
            total += 1
            target = str(node.get("target", ""))
            if any(item in target for item in fallback_targets):
                fallback += 1
            else:
                triton_like += 1
        return {"total": total, "triton_or_simple": triton_like, "fallback": fallback}

    def run(self, grad_out: torch.Tensor, inputs: Sequence[torch.Tensor]):
        from atenir.compose import run_graph
        from atenir.primitive_triton.dispatch import make_registry

        graph = self._extract_graph(tuple(tuple(t.shape) for t in inputs))
        self.fallback_report = self._fallback_counts(graph)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
            json.dump(graph, handle)
            graph_path = handle.name
        try:
            registry = make_registry(graph)
            placeholders = [node for node in graph["nodes"] if node["op"] == "placeholder"]
            env_tensors = [grad_out.contiguous()] + [tensor.detach().contiguous() for tensor in inputs]
            if len(placeholders) != len(env_tensors):
                raise AssertionError(
                    f"placeholder count {len(placeholders)} != tensor count {len(env_tensors)}"
                )
            env = {node["name"]: tensor for node, tensor in zip(placeholders, env_tensors)}
            return tuple(run_graph(graph_path, env, registry))
        finally:
            Path(graph_path).unlink(missing_ok=True)


class LayerNormAtenIRBackend:
    """Adapter backend for examples/triton_layernorm_backward_bench/backward_atenir.py."""

    name = "layernorm_atenir"

    def run(self, grad_out: torch.Tensor, inputs: Sequence[torch.Tensor]):
        from examples.triton_layernorm_backward_bench.backward_atenir import (
            layernorm_backward_triton,
        )

        x, weight, bias = inputs
        return tuple(layernorm_backward_triton(grad_out, x.detach(), weight.detach(), bias.detach()))


def check_backend_against_autograd(backend, fn_spec: str, case: ShapeCase) -> list[CompareReport]:
    fn = _load_callable(fn_spec)
    inputs = make_float_inputs(case)
    grad_out, expected = autograd_oracle(fn, inputs)
    actual = backend.run(grad_out, inputs)
    reports = compare_outputs(backend.name, case, actual, expected)
    failed = [report for report in reports if not report.passed]
    if failed:
        details = "\n".join(
            f"{r.backend}:{r.case}:grad[{r.output_index}] max_abs={r.max_abs:.3e} "
            f"max_rel={r.max_rel:.3e}"
            for r in failed
        )
        raise AssertionError(details)
    return reports
