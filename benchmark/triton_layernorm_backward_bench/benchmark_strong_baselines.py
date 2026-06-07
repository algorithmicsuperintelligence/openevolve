"""Compare evolved LayerNorm backward kernels against strong external baselines.

Usage:
    python benchmark_strong_baselines.py
    python benchmark_strong_baselines.py evolved_best_program.py

OpenEvolve fitness uses PyTorch autograd for both correctness and speedup.
This script is for post-optimization reporting against stronger references:
    - PyTorch autograd backward-only (matches evaluator API)
    - PyTorch native eager / torch.compile training step
    - naive Triton seed (legacy reference)
    - Liger LayerNorm (optional, requires ``pip install liger-kernel``)
"""

from __future__ import annotations

import importlib.util
import statistics
import sys
import uuid
from typing import Callable

import torch

from backward_naive_triton import layernorm_backward_naive_triton
from forward_triton import layernorm_forward_triton
from strong_baselines.liger_layernorm import (
    liger_available,
    make_liger_layernorm_backward_fn,
    make_liger_layernorm_train_step_fn,
)
from task_spec import BENCHMARK_CASES, EPS, TestCase, _dtype, torch_oracle


def _median_ms(fn: Callable[[], object], warmup: int = 10, reps: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings = []
    for _ in range(reps):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(float(start.elapsed_time(end)))
    return float(statistics.median(timings))


def _make_inputs(case: TestCase):
    torch.manual_seed(case.rows * 100000 + case.cols)
    dtype = _dtype(torch, case.dtype_name)
    x = torch.randn((case.rows, case.cols), device="cuda", dtype=dtype)
    weight = torch.randn((case.cols,), device="cuda", dtype=dtype)
    bias = torch.randn((case.cols,), device="cuda", dtype=dtype)
    dy = torch.randn_like(x)
    return dy, x, weight, bias


def _load_candidate(program_path: str):
    module_name = f"candidate_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load candidate from {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "layernorm_backward_triton", None)
    if fn is None or not callable(fn):
        raise AttributeError("candidate must define layernorm_backward_triton(dy, x, weight, bias, eps)")
    return fn


def _native_layernorm_loss(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    dy: torch.Tensor,
):
    y = torch.nn.functional.layer_norm(
        x,
        normalized_shape=(x.shape[1],),
        weight=weight,
        bias=bias,
        eps=EPS,
    )
    return torch.sum(y * dy)


def _make_train_tensors(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    x_req = x.detach().clone().requires_grad_(True)
    weight_req = weight.detach().clone().requires_grad_(True)
    bias_req = bias.detach().clone().requires_grad_(True)
    return x_req, weight_req, bias_req


def _zero_grads(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        tensor.grad = None


def _loss_backward_step(loss_fn: Callable, x_req, weight_req, bias_req, dy):
    _zero_grads(x_req, weight_req, bias_req)
    loss = loss_fn(x_req, weight_req, bias_req, dy)
    loss.backward()
    return x_req.grad, weight_req.grad, bias_req.grad


def _compile_fn(fn: Callable):
    try:
        return torch.compile(fn, fullgraph=False)
    except Exception as exc:
        print(f"WARN: torch.compile failed to create compiled function: {exc}")
        return None


def _max_abs_tuple(actual, expected) -> float:
    return max(float(torch.max(torch.abs(a.float() - e.float())).item()) for a, e in zip(actual, expected))


def _ratio(numerator_ms: float | None, denominator_ms: float | None) -> float | None:
    if numerator_ms is None or denominator_ms is None:
        return None
    return numerator_ms / max(denominator_ms, 1e-9)


def run_case(case: TestCase, candidate_fn: Callable | None) -> dict[str, float | str | list[int] | None]:
    dy, x, weight, bias = _make_inputs(case)
    native_x, native_weight, native_bias = _make_train_tensors(x, weight, bias)
    native_compile_x, native_compile_weight, native_compile_bias = _make_train_tensors(x, weight, bias)

    pytorch_autograd_ms = _median_ms(lambda: torch_oracle(torch, dy, x, weight, bias, EPS))
    naive_ms = _median_ms(lambda: layernorm_backward_naive_triton(dy, x, weight, bias, EPS))
    naive_full_step_ms = _median_ms(
        lambda: (
            layernorm_forward_triton(x, weight, bias, EPS),
            layernorm_backward_naive_triton(dy, x, weight, bias, EPS),
        )
    )

    native_eager_step_ms = _median_ms(
        lambda: _loss_backward_step(_native_layernorm_loss, native_x, native_weight, native_bias, dy)
    )

    native_compiled_loss = _compile_fn(_native_layernorm_loss)
    native_compile_step_ms = None
    if native_compiled_loss is not None:
        try:
            _loss_backward_step(native_compiled_loss, native_compile_x, native_compile_weight, native_compile_bias, dy)
            torch.cuda.synchronize()
            native_compile_step_ms = _median_ms(
                lambda: _loss_backward_step(
                    native_compiled_loss,
                    native_compile_x,
                    native_compile_weight,
                    native_compile_bias,
                    dy,
                )
            )
        except Exception as exc:
            print(f"WARN: compiled native training step failed: {type(exc).__name__}: {exc}")

    liger_backward_ms = None
    liger_full_step_ms = None
    if liger_available():
        liger_backward_fn = make_liger_layernorm_backward_fn()
        liger_train_step = make_liger_layernorm_train_step_fn(EPS)
        liger_x, liger_weight, liger_bias = _make_train_tensors(x, weight, bias)
        try:
            liger_backward_fn(dy, x, weight, bias, EPS)
            torch.cuda.synchronize()
            liger_backward_ms = _median_ms(lambda: liger_backward_fn(dy, x, weight, bias, EPS))
            _loss_backward_step(liger_train_step, liger_x, liger_weight, liger_bias, dy)
            torch.cuda.synchronize()
            liger_full_step_ms = _median_ms(
                lambda: _loss_backward_step(liger_train_step, liger_x, liger_weight, liger_bias, dy)
            )
        except Exception as exc:
            print(f"WARN: Liger LayerNorm baseline failed: {type(exc).__name__}: {exc}")
    else:
        print("WARN: liger-kernel is not installed; skipping Liger baselines.")

    candidate_ms = None
    candidate_full_step_ms = None
    candidate_max_abs_vs_pytorch = None
    if candidate_fn is not None:
        expected = torch_oracle(torch, dy, x, weight, bias, EPS)
        actual = candidate_fn(dy, x, weight, bias, EPS)
        torch.cuda.synchronize()
        candidate_max_abs_vs_pytorch = _max_abs_tuple(actual, expected)
        candidate_ms = _median_ms(lambda: candidate_fn(dy, x, weight, bias, EPS))
        candidate_full_step_ms = _median_ms(
            lambda: (
                layernorm_forward_triton(x, weight, bias, EPS),
                candidate_fn(dy, x, weight, bias, EPS),
            )
        )

    return {
        "shape": [case.rows, case.cols],
        "dtype": case.dtype_name,
        "pytorch_autograd_backward_ms": pytorch_autograd_ms,
        "naive_triton_backward_ms": naive_ms,
        "naive_triton_forward_backward_ms": naive_full_step_ms,
        "liger_backward_ms": liger_backward_ms,
        "liger_forward_backward_ms": liger_full_step_ms,
        "candidate_triton_backward_ms": candidate_ms,
        "candidate_triton_forward_backward_ms": candidate_full_step_ms,
        "pytorch_native_eager_train_step_ms": native_eager_step_ms,
        "pytorch_native_compile_train_step_ms": native_compile_step_ms,
        "candidate_max_abs_vs_pytorch_autograd": candidate_max_abs_vs_pytorch,
        "candidate_speedup_vs_pytorch_autograd": _ratio(pytorch_autograd_ms, candidate_ms),
        "candidate_speedup_vs_naive_triton": _ratio(naive_ms, candidate_ms),
        "candidate_speedup_vs_liger_backward": _ratio(liger_backward_ms, candidate_ms),
        "candidate_full_step_speedup_vs_liger_full_step": _ratio(liger_full_step_ms, candidate_full_step_ms),
        "candidate_full_step_speedup_vs_pytorch_eager_train_step": _ratio(
            native_eager_step_ms,
            candidate_full_step_ms,
        ),
        "candidate_full_step_speedup_vs_pytorch_compile_train_step": _ratio(
            native_compile_step_ms,
            candidate_full_step_ms,
        ),
    }


def main(argv: list[str]) -> int:
    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available; run this benchmark on a GPU node.")
        return 0

    candidate_fn = _load_candidate(argv[1]) if len(argv) == 2 else None
    if len(argv) > 2:
        print(f"Usage: {argv[0]} [OPTIONAL_CANDIDATE_PROGRAM.py]")
        return 2

    for case in BENCHMARK_CASES:
        print(run_case(case, candidate_fn))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
