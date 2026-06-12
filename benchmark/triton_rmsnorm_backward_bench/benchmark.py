"""Latency benchmark for the RMSNorm backward sample."""

import statistics

import torch

from backward_naive_triton import rmsnorm_backward_naive_triton
from forward_ref import rmsnorm_forward_ref
from task_spec import BENCHMARK_CASES, EPS, TestCase, _dtype


def _median_ms(fn, warmup: int = 10, reps: int = 50) -> float:
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
    dy = torch.randn_like(x)
    return dy, x, weight


def _pytorch_autograd_step(dy: torch.Tensor, x: torch.Tensor, weight: torch.Tensor) -> None:
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    y_ref = rmsnorm_forward_ref(x_ref, weight_ref, EPS)
    y_ref.backward(dy)


def run_case(case: TestCase) -> dict[str, float | str | list[int]]:
    dy, x, weight = _make_inputs(case)
    triton_ms = _median_ms(lambda: rmsnorm_backward_naive_triton(dy, x, weight, EPS))
    pytorch_ms = _median_ms(lambda: _pytorch_autograd_step(dy, x, weight))
    return {
        "shape": [case.rows, case.cols],
        "dtype": case.dtype_name,
        "triton_backward_ms": triton_ms,
        "pytorch_autograd_forward_backward_ms": pytorch_ms,
        "speedup_vs_pytorch_step": pytorch_ms / max(triton_ms, 1e-9),
    }


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available; run this benchmark on a GPU node.")
        return 0

    for case in BENCHMARK_CASES:
        print(run_case(case))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
