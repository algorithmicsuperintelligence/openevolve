"""Latency benchmark for the Linear backward sample."""

import statistics

import torch
import torch.nn.functional as F

from backward_naive_triton import linear_backward_naive_triton
from task_spec import BENCHMARK_CASES, TestCase, make_inputs, seed_for_case


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


def _pytorch_autograd_step(dy: torch.Tensor, x: torch.Tensor, weight: torch.Tensor) -> None:
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    bias_ref = torch.zeros(weight.shape[0], device=x.device, dtype=x.dtype, requires_grad=True)
    y_ref = F.linear(x_ref, weight_ref, bias_ref)
    y_ref.backward(dy)


def run_case(case: TestCase) -> dict[str, float | str | list[int]]:
    torch.manual_seed(seed_for_case(case))
    dy, x, weight = make_inputs(torch, case)
    triton_ms = _median_ms(lambda: linear_backward_naive_triton(dy, x, weight))
    pytorch_ms = _median_ms(lambda: _pytorch_autograd_step(dy, x, weight))
    return {
        "shape": [case.m, case.n, case.k],
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
