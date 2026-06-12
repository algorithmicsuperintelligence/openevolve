"""Latency benchmark for the matmul backward sample."""

import statistics

import torch

from backward_naive_triton import matmul_backward_naive_triton
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


def _pytorch_autograd_step(dc: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> None:
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    c_ref = a_ref @ b_ref
    c_ref.backward(dc)


def run_case(case: TestCase) -> dict[str, float | str | list[int]]:
    torch.manual_seed(seed_for_case(case))
    dc, a, b = make_inputs(torch, case)
    triton_ms = _median_ms(lambda: matmul_backward_naive_triton(dc, a, b))
    pytorch_ms = _median_ms(lambda: _pytorch_autograd_step(dc, a, b))
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
