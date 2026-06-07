"""Latency benchmark for the bias + ReLU backward sample."""

from dataclasses import dataclass
import statistics

import torch

from backward_naive_triton import bias_relu_backward_naive_triton
from forward_ref import bias_relu_forward_ref


@dataclass(frozen=True)
class BenchCase:
    rows: int
    cols: int
    dtype: torch.dtype


CASES = [
    BenchCase(512, 1024, torch.float16),
    BenchCase(1024, 1024, torch.float16),
    BenchCase(1024, 2048, torch.float16),
]


def _sync() -> None:
    torch.cuda.synchronize()


def _median_ms(fn, warmup: int = 10, reps: int = 50) -> float:
    for _ in range(warmup):
        fn()
    _sync()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings = []
    for _ in range(reps):
        start.record()
        fn()
        end.record()
        _sync()
        timings.append(float(start.elapsed_time(end)))
    return float(statistics.median(timings))


def _make_inputs(case: BenchCase):
    torch.manual_seed(case.rows * 100000 + case.cols)
    x = torch.randn((case.rows, case.cols), device="cuda", dtype=case.dtype)
    bias = torch.randn((case.cols,), device="cuda", dtype=case.dtype)
    dy = torch.randn_like(x)
    return dy, x, bias


def _pytorch_autograd_step(dy: torch.Tensor, x: torch.Tensor, bias: torch.Tensor) -> None:
    x_ref = x.detach().clone().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)
    y_ref = bias_relu_forward_ref(x_ref, bias_ref)
    y_ref.backward(dy)


def run_case(case: BenchCase) -> dict[str, float | str | list[int]]:
    dy, x, bias = _make_inputs(case)
    triton_ms = _median_ms(lambda: bias_relu_backward_naive_triton(dy, x, bias))
    pytorch_ms = _median_ms(lambda: _pytorch_autograd_step(dy, x, bias))
    return {
        "shape": [case.rows, case.cols],
        "dtype": str(case.dtype).replace("torch.", ""),
        "triton_backward_ms": triton_ms,
        "pytorch_autograd_forward_backward_ms": pytorch_ms,
        "speedup_vs_pytorch_step": pytorch_ms / max(triton_ms, 1e-9),
    }


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available; run this benchmark on a GPU node.")
        return 0

    for case in CASES:
        print(run_case(case))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
