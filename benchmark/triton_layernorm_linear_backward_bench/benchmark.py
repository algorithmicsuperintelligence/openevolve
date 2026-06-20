"""Latency benchmark for the fused LayerNorm -> Linear backward sample."""

import statistics

import torch
import torch.nn.functional as F

from backward_naive import layernorm_linear_backward_naive
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


def _pytorch_autograd_step(dout, x, weight, bias, linear_weight, eps) -> None:
    k = x.shape[-1]
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)
    lw_ref = linear_weight.detach().clone().requires_grad_(True)
    out = F.layer_norm(x_ref, (k,), weight_ref, bias_ref, eps) @ lw_ref
    out.backward(dout)


def run_case(case: TestCase) -> dict[str, float | str | list[int]]:
    torch.manual_seed(seed_for_case(case))
    dout, x, weight, bias, linear_weight, eps = make_inputs(torch, case)
    naive_ms = _median_ms(
        lambda: layernorm_linear_backward_naive(dout, x, weight, bias, linear_weight, eps)
    )
    pytorch_ms = _median_ms(
        lambda: _pytorch_autograd_step(dout, x, weight, bias, linear_weight, eps)
    )
    return {
        "shape": [case.m, case.k, case.n],
        "dtype": case.dtype_name,
        "naive_backward_ms": naive_ms,
        "pytorch_autograd_forward_backward_ms": pytorch_ms,
        "speedup_vs_pytorch_step": pytorch_ms / max(naive_ms, 1e-9),
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
