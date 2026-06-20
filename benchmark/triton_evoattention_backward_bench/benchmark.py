"""Latency benchmark for the EvoformerAttention backward sample."""

import statistics

import torch

from backward_naive import evoattention_backward_naive
from forward_ref import evoattention_forward_ref
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


def _pytorch_autograd_step(do, q, k, v, res_mask, pair_bias) -> None:
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    pair_bias_ref = pair_bias.detach().clone().requires_grad_(True)
    out = evoattention_forward_ref(q_ref, k_ref, v_ref, res_mask, pair_bias_ref)
    out.backward(do)


def run_case(case: TestCase) -> dict[str, float | str | list[int]]:
    torch.manual_seed(seed_for_case(case))
    do, q, k, v, res_mask, pair_bias = make_inputs(torch, case)
    naive_ms = _median_ms(lambda: evoattention_backward_naive(do, q, k, v, res_mask, pair_bias))
    pytorch_ms = _median_ms(lambda: _pytorch_autograd_step(do, q, k, v, res_mask, pair_bias))
    return {
        "shape": [case.b, case.n_seq, case.head, case.n_res, case.dim],
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
