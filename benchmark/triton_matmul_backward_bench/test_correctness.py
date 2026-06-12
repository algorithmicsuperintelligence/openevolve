"""Randomized correctness checks for the matmul backward sample."""

import torch

from autograd_wrapper import matmul_autograd_triton
from backward_naive_triton import matmul_backward_naive_triton
from backward_ref import matmul_backward_ref
from forward_ref import matmul_forward_ref
from forward_triton import matmul_forward_triton
from task_spec import CORRECTNESS_CASES, TestCase, _dtype, make_inputs, seed_for_case


def _max_abs(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(torch.max(torch.abs(actual.float() - expected.float())).item())


def _check_close(name: str, actual: torch.Tensor, expected: torch.Tensor, case: TestCase) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"{name} shape mismatch: got {actual.shape}, expected {expected.shape}")
    if not torch.allclose(actual, expected, atol=case.atol_value, rtol=case.rtol_value):
        raise AssertionError(
            f"{name} mismatch for (M,N,K)=({case.m},{case.n},{case.k}) dtype={case.dtype_name}: "
            f"max_abs={_max_abs(actual, expected):.6e}, "
            f"atol={case.atol_value}, rtol={case.rtol_value}"
        )


def run_case(case: TestCase) -> dict[str, float | str | list[int]]:
    torch.manual_seed(seed_for_case(case))
    dc, a, b = make_inputs(torch, case)

    expected_c = matmul_forward_ref(a, b)
    actual_c = matmul_forward_triton(a, b)
    torch.cuda.synchronize()
    _check_close("forward c", actual_c, expected_c, case)

    exp_da, exp_db = matmul_backward_ref(dc, a, b)
    act_da, act_db = matmul_backward_naive_triton(dc, a, b)
    torch.cuda.synchronize()
    _check_close("da", act_da, exp_da, case)
    _check_close("db", act_db, exp_db, case)

    wrapped_a = a.detach().clone().requires_grad_(True)
    wrapped_b = b.detach().clone().requires_grad_(True)
    wrapped_c = matmul_autograd_triton(wrapped_a, wrapped_b)
    wrapped_c.backward(dc)
    torch.cuda.synchronize()
    _check_close("autograd da", wrapped_a.grad, exp_da, case)
    _check_close("autograd db", wrapped_b.grad, exp_db, case)

    return {
        "shape": [case.m, case.n, case.k],
        "dtype": case.dtype_name,
        "forward_max_abs": _max_abs(actual_c, expected_c),
        "da_max_abs": _max_abs(act_da, exp_da),
        "db_max_abs": _max_abs(act_db, exp_db),
    }


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available; run this test on a GPU node.")
        return 0

    reports = [run_case(case) for case in CORRECTNESS_CASES]
    for report in reports:
        print(report)
    print(f"PASS: {len(reports)} randomized matmul correctness cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
