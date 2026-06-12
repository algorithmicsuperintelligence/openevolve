"""Randomized correctness checks for the Linear backward sample."""

import torch

from autograd_wrapper import linear_autograd_triton
from backward_naive_triton import linear_backward_naive_triton
from backward_ref import linear_backward_ref
from forward_ref import linear_forward_ref
from forward_triton import linear_forward_triton
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
    dtype = _dtype(torch, case.dtype_name)
    dy, x, weight = make_inputs(torch, case)
    bias = torch.randn(case.n, device="cuda", dtype=dtype)

    expected_y = linear_forward_ref(x, weight, bias)
    actual_y = linear_forward_triton(x, weight, bias)
    torch.cuda.synchronize()
    _check_close("forward y", actual_y, expected_y, case)

    exp_dx, exp_dweight, exp_dbias = linear_backward_ref(dy, x, weight)
    act_dx, act_dweight, act_dbias = linear_backward_naive_triton(dy, x, weight)
    torch.cuda.synchronize()
    _check_close("dx", act_dx, exp_dx, case)
    _check_close("dweight", act_dweight, exp_dweight, case)
    _check_close("dbias", act_dbias, exp_dbias, case)

    wrapped_x = x.detach().clone().requires_grad_(True)
    wrapped_weight = weight.detach().clone().requires_grad_(True)
    wrapped_bias = bias.detach().clone().requires_grad_(True)
    wrapped_y = linear_autograd_triton(wrapped_x, wrapped_weight, wrapped_bias)
    wrapped_y.backward(dy)
    torch.cuda.synchronize()
    _check_close("autograd dx", wrapped_x.grad, exp_dx, case)
    _check_close("autograd dweight", wrapped_weight.grad, exp_dweight, case)
    _check_close("autograd dbias", wrapped_bias.grad, exp_dbias, case)

    return {
        "shape": [case.m, case.n, case.k],
        "dtype": case.dtype_name,
        "forward_max_abs": _max_abs(actual_y, expected_y),
        "dx_max_abs": _max_abs(act_dx, exp_dx),
        "dweight_max_abs": _max_abs(act_dweight, exp_dweight),
        "dbias_max_abs": _max_abs(act_dbias, exp_dbias),
    }


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available; run this test on a GPU node.")
        return 0

    reports = [run_case(case) for case in CORRECTNESS_CASES]
    for report in reports:
        print(report)
    print(f"PASS: {len(reports)} randomized Linear correctness cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
