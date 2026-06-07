"""Randomized correctness checks for the LayerNorm backward sample."""

import torch

from autograd_wrapper import layernorm_autograd_triton
from backward_naive_triton import layernorm_backward_naive_triton
from backward_ref import layernorm_backward_ref
from forward_ref import layernorm_forward_ref
from forward_triton import layernorm_forward_triton
from task_spec import CORRECTNESS_CASES, EPS, TestCase, _dtype


def _max_abs(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(torch.max(torch.abs(actual.float() - expected.float())).item())


def _check_close(name: str, actual: torch.Tensor, expected: torch.Tensor, case: TestCase) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"{name} shape mismatch: got {actual.shape}, expected {expected.shape}")
    if not torch.allclose(actual, expected, atol=case.atol_value, rtol=case.rtol_value):
        raise AssertionError(
            f"{name} mismatch for shape=({case.rows}, {case.cols}) dtype={case.dtype_name}: "
            f"max_abs={_max_abs(actual, expected):.6e}, "
            f"atol={case.atol_value}, rtol={case.rtol_value}"
        )


def run_case(case: TestCase) -> dict[str, float | str | list[int]]:
    torch.manual_seed(case.rows * 100000 + case.cols)
    dtype = _dtype(torch, case.dtype_name)
    x = torch.randn((case.rows, case.cols), device="cuda", dtype=dtype)
    weight = torch.randn((case.cols,), device="cuda", dtype=dtype)
    bias = torch.randn((case.cols,), device="cuda", dtype=dtype)
    dy = torch.randn_like(x)

    expected_y = layernorm_forward_ref(x, weight, bias, EPS)
    actual_y = layernorm_forward_triton(x, weight, bias, EPS)
    torch.cuda.synchronize()
    _check_close("forward y", actual_y, expected_y, case)

    expected_dx, expected_dweight, expected_dbias = layernorm_backward_ref(dy, x, weight, bias, EPS)
    actual_dx, actual_dweight, actual_dbias = layernorm_backward_naive_triton(dy, x, weight, bias, EPS)
    torch.cuda.synchronize()
    _check_close("dx", actual_dx, expected_dx, case)
    _check_close("dweight", actual_dweight, expected_dweight, case)
    _check_close("dbias", actual_dbias, expected_dbias, case)

    wrapped_x = x.detach().clone().requires_grad_(True)
    wrapped_weight = weight.detach().clone().requires_grad_(True)
    wrapped_bias = bias.detach().clone().requires_grad_(True)
    wrapped_y = layernorm_autograd_triton(wrapped_x, wrapped_weight, wrapped_bias, EPS)
    wrapped_y.backward(dy)
    torch.cuda.synchronize()
    _check_close("autograd dx", wrapped_x.grad, expected_dx, case)
    _check_close("autograd dweight", wrapped_weight.grad, expected_dweight, case)
    _check_close("autograd dbias", wrapped_bias.grad, expected_dbias, case)

    return {
        "shape": [case.rows, case.cols],
        "dtype": case.dtype_name,
        "forward_max_abs": _max_abs(actual_y, expected_y),
        "dx_max_abs": _max_abs(actual_dx, expected_dx),
        "dweight_max_abs": _max_abs(actual_dweight, expected_dweight),
        "dbias_max_abs": _max_abs(actual_dbias, expected_dbias),
    }


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available; run this test on a GPU node.")
        return 0

    reports = [run_case(case) for case in CORRECTNESS_CASES]
    for report in reports:
        print(report)
    print(f"PASS: {len(reports)} randomized LayerNorm correctness cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
