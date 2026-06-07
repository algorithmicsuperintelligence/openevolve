"""Randomized correctness checks for the bias + ReLU backward sample."""

from dataclasses import dataclass

import torch

from autograd_wrapper import bias_relu_autograd_triton
from backward_naive_triton import bias_relu_backward_naive_triton
from backward_ref import bias_relu_backward_ref
from forward_ref import bias_relu_forward_ref
from forward_triton import bias_relu_forward_triton


@dataclass(frozen=True)
class TestCase:
    rows: int
    cols: int
    dtype: torch.dtype
    atol: float
    rtol: float


CASES = [
    TestCase(17, 31, torch.float32, 1e-5, 1e-5),
    TestCase(128, 256, torch.float32, 1e-5, 1e-5),
    TestCase(129, 257, torch.float16, 2e-2, 2e-2),
    TestCase(512, 1024, torch.float16, 2e-2, 2e-2),
]


def _max_abs(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(torch.max(torch.abs(actual.float() - expected.float())).item())


def _check_close(name: str, actual: torch.Tensor, expected: torch.Tensor, case: TestCase) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"{name} shape mismatch: got {actual.shape}, expected {expected.shape}")
    if not torch.allclose(actual, expected, atol=case.atol, rtol=case.rtol):
        raise AssertionError(
            f"{name} mismatch for shape=({case.rows}, {case.cols}) dtype={case.dtype}: "
            f"max_abs={_max_abs(actual, expected):.6e}, atol={case.atol}, rtol={case.rtol}"
        )


def run_case(case: TestCase) -> dict[str, float | str | list[int]]:
    seed = case.rows * 100000 + case.cols
    torch.manual_seed(seed)
    x = torch.randn((case.rows, case.cols), device="cuda", dtype=case.dtype)
    bias = torch.randn((case.cols,), device="cuda", dtype=case.dtype)
    dy = torch.randn_like(x)

    expected_y = bias_relu_forward_ref(x, bias)
    actual_y = bias_relu_forward_triton(x, bias)
    torch.cuda.synchronize()
    _check_close("forward y", actual_y, expected_y, case)

    expected_dx, expected_dbias = bias_relu_backward_ref(dy, x, bias)
    actual_dx, actual_dbias = bias_relu_backward_naive_triton(dy, x, bias)
    torch.cuda.synchronize()
    _check_close("dx", actual_dx, expected_dx, case)
    _check_close("dbias", actual_dbias, expected_dbias, case)

    wrapped_x = x.detach().clone().requires_grad_(True)
    wrapped_bias = bias.detach().clone().requires_grad_(True)
    wrapped_y = bias_relu_autograd_triton(wrapped_x, wrapped_bias)
    wrapped_y.backward(dy)
    torch.cuda.synchronize()
    _check_close("autograd dx", wrapped_x.grad, expected_dx, case)
    _check_close("autograd dbias", wrapped_bias.grad, expected_dbias, case)

    return {
        "shape": [case.rows, case.cols],
        "dtype": str(case.dtype).replace("torch.", ""),
        "forward_max_abs": _max_abs(actual_y, expected_y),
        "dx_max_abs": _max_abs(actual_dx, expected_dx),
        "dbias_max_abs": _max_abs(actual_dbias, expected_dbias),
    }


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available; run this test on a GPU node.")
        return 0

    reports = [run_case(case) for case in CASES]
    for report in reports:
        print(report)
    print(f"PASS: {len(reports)} randomized correctness cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
