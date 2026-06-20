"""Randomized correctness checks for the fused LayerNorm -> Linear backward sample.

Verifies every backward backend (PyTorch naive, naive Triton, AtenIR lowering) and
the autograd.Function integration against the PyTorch autograd oracle.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from autograd_wrapper import layernorm_linear_autograd
from backward_atenir import layernorm_linear_backward_triton as atenir_backward
from backward_naive import layernorm_linear_backward_naive
from backward_naive_triton import layernorm_linear_backward_naive_triton
from backward_ref import layernorm_linear_backward_ref
from task_spec import CORRECTNESS_CASES, TestCase, atol, make_inputs, seed_for_case

# (label, callable) backward backends sharing the bench's frozen signature.
BACKENDS = [
    ("backward_naive (torch)", layernorm_linear_backward_naive),
    ("backward_naive_triton", layernorm_linear_backward_naive_triton),
    ("backward_atenir", atenir_backward),
]


def _max_abs(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(torch.max(torch.abs(actual.float() - expected.float())).item())


def _check_close(name: str, actual: torch.Tensor, expected: torch.Tensor, case: TestCase) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"{name} shape mismatch: got {tuple(actual.shape)}, expected {tuple(expected.shape)}")
    output_key = name.split()[-1]  # strip backend / "autograd" prefix
    tol = atol(case, output_key)
    if not torch.allclose(actual, expected, atol=tol, rtol=case.rtol_value):
        raise AssertionError(
            f"{name} mismatch for (M,K,N)=({case.m},{case.k},{case.n}) dtype={case.dtype_name}: "
            f"max_abs={_max_abs(actual, expected):.6e}, atol={tol}, rtol={case.rtol_value}"
        )


def run_case(case: TestCase) -> dict:
    torch.manual_seed(seed_for_case(case))
    dout, x, weight, bias, linear_weight, eps = make_inputs(torch, case)

    exp_dx, exp_dlw, exp_dg, exp_db = layernorm_linear_backward_ref(
        dout, x, weight, bias, linear_weight, eps
    )

    worst = {}
    for label, fn in BACKENDS:
        act_dx, act_dlw, act_dg, act_db = fn(dout, x, weight, bias, linear_weight, eps)
        torch.cuda.synchronize()
        _check_close(f"{label} dx", act_dx, exp_dx, case)
        _check_close(f"{label} dlinear_weight", act_dlw, exp_dlw, case)
        _check_close(f"{label} dweight", act_dg, exp_dg, case)
        _check_close(f"{label} dbias", act_db, exp_db, case)
        worst[label] = max(
            _max_abs(act_dx, exp_dx), _max_abs(act_dlw, exp_dlw),
            _max_abs(act_dg, exp_dg), _max_abs(act_db, exp_db),
        )

    # autograd.Function integration (uses the torch naive backward).
    wx = x.detach().clone().requires_grad_(True)
    ww = weight.detach().clone().requires_grad_(True)
    wb = bias.detach().clone().requires_grad_(True)
    wlw = linear_weight.detach().clone().requires_grad_(True)
    out = layernorm_linear_autograd(wx, ww, wb, wlw, eps)
    out.backward(dout)
    torch.cuda.synchronize()
    _check_close("autograd dx", wx.grad, exp_dx, case)
    _check_close("autograd dlinear_weight", wlw.grad, exp_dlw, case)
    _check_close("autograd dweight", ww.grad, exp_dg, case)
    _check_close("autograd dbias", wb.grad, exp_db, case)

    return {"shape": [case.m, case.k, case.n], "dtype": case.dtype_name, "worst_max_abs": worst}


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available; run this test on a GPU node.")
        return 0

    for case in CORRECTNESS_CASES:
        print(run_case(case))
    print(f"PASS: {len(CORRECTNESS_CASES)} cases x {len(BACKENDS)} backends + autograd integration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
