"""Randomized correctness checks for the EvoformerAttention backward sample.

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

from autograd_wrapper import evoattention_autograd
from backward_atenir import evoattention_backward_triton as atenir_backward
from backward_naive import evoattention_backward_naive
from backward_naive_triton import evoattention_backward_naive_triton
from backward_ref import evoattention_backward_ref
from task_spec import CORRECTNESS_CASES, TestCase, atol, make_inputs, seed_for_case

BACKENDS = [
    ("backward_naive (torch)", evoattention_backward_naive),
    ("backward_naive_triton", evoattention_backward_naive_triton),
    ("backward_atenir", atenir_backward),
]


def _max_abs(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(torch.max(torch.abs(actual.float() - expected.float())).item())


def _check_close(name: str, actual: torch.Tensor, expected: torch.Tensor, case: TestCase) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"{name} shape mismatch: got {tuple(actual.shape)}, expected {tuple(expected.shape)}")
    output_key = name.split()[-1]
    tol = atol(case, output_key)
    if not torch.allclose(actual, expected, atol=tol, rtol=case.rtol_value):
        raise AssertionError(
            f"{name} mismatch for (B,S,H,R,D)="
            f"({case.b},{case.n_seq},{case.head},{case.n_res},{case.dim}) dtype={case.dtype_name}: "
            f"max_abs={_max_abs(actual, expected):.6e}, atol={tol}, rtol={case.rtol_value}"
        )


def run_case(case: TestCase) -> dict:
    torch.manual_seed(seed_for_case(case))
    do, q, k, v, res_mask, pair_bias = make_inputs(torch, case)

    exp = evoattention_backward_ref(do, q, k, v, res_mask, pair_bias)

    worst = {}
    for label, fn in BACKENDS:
        act = fn(do, q, k, v, res_mask, pair_bias)
        torch.cuda.synchronize()
        for name, a, e in zip(("dq", "dk", "dv", "d_pair_bias"), act, exp):
            _check_close(f"{label} {name}", a, e, case)
        worst[label] = max(_max_abs(a, e) for a, e in zip(act, exp))

    # autograd.Function integration (uses the torch naive backward).
    wq = q.detach().clone().requires_grad_(True)
    wk = k.detach().clone().requires_grad_(True)
    wv = v.detach().clone().requires_grad_(True)
    wpb = pair_bias.detach().clone().requires_grad_(True)
    out = evoattention_autograd(wq, wk, wv, res_mask, wpb)
    out.backward(do)
    torch.cuda.synchronize()
    _check_close("autograd dq", wq.grad, exp[0], case)
    _check_close("autograd dk", wk.grad, exp[1], case)
    _check_close("autograd dv", wv.grad, exp[2], case)
    _check_close("autograd d_pair_bias", wpb.grad, exp[3], case)

    return {"shape": [case.b, case.n_seq, case.head, case.n_res, case.dim], "dtype": case.dtype_name, "worst_max_abs": worst}


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
