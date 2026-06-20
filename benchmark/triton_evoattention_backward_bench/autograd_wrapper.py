"""Autograd integration for the EvoformerAttention backward sample.

Wraps the reference forward and the naive backward into a
``torch.autograd.Function`` so the example can be exercised as a training-ready
module and checked end to end against PyTorch autograd in test_correctness.py.
"""

import torch

try:
    from backward_naive import evoattention_backward_naive
    from forward_ref import evoattention_forward_ref
except ImportError:  # pragma: no cover - supports package-style imports
    from .backward_naive import evoattention_backward_naive
    from .forward_ref import evoattention_forward_ref


class EvoAttentionFunction(torch.autograd.Function):
    """Reference forward + naive (materialized) backward, training-ready."""

    @staticmethod
    def forward(ctx, q, k, v, res_mask, pair_bias):
        out = evoattention_forward_ref(q, k, v, res_mask, pair_bias)
        ctx.save_for_backward(q, k, v, res_mask, pair_bias)
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, res_mask, pair_bias = ctx.saved_tensors
        dq, dk, dv, d_pair_bias = evoattention_backward_naive(do, q, k, v, res_mask, pair_bias)
        # res_mask is additive and carries no gradient.
        return dq, dk, dv, None, d_pair_bias


def evoattention_autograd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    res_mask: torch.Tensor,
    pair_bias: torch.Tensor,
) -> torch.Tensor:
    """Apply EvoformerAttention through a custom autograd.Function."""
    return EvoAttentionFunction.apply(q, k, v, res_mask, pair_bias)
