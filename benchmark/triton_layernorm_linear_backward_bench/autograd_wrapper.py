"""Autograd integration for the fused LayerNorm -> Linear backward sample.

Wraps the reference forward and the naive backward into a
``torch.autograd.Function`` so the example can be exercised as a training-ready
module and checked end to end against PyTorch autograd in test_correctness.py.
"""

import torch

try:
    from backward_naive import layernorm_linear_backward_naive
    from forward_ref import layernorm_linear_forward_ref
except ImportError:  # pragma: no cover - supports package-style imports
    from .backward_naive import layernorm_linear_backward_naive
    from .forward_ref import layernorm_linear_forward_ref


class LayernormLinearFunction(torch.autograd.Function):
    """Reference forward + naive backward, training-ready."""

    @staticmethod
    def forward(ctx, x, weight, bias, linear_weight, eps):
        out = layernorm_linear_forward_ref(x, weight, bias, linear_weight, eps)
        ctx.save_for_backward(x, weight, bias, linear_weight)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, dout):
        x, weight, bias, linear_weight = ctx.saved_tensors
        dx, dlw, dweight, dbias = layernorm_linear_backward_naive(
            dout, x, weight, bias, linear_weight, ctx.eps
        )
        # eps is a non-tensor argument -> no gradient.
        return dx, dweight, dbias, dlw, None


def layernorm_linear_autograd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    linear_weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Apply LayerNorm -> Linear through a custom autograd.Function."""
    return LayernormLinearFunction.apply(x, weight, bias, linear_weight, eps)
