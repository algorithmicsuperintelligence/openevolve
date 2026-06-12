"""Autograd integration for the Triton Linear backward sample."""

import torch

try:
    from backward_naive_triton import linear_backward_naive_triton
    from forward_triton import linear_forward_triton
except ImportError:  # pragma: no cover - supports package-style imports
    from .backward_naive_triton import linear_backward_naive_triton
    from .forward_triton import linear_forward_triton


class LinearTritonFunction(torch.autograd.Function):
    """Training-ready wrapper using Triton for forward and naive backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        y = linear_forward_triton(x, weight, bias)
        ctx.save_for_backward(x, weight)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, weight = ctx.saved_tensors
        dx, dweight, dbias = linear_backward_naive_triton(dy, x, weight)
        return dx, dweight, dbias


def linear_autograd_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Apply a Linear layer through a custom autograd.Function."""
    return LinearTritonFunction.apply(x, weight, bias)
