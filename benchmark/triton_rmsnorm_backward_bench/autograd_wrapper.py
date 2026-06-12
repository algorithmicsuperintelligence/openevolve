"""Autograd integration for the Triton RMSNorm backward sample."""

import torch

try:
    from backward_naive_triton import rmsnorm_backward_naive_triton
    from forward_triton import rmsnorm_forward_triton
except ImportError:  # pragma: no cover - supports package-style imports
    from .backward_naive_triton import rmsnorm_backward_naive_triton
    from .forward_triton import rmsnorm_forward_triton


class RMSNormTritonFunction(torch.autograd.Function):
    """Training-ready wrapper using Triton for forward and naive backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float):
        y = rmsnorm_forward_triton(x, weight, eps)
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, weight = ctx.saved_tensors
        dx, dweight = rmsnorm_backward_naive_triton(dy, x, weight, ctx.eps)
        return dx, dweight, None


def rmsnorm_autograd_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Apply row-wise RMSNorm through a custom autograd.Function."""
    return RMSNormTritonFunction.apply(x, weight, eps)
