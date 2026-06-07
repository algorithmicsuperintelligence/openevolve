"""Autograd integration for the Triton LayerNorm backward sample."""

import torch

try:
    from backward_naive_triton import layernorm_backward_naive_triton
    from forward_triton import layernorm_forward_triton
except ImportError:  # pragma: no cover - supports package-style imports
    from .backward_naive_triton import layernorm_backward_naive_triton
    from .forward_triton import layernorm_forward_triton


class LayerNormTritonFunction(torch.autograd.Function):
    """Training-ready wrapper using Triton for forward and naive backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
        y = layernorm_forward_triton(x, weight, bias, eps)
        ctx.save_for_backward(x, weight, bias)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, weight, bias = ctx.saved_tensors
        dx, dweight, dbias = layernorm_backward_naive_triton(dy, x, weight, bias, ctx.eps)
        return dx, dweight, dbias, None


def layernorm_autograd_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Apply row-wise LayerNorm through a custom autograd.Function."""
    return LayerNormTritonFunction.apply(x, weight, bias, eps)
