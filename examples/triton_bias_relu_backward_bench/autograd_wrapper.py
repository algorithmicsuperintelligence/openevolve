"""Autograd integration for the Triton bias + ReLU sample task."""

import torch

try:
    from backward_naive_triton import bias_relu_backward_naive_triton
    from forward_triton import bias_relu_forward_triton
except ImportError:  # pragma: no cover - supports package-style imports
    from .backward_naive_triton import bias_relu_backward_naive_triton
    from .forward_triton import bias_relu_forward_triton


class BiasReLUTritonFunction(torch.autograd.Function):
    """Training-ready wrapper that uses Triton for both forward and backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        y = bias_relu_forward_triton(x, bias)
        ctx.save_for_backward(x, bias)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, bias = ctx.saved_tensors
        dx, dbias = bias_relu_backward_naive_triton(dy, x, bias)
        return dx, dbias


def bias_relu_autograd_triton(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Apply y = relu(x + bias) through a custom autograd.Function."""
    return BiasReLUTritonFunction.apply(x, bias)
