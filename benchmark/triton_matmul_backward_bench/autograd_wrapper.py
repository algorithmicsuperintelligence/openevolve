"""Autograd integration for the Triton matmul backward sample."""

import torch

try:
    from backward_naive_triton import matmul_backward_naive_triton
    from forward_triton import matmul_forward_triton
except ImportError:  # pragma: no cover - supports package-style imports
    from .backward_naive_triton import matmul_backward_naive_triton
    from .forward_triton import matmul_forward_triton


class MatmulTritonFunction(torch.autograd.Function):
    """Training-ready wrapper using Triton for forward and naive backward."""

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor):
        c = matmul_forward_triton(a, b)
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    def backward(ctx, dc: torch.Tensor):
        a, b = ctx.saved_tensors
        da, db = matmul_backward_naive_triton(dc, a, b)
        return da, db


def matmul_autograd_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Apply c = a @ b through a custom autograd.Function."""
    return MatmulTritonFunction.apply(a, b)
