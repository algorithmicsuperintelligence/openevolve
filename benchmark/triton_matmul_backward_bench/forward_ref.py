"""PyTorch reference for a plain matmul: c = a @ b."""

import torch


def matmul_forward_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute c = a @ b for a [M, K], b [K, N] -> c [M, N]."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("a.shape[1] must match b.shape[0] (K)")
    return a @ b
