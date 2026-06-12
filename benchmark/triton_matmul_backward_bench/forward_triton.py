"""Simple tiled Triton forward kernel for c = a @ b.

Reuses the generic strided matmul launcher from backward_naive_triton.
"""

import torch

try:
    from backward_naive_triton import _mm
except ImportError:  # pragma: no cover - supports package-style imports
    from .backward_naive_triton import _mm


def matmul_forward_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute c = a @ b for contiguous 2D CUDA tensors."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("a.shape[1] must match b.shape[0] (K)")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("matmul_forward_triton requires CUDA tensors")

    a = a.contiguous()
    b = b.contiguous()
    M, K = a.shape
    N = b.shape[1]
    # c[M, N] = a[M, K] @ b[K, N]   (out rows M, cols N, contraction K)
    return _mm(a, b, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), a.dtype)
