"""Naive tiled Triton backward baseline for a plain matmul c = a @ b.

Gradients:
    da = dc @ b.T     [M, K]   (contract over N)
    db = a.T @ dc     [K, N]   (contract over M)

Both gradients use one generic tiled matmul kernel; the transposes are handled
purely through strides. fp32 accumulation, fixed 64x64x32 tiles, no autotuning
or fusion -- readable and verifiable, not fast.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _mm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Generic tiled C[M, N] = A[M, K] @ B[K, N], fp32 accumulation, strided."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k = 0
    while k < K:
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] + k < K)
        b_mask = (offs_k[:, None] + k < K) & (offs_n[None, :] < N)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a_tile, b_tile)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _mm(a, b, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, out_dtype):
    """Launch the generic matmul; returns C[M, N] = A @ B given explicit strides."""
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    _mm_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))](
        a,
        b,
        c,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    return c


def matmul_backward_naive_triton(
    dc: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (da, db) for c = a @ b."""
    if dc.ndim != 2 or a.ndim != 2 or b.ndim != 2:
        raise ValueError("dc, a, b must be 2D tensors")
    M, K = a.shape
    if b.shape[0] != K:
        raise ValueError("a.shape[1] must match b.shape[0] (K)")
    N = b.shape[1]
    if dc.shape != (M, N):
        raise ValueError(f"dc must have shape {(M, N)}, got {tuple(dc.shape)}")
    if not dc.is_cuda or not a.is_cuda or not b.is_cuda:
        raise ValueError("matmul_backward_naive_triton requires CUDA tensors")

    dc = dc.contiguous()
    a = a.contiguous()
    b = b.contiguous()

    # da[M, K] = dc[M, N] @ b[K, N].T   (out rows M, cols K, contraction N)
    da = _mm(dc, b, M, K, N, dc.stride(0), dc.stride(1), b.stride(1), b.stride(0), a.dtype)
    # db[K, N] = a[M, K].T @ dc[M, N]   (out rows K, cols N, contraction M)
    db = _mm(a, dc, K, N, M, a.stride(1), a.stride(0), dc.stride(0), dc.stride(1), b.dtype)
    return da, db
