"""Naive tiled Triton backward baseline for a Linear layer.

For y = x @ weight.T + bias the gradients are:
    dx      = dy @ weight       [M, K]   (contract over N)
    dweight = dy.T @ x          [N, K]   (contract over M)
    dbias   = sum(dy, dim=0)    [N]

This baseline launches three deliberately simple kernels: a generic tiled
matmul used for both dx and dweight (the transpose is handled purely through
strides), plus a column-reduction for dbias. Accumulation is in float32. It
favors readability/verifiability over speed (no autotuning, fixed tiles, no
fusion of dbias into the dweight pass).
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
    """Generic tiled C[M, N] = A[M, K] @ B[K, N] with fp32 accumulation.

    Strides are passed explicitly so a transposed operand (e.g. dy.T) is handled
    without a physical transpose.
    """
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


@triton.jit
def _dbias_kernel(
    dy_ptr,
    dbias_ptr,
    M,
    N,
    stride_dym,
    stride_dyn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """dbias[n] = sum_m dy[m, n], looping over rows in BLOCK_M chunks."""
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    m = 0
    while m < M:
        offs_m = m + tl.arange(0, BLOCK_M)
        ptrs = dy_ptr + offs_m[:, None] * stride_dym + offs_n[None, :] * stride_dyn
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(vals, axis=0)
        m += BLOCK_M
    tl.store(dbias_ptr + offs_n, acc, mask=offs_n < N)


def linear_backward_naive_triton(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (dx, dweight, dbias) for y = x @ weight.T + bias."""
    if dy.ndim != 2 or x.ndim != 2 or weight.ndim != 2:
        raise ValueError("dy, x, weight must be 2D tensors")
    M, K = x.shape
    N = weight.shape[0]
    if dy.shape != (M, N):
        raise ValueError(f"dy must have shape {(M, N)}, got {tuple(dy.shape)}")
    if weight.shape[1] != K:
        raise ValueError("weight.shape[1] must match x.shape[1] (K)")
    if not dy.is_cuda or not x.is_cuda or not weight.is_cuda:
        raise ValueError("linear_backward_naive_triton requires CUDA tensors")

    dy = dy.contiguous()
    x = x.contiguous()
    weight = weight.contiguous()

    dx = torch.empty((M, K), device=x.device, dtype=x.dtype)
    dweight = torch.empty((N, K), device=weight.device, dtype=weight.dtype)
    dbias = torch.empty((N,), device=dy.device, dtype=dy.dtype)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32

    # dx[M, K] = dy[M, N] @ weight[N, K]  (contract over N)
    _mm_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))](
        dy,
        weight,
        dx,
        M,
        K,
        N,
        dy.stride(0),
        dy.stride(1),
        weight.stride(0),
        weight.stride(1),
        dx.stride(0),
        dx.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )

    # dweight[N, K] = dy.T[N, M] @ x[M, K]  (contract over M)
    # A = dy.T: row index n -> dy.stride(1), contraction m -> dy.stride(0)
    _mm_kernel[(triton.cdiv(N, BLOCK_M), triton.cdiv(K, BLOCK_N))](
        dy,
        x,
        dweight,
        N,
        K,
        M,
        dy.stride(1),
        dy.stride(0),
        x.stride(0),
        x.stride(1),
        dweight.stride(0),
        dweight.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )

    # dbias[N] = sum_m dy[m, n]
    _dbias_kernel[(triton.cdiv(N, 128),)](
        dy,
        dbias,
        M,
        N,
        dy.stride(0),
        dy.stride(1),
        BLOCK_M=64,
        BLOCK_N=128,
    )

    return dx, dweight, dbias
