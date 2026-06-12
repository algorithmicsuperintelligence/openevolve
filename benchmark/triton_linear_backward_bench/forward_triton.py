"""Simple tiled Triton forward kernel for a Linear layer: y = x @ weight.T + bias."""

import torch
import triton
import triton.language as tl


@triton.jit
def _linear_forward_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # x[m, k] and weight[n, k] -> contract over k; B tile is W[n, k] indexed as [k, n].
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k = 0
    while k < K:
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] + k < K)
        w_mask = (offs_k[:, None] + k < K) & (offs_n[None, :] < N)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x_tile, w_tile)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
        k += BLOCK_K

    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    y = acc.to(y_ptr.dtype.element_ty)
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, y, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def linear_forward_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Compute y = x @ weight.T + bias for contiguous 2D CUDA tensors."""
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("x and weight must be 2D tensors")
    if weight.shape[1] != x.shape[1]:
        raise ValueError("weight.shape[1] must match x.shape[1] (K)")
    if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
        raise ValueError("bias must be a 1D tensor of length N (= weight.shape[0])")
    if not x.is_cuda or not weight.is_cuda or not bias.is_cuda:
        raise ValueError("linear_forward_triton requires CUDA tensors")

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    M, K = x.shape
    N = weight.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _linear_forward_kernel[grid](
        x,
        weight,
        bias,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    return y
