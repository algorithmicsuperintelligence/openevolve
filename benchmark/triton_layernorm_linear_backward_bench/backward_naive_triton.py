"""Naive (unfused) Triton backward baseline for the fused LayerNorm -> Linear op.

This is a deliberately simple, easy-to-verify hand-written Triton baseline. It
mirrors MegaFold's math (arXiv:2506.20686) but keeps every step in its own small
kernel instead of fusing the LayerNorm into the matmul epilogue:

    1. _stats_kernel        : per-row mean/rstd, and materialize x_hat (fp32) and y_hat
    2. _mm_kernel (dy_hat)  : dy_hat        = dout @ B^T          (contract over N)
    3. _mm_kernel (dB)      : dlinear_weight = y_hat^T @ dout      (contract over M)
    4. _dwdb_kernel         : dweight = sum(dy_hat*x_hat, 0), dbias = sum(dy_hat, 0)
    5. _dx_kernel           : dx = (wdy - x_hat*mean(x_hat*wdy) - mean(wdy)) * rstd

Accumulation is in float32; tiles are fixed (no autotuning, no fusion). The
fused, faster version is the OpenEvolve evolve target.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _stats_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    mean_ptr,
    rstd_ptr,
    xhat_ptr,
    yhat_ptr,
    K: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Per-row LayerNorm statistics; store x_hat (fp32) and y_hat (input dtype)."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_K)
    mask = cols < K
    off = row * K + cols

    x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / K
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / K
    rstd = tl.rsqrt(var + eps)
    xhat = xc * rstd

    gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    yhat = xhat * gamma + beta

    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)
    tl.store(xhat_ptr + off, xhat, mask=mask)
    tl.store(yhat_ptr + off, yhat.to(yhat_ptr.dtype.element_ty), mask=mask)


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

    Strides are passed explicitly so a transposed operand is handled without a
    physical transpose.
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
def _dwdb_kernel(
    dyhat_ptr,
    xhat_ptr,
    dweight_ptr,
    dbias_ptr,
    M,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Column reductions: dweight[k]=sum_m dyhat*xhat, dbias[k]=sum_m dyhat."""
    pid_k = tl.program_id(0)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    dw = tl.zeros((BLOCK_K,), dtype=tl.float32)
    db = tl.zeros((BLOCK_K,), dtype=tl.float32)
    m = 0
    while m < M:
        offs_m = m + tl.arange(0, BLOCK_M)
        ptrs = offs_m[:, None] * K + offs_k[None, :]
        mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        dyhat = tl.load(dyhat_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
        xhat = tl.load(xhat_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
        dw += tl.sum(dyhat * xhat, axis=0)
        db += tl.sum(dyhat, axis=0)
        m += BLOCK_M
    tl.store(dweight_ptr + offs_k, dw, mask=offs_k < K)
    tl.store(dbias_ptr + offs_k, db, mask=offs_k < K)


@triton.jit
def _dx_kernel(
    dyhat_ptr,
    xhat_ptr,
    gamma_ptr,
    rstd_ptr,
    dx_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Per-row LayerNorm input backward: dx = (wdy - xhat*c1 - c2) * rstd."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_K)
    mask = cols < K
    off = row * K + cols

    dyhat = tl.load(dyhat_ptr + off, mask=mask, other=0.0).to(tl.float32)
    xhat = tl.load(xhat_ptr + off, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)

    wdy = dyhat * gamma
    c1 = tl.sum(tl.where(mask, xhat * wdy, 0.0), axis=0) / K
    c2 = tl.sum(tl.where(mask, wdy, 0.0), axis=0) / K
    dx = (wdy - (xhat * c1 + c2)) * rstd
    tl.store(dx_ptr + off, dx.to(dx_ptr.dtype.element_ty), mask=mask)


def _next_pow2(v: int) -> int:
    return 1 if v <= 1 else 1 << (v - 1).bit_length()


def layernorm_linear_backward_naive_triton(
    dout: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    linear_weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (dx, dlinear_weight, dweight, dbias) for LayerNorm -> Linear."""
    if x.ndim != 2:
        raise ValueError(f"expected 2D x [M, K], got {tuple(x.shape)}")
    if not (x.is_cuda and dout.is_cuda and linear_weight.is_cuda):
        raise ValueError("layernorm_linear_backward_naive_triton requires CUDA tensors")

    x = x.contiguous()
    dout = dout.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    linear_weight = linear_weight.contiguous()  # [K, N]

    M, K = x.shape
    N = linear_weight.shape[1]

    mean = torch.empty((M,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((M,), device=x.device, dtype=torch.float32)
    x_hat = torch.empty((M, K), device=x.device, dtype=torch.float32)
    y_hat = torch.empty((M, K), device=x.device, dtype=x.dtype)

    block_k_row = _next_pow2(K)
    _stats_kernel[(M,)](
        x, weight, bias, mean, rstd, x_hat, y_hat,
        K=K, eps=eps, BLOCK_K=block_k_row,
        num_warps=8 if block_k_row >= 2048 else 4,
    )

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32

    # dy_hat[M, K] = dout[M, N] @ B^T[N, K]  (contract over N), fp32 output.
    dy_hat = torch.empty((M, K), device=x.device, dtype=torch.float32)
    _mm_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))](
        dout, linear_weight, dy_hat,
        M, K, N,
        dout.stride(0), dout.stride(1),
        linear_weight.stride(1), linear_weight.stride(0),  # B^T: contract n, output k
        dy_hat.stride(0), dy_hat.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4, num_stages=3,
    )

    # dlinear_weight[K, N] = y_hat^T[K, M] @ dout[M, N]  (contract over M).
    dlinear_weight = torch.empty((K, N), device=x.device, dtype=torch.float32)
    _mm_kernel[(triton.cdiv(K, BLOCK_M), triton.cdiv(N, BLOCK_N))](
        y_hat, dout, dlinear_weight,
        K, N, M,
        y_hat.stride(1), y_hat.stride(0),  # y_hat^T: row k, contract m
        dout.stride(0), dout.stride(1),
        dlinear_weight.stride(0), dlinear_weight.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4, num_stages=3,
    )

    # dweight, dbias: column reductions over M.
    dweight = torch.empty((K,), device=x.device, dtype=torch.float32)
    dbias = torch.empty((K,), device=x.device, dtype=torch.float32)
    BLOCK_K_RED = 64
    _dwdb_kernel[(triton.cdiv(K, BLOCK_K_RED),)](
        dy_hat, x_hat, dweight, dbias,
        M, K, BLOCK_M=64, BLOCK_K=BLOCK_K_RED,
    )

    # dx: per-row LayerNorm input backward.
    dx = torch.empty((M, K), device=x.device, dtype=x.dtype)
    _dx_kernel[(M,)](
        dy_hat, x_hat, weight, rstd, dx,
        K=K, BLOCK_K=block_k_row,
        num_warps=8 if block_k_row >= 2048 else 4,
    )

    return (
        dx,
        dlinear_weight.to(linear_weight.dtype),
        dweight.to(weight.dtype),
        dbias.to(bias.dtype),
    )
