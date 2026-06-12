"""Naive decomposed Triton backward baseline for row-wise RMSNorm.

This mirrors the LayerNorm construction: split RMSNorm backward into small
pointwise and reduction kernels that are easy to inspect and verify. RMSNorm has
no mean-centering and no bias, so the decomposition needs five kernels:
xhat/rrms, g=dy*weight, mean(g*xhat), dx, and dweight.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _compute_xhat_kernel(
    x_ptr,
    xhat_ptr,
    rrms_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row_offsets = row * n_cols + offsets

    x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / n_cols
    rrms = tl.rsqrt(ms + eps)
    xhat = x * rrms

    tl.store(xhat_ptr + row_offsets, xhat, mask=mask)
    tl.store(rrms_ptr + row, rrms)


@triton.jit
def _compute_g_kernel(
    dy_ptr,
    weight_ptr,
    g_ptr,
    total: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    cols = offsets % n_cols

    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    g = dy * weight
    tl.store(g_ptr + offsets, g, mask=mask)


@triton.jit
def _mean_g_xhat_kernel(
    g_ptr,
    xhat_ptr,
    mean_g_xhat_ptr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row_offsets = row * n_cols + offsets

    g = tl.load(g_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    xhat = tl.load(xhat_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    mean_g_xhat = tl.sum(g * xhat, axis=0) / n_cols
    tl.store(mean_g_xhat_ptr + row, mean_g_xhat)


@triton.jit
def _dx_kernel(
    g_ptr,
    xhat_ptr,
    rrms_ptr,
    mean_g_xhat_ptr,
    dx_ptr,
    total: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    rows = offsets // n_cols

    g = tl.load(g_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    xhat = tl.load(xhat_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    rrms = tl.load(rrms_ptr + rows, mask=mask, other=0.0).to(tl.float32)
    mean_g_xhat = tl.load(mean_g_xhat_ptr + rows, mask=mask, other=0.0).to(tl.float32)

    dx = (g - xhat * mean_g_xhat) * rrms
    tl.store(dx_ptr + offsets, dx, mask=mask)


@triton.jit
def _dweight_kernel(
    dy_ptr,
    xhat_ptr,
    dweight_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    col = tl.program_id(0)
    rows = tl.arange(0, BLOCK_ROWS)
    mask = rows < n_rows
    offsets = rows * n_cols + col

    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    xhat = tl.load(xhat_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dweight = tl.sum(dy * xhat, axis=0)
    tl.store(dweight_ptr + col, dweight)


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _row_launch_params(n_cols: int) -> tuple[int, int, int]:
    block_size = _next_power_of_2(n_cols)
    num_warps = 4
    num_stages = 4
    if block_size >= 2048:
        num_warps = 8
    if block_size >= 4096:
        num_stages = 3
    return block_size, num_warps, num_stages


def _col_reduce_launch_params(n_rows: int) -> tuple[int, int]:
    block_rows = _next_power_of_2(n_rows)
    num_warps = 4
    if block_rows >= 2048:
        num_warps = 8
    return block_rows, num_warps


def rmsnorm_backward_naive_triton(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (dx, dweight) for row-wise RMSNorm.

    The implementation intentionally launches five small kernels:
    xhat/rrms, g=dy*weight, mean(g*xhat), dx, and dweight.
    """
    if dy.shape != x.shape:
        raise ValueError(f"dy and x must have the same shape, got {dy.shape} and {x.shape}")
    if dy.ndim != 2:
        raise ValueError(f"expected 2D tensors, got shape {dy.shape}")
    if weight.ndim != 1:
        raise ValueError("weight must be a 1D tensor")
    if weight.shape[0] != x.shape[1]:
        raise ValueError("weight length must match x hidden dimension")
    if not dy.is_cuda or not x.is_cuda or not weight.is_cuda:
        raise ValueError("rmsnorm_backward_naive_triton requires CUDA tensors")

    dy = dy.contiguous()
    x = x.contiguous()
    weight = weight.contiguous()

    n_rows, n_cols = x.shape
    total = x.numel()
    dx = torch.empty_like(x)
    dweight = torch.empty_like(weight)
    xhat = torch.empty_like(x, dtype=torch.float32)
    g = torch.empty_like(x, dtype=torch.float32)
    rrms = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
    mean_g_xhat = torch.empty((n_rows,), device=x.device, dtype=torch.float32)

    row_block, row_warps, row_stages = _row_launch_params(n_cols)
    _compute_xhat_kernel[(n_rows,)](
        x,
        xhat,
        rrms,
        n_cols,
        eps,
        BLOCK_SIZE=row_block,
        num_warps=row_warps,
        num_stages=row_stages,
    )

    point_block = 256
    _compute_g_kernel[(triton.cdiv(total, point_block),)](
        dy,
        weight,
        g,
        total,
        n_cols,
        BLOCK_SIZE=point_block,
        num_warps=4,
    )

    _mean_g_xhat_kernel[(n_rows,)](
        g,
        xhat,
        mean_g_xhat,
        n_cols,
        BLOCK_SIZE=row_block,
        num_warps=row_warps,
        num_stages=row_stages,
    )

    _dx_kernel[(triton.cdiv(total, point_block),)](
        g,
        xhat,
        rrms,
        mean_g_xhat,
        dx,
        total,
        n_cols,
        BLOCK_SIZE=point_block,
        num_warps=4,
    )

    col_block, col_warps = _col_reduce_launch_params(n_rows)
    _dweight_kernel[(n_cols,)](
        dy,
        xhat,
        dweight,
        n_rows,
        n_cols,
        BLOCK_ROWS=col_block,
        num_warps=col_warps,
    )

    return dx, dweight
