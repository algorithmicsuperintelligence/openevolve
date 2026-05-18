"""Naive decomposed Triton backward baseline for row-wise LayerNorm.

This follows the coworker-proposed construction: split LayerNorm backward into
small pointwise and reduction kernels that are easy to inspect and verify.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _compute_xhat_kernel(
    x_ptr,
    xhat_ptr,
    rstd_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row_offsets = row * n_cols + offsets

    x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)
    xhat = centered * rstd

    tl.store(xhat_ptr + row_offsets, xhat, mask=mask)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _compute_one_kernel(
    dy_ptr,
    weight_ptr,
    one_ptr,
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
    one = dy * weight
    tl.store(one_ptr + offsets, one, mask=mask)


@triton.jit
def _mean_one_kernel(
    one_ptr,
    mean_one_ptr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row_offsets = row * n_cols + offsets

    one = tl.load(one_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    mean_one = tl.sum(one, axis=0) / n_cols
    tl.store(mean_one_ptr + row, mean_one)


@triton.jit
def _mean_xhat_one_kernel(
    xhat_ptr,
    one_ptr,
    mean_xhat_one_ptr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row_offsets = row * n_cols + offsets

    xhat = tl.load(xhat_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    one = tl.load(one_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    mean_xhat_one = tl.sum(xhat * one, axis=0) / n_cols
    tl.store(mean_xhat_one_ptr + row, mean_xhat_one)


@triton.jit
def _dx_kernel(
    one_ptr,
    xhat_ptr,
    rstd_ptr,
    mean_one_ptr,
    mean_xhat_one_ptr,
    dx_ptr,
    total: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    rows = offsets // n_cols

    one = tl.load(one_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    xhat = tl.load(xhat_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + rows, mask=mask, other=0.0).to(tl.float32)
    mean_one = tl.load(mean_one_ptr + rows, mask=mask, other=0.0).to(tl.float32)
    mean_xhat_one = tl.load(mean_xhat_one_ptr + rows, mask=mask, other=0.0).to(tl.float32)

    dx = (one - mean_one - xhat * mean_xhat_one) * rstd
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


@triton.jit
def _dbias_kernel(
    dy_ptr,
    dbias_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    col = tl.program_id(0)
    rows = tl.arange(0, BLOCK_ROWS)
    mask = rows < n_rows
    values = tl.load(dy_ptr + rows * n_cols + col, mask=mask, other=0.0).to(tl.float32)
    dbias = tl.sum(values, axis=0)
    tl.store(dbias_ptr + col, dbias)


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


def layernorm_backward_naive_triton(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (dx, dweight, dbias) for row-wise LayerNorm.

    The implementation intentionally launches seven small kernels:
    xhat/rstd, one=dy*weight, mean(one), mean(xhat*one), dx, dweight, and dbias.
    """
    if dy.shape != x.shape:
        raise ValueError(f"dy and x must have the same shape, got {dy.shape} and {x.shape}")
    if dy.ndim != 2:
        raise ValueError(f"expected 2D tensors, got shape {dy.shape}")
    if weight.ndim != 1 or bias.ndim != 1:
        raise ValueError("weight and bias must be 1D tensors")
    if weight.shape[0] != x.shape[1] or bias.shape[0] != x.shape[1]:
        raise ValueError("weight and bias length must match x hidden dimension")
    if not dy.is_cuda or not x.is_cuda or not weight.is_cuda or not bias.is_cuda:
        raise ValueError("layernorm_backward_naive_triton requires CUDA tensors")

    dy = dy.contiguous()
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    n_rows, n_cols = x.shape
    total = x.numel()
    dx = torch.empty_like(x)
    dweight = torch.empty_like(weight)
    dbias = torch.empty_like(bias)
    xhat = torch.empty_like(x, dtype=torch.float32)
    one = torch.empty_like(x, dtype=torch.float32)
    rstd = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
    mean_one = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
    mean_xhat_one = torch.empty((n_rows,), device=x.device, dtype=torch.float32)

    row_block, row_warps, row_stages = _row_launch_params(n_cols)
    _compute_xhat_kernel[(n_rows,)](
        x,
        xhat,
        rstd,
        n_cols,
        eps,
        BLOCK_SIZE=row_block,
        num_warps=row_warps,
        num_stages=row_stages,
    )

    point_block = 256
    _compute_one_kernel[(triton.cdiv(total, point_block),)](
        dy,
        weight,
        one,
        total,
        n_cols,
        BLOCK_SIZE=point_block,
        num_warps=4,
    )

    _mean_one_kernel[(n_rows,)](
        one,
        mean_one,
        n_cols,
        BLOCK_SIZE=row_block,
        num_warps=row_warps,
        num_stages=row_stages,
    )
    _mean_xhat_one_kernel[(n_rows,)](
        xhat,
        one,
        mean_xhat_one,
        n_cols,
        BLOCK_SIZE=row_block,
        num_warps=row_warps,
        num_stages=row_stages,
    )

    _dx_kernel[(triton.cdiv(total, point_block),)](
        one,
        xhat,
        rstd,
        mean_one,
        mean_xhat_one,
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
    _dbias_kernel[(n_cols,)](
        dy,
        dbias,
        n_rows,
        n_cols,
        BLOCK_ROWS=col_block,
        num_warps=col_warps,
    )

    return dx, dweight, dbias
