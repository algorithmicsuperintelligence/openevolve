"""Naive decomposed Triton backward baseline for y = relu(x + bias).

This mirrors the LayerNorm sample's construction style: split the math into
small, inspectable Triton kernels instead of starting from a fused optimized
implementation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _bias_relu_backward_dx_kernel(
    dy_ptr,
    x_ptr,
    bias_ptr,
    dx_ptr,
    total: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    cols = offsets % n_cols

    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    active = (x + bias) > 0.0
    dx = tl.where(active, dy, 0.0)

    tl.store(dx_ptr + offsets, dx, mask=mask)


@triton.jit
def _bias_relu_backward_dbias_kernel(
    dx_ptr,
    dbias_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    col = tl.program_id(0)
    rows = tl.arange(0, BLOCK_ROWS)
    mask = rows < n_rows
    values = tl.load(dx_ptr + rows * n_cols + col, mask=mask, other=0.0).to(tl.float32)
    dbias = tl.sum(values, axis=0)

    tl.store(dbias_ptr + col, dbias)


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _dbias_launch_params(n_rows: int) -> tuple[int, int]:
    block_rows = _next_power_of_2(n_rows)
    num_warps = 4
    if block_rows >= 2048:
        num_warps = 8
    return block_rows, num_warps


def bias_relu_backward_naive_triton(
    dy: torch.Tensor,
    x: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (dx, dbias) for y = relu(x + bias).

    This baseline intentionally uses a simple two-kernel structure:
    one elementwise kernel for dx and one column-wise reduction for dbias.
    """
    if dy.shape != x.shape:
        raise ValueError(f"dy and x must have the same shape, got {dy.shape} and {x.shape}")
    if dy.ndim != 2:
        raise ValueError(f"expected 2D tensors, got shape {dy.shape}")
    if bias.ndim != 1 or bias.shape[0] != x.shape[1]:
        raise ValueError(f"bias shape {bias.shape} must match x cols {x.shape[1]}")
    if not dy.is_cuda or not x.is_cuda or not bias.is_cuda:
        raise ValueError("bias_relu_backward_naive_triton requires CUDA tensors")

    dy = dy.contiguous()
    x = x.contiguous()
    bias = bias.contiguous()
    dx = torch.empty_like(x)
    dbias = torch.empty_like(bias)

    n_rows, n_cols = x.shape
    total = x.numel()
    block_size = 256
    _bias_relu_backward_dx_kernel[(triton.cdiv(total, block_size),)](
        dy,
        x,
        bias,
        dx,
        total,
        n_cols,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )

    block_rows, num_warps = _dbias_launch_params(n_rows)
    _bias_relu_backward_dbias_kernel[(n_cols,)](
        dx,
        dbias,
        n_rows,
        n_cols,
        BLOCK_ROWS=block_rows,
        num_warps=num_warps,
    )
    return dx, dbias
