"""Triton bias + ReLU backward optimization target for OpenEvolve.

The fixed public API is ``bias_relu_backward_triton(dy, x, bias)``. OpenEvolve
should only modify the code inside the EVOLVE-BLOCK. The seed implementation is
a deliberately naive decomposed baseline: one pointwise kernel computes dx, then
one column-reduction kernel computes dbias.
"""

import torch
import triton
import triton.language as tl


# EVOLVE-BLOCK-START
@triton.jit
def _bias_relu_backward_fused_kernel(dy_ptr, x_ptr, bias_ptr, dx_ptr, dbias_ptr,
                                     n_rows: tl.constexpr, n_cols: tl.constexpr,
                                     BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    rows = tl.arange(0, BM)
    cols = pid * BN + tl.arange(0, BN)
    m = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)
    offs = rows[:, None] * n_cols + cols[None, :]
    b = tl.load(bias_ptr + cols, mask=cols < n_cols, other=0.0)
    dy = tl.load(dy_ptr + offs, mask=m, other=0.0)
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    dxf = tl.where((x + b[None, :]) > 0, dy.to(tl.float32), 0.0)
    tl.store(dx_ptr + offs, dxf, mask=m)
    tl.store(dbias_ptr + cols, tl.sum(dxf, axis=0), mask=cols < n_cols)


@triton.jit
def _bias_relu_backward_dx_kernel(dy_ptr, x_ptr, bias_ptr, dx_ptr, total, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total
    cols = offs % n_cols
    dy = tl.load(dy_ptr + offs, mask=mask, other=0.0)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    tl.store(dx_ptr + offs, tl.where((x + b) > 0, dy, 0), mask=mask)


@triton.jit
def _bias_relu_backward_dbias_kernel(dx_ptr, dbias_ptr, n_rows, n_cols, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for r in range(0, n_rows, BLOCK_M):
        rr = r + rows
        m = (rr[:, None] < n_rows) & (cols[None, :] < n_cols)
        v = tl.load(dx_ptr + rr[:, None] * n_cols + cols[None, :], mask=m, other=0.0)
        acc += tl.sum(v.to(tl.float32), axis=0)
    tl.store(dbias_ptr + cols, acc, mask=cols < n_cols)


def _fused_launch_params(n_rows: int, n_cols: int) -> tuple[int, int, int]:
    bm = 32 if n_rows <= 32 else 64 if n_rows <= 64 else 128 if n_rows <= 128 else 256 if n_rows <= 256 else 512 if n_rows <= 512 else 1024
    bn = 32 if n_cols >= 32 else 16
    return bm, bn, 8 if bm >= 512 else 4


def _dbias_launch_params(n_rows: int, n_cols: int) -> tuple[int, int, int]:
    block_m = 64 if n_rows <= 64 else 128 if n_rows <= 128 else 256 if n_rows <= 256 else 512
    block_n = 64 if n_cols >= 64 else 32
    return block_m, block_n, 8 if block_n == 64 else 4
# EVOLVE-BLOCK-END


def bias_relu_backward_torch(
    dy: torch.Tensor,
    x: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference formula for the evaluator."""
    dx = torch.where(x + bias > 0, dy, torch.zeros_like(dy))
    dbias = torch.sum(dx.float(), dim=0).to(bias.dtype)
    return dx, dbias


def bias_relu_backward_triton(
    dy: torch.Tensor,
    x: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (dx, dbias) for y = relu(x + bias)."""
    if dy.shape != x.shape:
        raise ValueError(f"dy and x must have the same shape, got {dy.shape} and {x.shape}")
    if dy.ndim != 2:
        raise ValueError(f"expected 2D tensors, got shape {dy.shape}")
    if bias.ndim != 1 or bias.shape[0] != x.shape[1]:
        raise ValueError(f"bias shape {bias.shape} must match x cols {x.shape[1]}")
    if not dy.is_cuda or not x.is_cuda or not bias.is_cuda:
        raise ValueError("bias_relu_backward_triton requires CUDA tensors")

    dy = dy.contiguous()
    x = x.contiguous()
    bias = bias.contiguous()
    dx = torch.empty_like(x)
    dbias = torch.empty_like(bias)

    n_rows, n_cols = x.shape

    if n_rows <= 1024:
        bm, bn, nw = _fused_launch_params(n_rows, n_cols)
        _bias_relu_backward_fused_kernel[(triton.cdiv(n_cols, bn),)](
            dy, x, bias, dx, dbias, n_rows, n_cols,
            BM=bm, BN=bn, num_warps=nw,
        )
        return dx, dbias

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

    block_m, block_n, num_warps = _dbias_launch_params(n_rows, n_cols)
    _bias_relu_backward_dbias_kernel[(triton.cdiv(n_cols, block_n),)](
        dx,
        dbias,
        n_rows,
        n_cols,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return dx, dbias


def run_example():
    """Small manual smoke test for GPU nodes."""
    x = torch.randn((128, 256), device="cuda", dtype=torch.float32)
    bias = torch.randn((256,), device="cuda", dtype=torch.float32)
    dy = torch.randn_like(x)
    dx, dbias = bias_relu_backward_triton(dy, x, bias)
    ref_dx, ref_dbias = bias_relu_backward_torch(dy, x, bias)
    return {
        "dx_max_abs_error": torch.max(torch.abs(dx - ref_dx)).item(),
        "dbias_max_abs_error": torch.max(torch.abs(dbias - ref_dbias)).item(),
    }


if __name__ == "__main__":
    print(run_example())
