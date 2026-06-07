"""Triton softmax backward optimization target for OpenEvolve.

The fixed public API is ``softmax_backward_triton(dy, y)``. OpenEvolve should
only modify the code inside the EVOLVE-BLOCK.
"""

import torch
import triton
import triton.language as tl


# EVOLVE-BLOCK-START
@triton.jit
def _softmax_backward_kernel(
    dy_ptr,
    y_ptr,
    dx_ptr,
    n_cols: tl.constexpr,
    stride_row: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One Triton program computes the backward pass for one row."""
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row_offsets = row_idx * stride_row + offsets

    dy = tl.load(dy_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)

    dot = tl.sum(dy * y, axis=0)
    dx = y * (dy - dot)

    tl.store(dx_ptr + row_offsets, dx, mask=mask)


def _kernel_launch_params(n_cols: int):
    """Return conservative launch parameters for row-wise reductions."""
    block_size = 1 << (n_cols - 1).bit_length()
    num_warps = 4
    num_stages = 4

    if block_size >= 2048:
        num_warps = 8
    if block_size >= 4096:
        num_warps = 8
        num_stages = 3

    return block_size, num_warps, num_stages
# EVOLVE-BLOCK-END


def softmax_backward_torch(dy: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Reference formula used by the evaluator."""
    return y * (dy - torch.sum(dy * y, dim=-1, keepdim=True))


def softmax_backward_triton(dy: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute softmax backward for 2D tensors shaped [rows, cols]."""
    if dy.shape != y.shape:
        raise ValueError(f"dy and y must have the same shape, got {dy.shape} and {y.shape}")
    if dy.ndim != 2:
        raise ValueError(f"expected 2D tensors, got shape {dy.shape}")
    if not dy.is_cuda or not y.is_cuda:
        raise ValueError("softmax_backward_triton requires CUDA tensors")

    dy = dy.contiguous()
    y = y.contiguous()
    dx = torch.empty_like(dy)

    n_rows, n_cols = dy.shape
    block_size, num_warps, num_stages = _kernel_launch_params(n_cols)
    if block_size < n_cols:
        raise ValueError(f"BLOCK_SIZE {block_size} is smaller than n_cols {n_cols}")

    _softmax_backward_kernel[(n_rows,)](
        dy,
        y,
        dx,
        n_cols,
        dy.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return dx


def run_example():
    """Small manual smoke test for GPU nodes."""
    x = torch.randn((128, 1024), device="cuda", dtype=torch.float32)
    y = torch.softmax(x, dim=-1)
    dy = torch.randn_like(y)
    dx = softmax_backward_triton(dy, y)
    ref = softmax_backward_torch(dy, y)
    return torch.max(torch.abs(dx - ref)).item()


if __name__ == "__main__":
    print(f"max_abs_error={run_example():.6e}")
