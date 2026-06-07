"""Triton RMSNorm backward-dx optimization target for OpenEvolve.

The fixed public API is ``rmsnorm_backward_dx_triton(dy, x, weight, eps)``.
OpenEvolve should only modify the code inside the EVOLVE-BLOCK.
"""

import torch
import triton
import triton.language as tl


# EVOLVE-BLOCK-START
@triton.jit
def _rmsnorm_backward_dx_kernel(
    dy_ptr,
    x_ptr,
    weight_ptr,
    dx_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    stride_row: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One Triton program computes dx for one RMSNorm row."""
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row_offsets = row_idx * stride_row + offsets

    x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    grad_normed = dy * weight
    projection = tl.sum(grad_normed * x, axis=0)
    dx = grad_normed * inv_rms - x * (inv_rms * inv_rms * inv_rms) * projection / n_cols

    tl.store(dx_ptr + row_offsets, dx, mask=mask)


def _kernel_launch_params(n_cols: int):
    """Return conservative launch parameters for RMSNorm row reductions."""
    block_size = 1 << (n_cols - 1).bit_length()
    num_warps = 4
    num_stages = 4

    if block_size >= 2048:
        num_warps = 8
        num_stages = 3
    if block_size >= 4096:
        num_warps = 8
        num_stages = 3

    return block_size, num_warps, num_stages
# EVOLVE-BLOCK-END


def rmsnorm_forward_torch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    inv_rms = torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + eps)
    return (x.float() * inv_rms * weight.float()).to(x.dtype)


def rmsnorm_backward_dx_torch(
    dy: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    x_f32 = x.float()
    dy_f32 = dy.float()
    weight_f32 = weight.float()
    inv_rms = torch.rsqrt(torch.mean(x_f32 * x_f32, dim=-1, keepdim=True) + eps)
    grad_normed = dy_f32 * weight_f32
    projection = torch.sum(grad_normed * x_f32, dim=-1, keepdim=True)
    n_cols = x.shape[-1]
    dx = grad_normed * inv_rms - x_f32 * (inv_rms**3) * projection / n_cols
    return dx.to(x.dtype)


def rmsnorm_backward_dx_triton(
    dy: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Compute RMSNorm dx for 2D tensors shaped [rows, cols]."""
    if dy.shape != x.shape:
        raise ValueError(f"dy and x must have the same shape, got {dy.shape} and {x.shape}")
    if dy.ndim != 2:
        raise ValueError(f"expected 2D tensors, got shape {dy.shape}")
    if weight.ndim != 1 or weight.shape[0] != x.shape[1]:
        raise ValueError(f"weight must have shape [{x.shape[1]}], got {weight.shape}")
    if not dy.is_cuda or not x.is_cuda or not weight.is_cuda:
        raise ValueError("rmsnorm_backward_dx_triton requires CUDA tensors")

    dy = dy.contiguous()
    x = x.contiguous()
    weight = weight.contiguous()
    dx = torch.empty_like(x)

    n_rows, n_cols = x.shape
    block_size, num_warps, num_stages = _kernel_launch_params(n_cols)
    if block_size < n_cols:
        raise ValueError(f"BLOCK_SIZE {block_size} is smaller than n_cols {n_cols}")

    _rmsnorm_backward_dx_kernel[(n_rows,)](
        dy,
        x,
        weight,
        dx,
        n_cols,
        eps,
        x.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return dx


def run_example():
    """Small manual smoke test for GPU nodes."""
    x = torch.randn((256, 2048), device="cuda", dtype=torch.float16)
    weight = torch.randn((2048,), device="cuda", dtype=torch.float16)
    dy = torch.randn_like(x)
    dx = rmsnorm_backward_dx_triton(dy, x, weight)
    ref = rmsnorm_backward_dx_torch(dy, x, weight, 1e-6)
    return torch.max(torch.abs(dx.float() - ref.float())).item()


if __name__ == "__main__":
    print(f"max_abs_error={run_example():.6e}")
