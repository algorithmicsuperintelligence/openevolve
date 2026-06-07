"""Simple Triton forward kernel for y = relu(x + bias)."""

import torch
import triton
import triton.language as tl


@triton.jit
def _bias_relu_forward_kernel(
    x_ptr,
    bias_ptr,
    y_ptr,
    total: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    cols = offsets % n_cols

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = tl.maximum(x + bias, 0.0)

    tl.store(y_ptr + offsets, y, mask=mask)


def bias_relu_forward_triton(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Compute y = relu(x + bias) for contiguous 2D CUDA tensors."""
    if x.ndim != 2:
        raise ValueError(f"expected x to be 2D, got shape {tuple(x.shape)}")
    if bias.ndim != 1:
        raise ValueError(f"expected bias to be 1D, got shape {tuple(bias.shape)}")
    if x.shape[1] != bias.shape[0]:
        raise ValueError(f"bias length {bias.shape[0]} must match x cols {x.shape[1]}")
    if not x.is_cuda or not bias.is_cuda:
        raise ValueError("bias_relu_forward_triton requires CUDA tensors")

    x = x.contiguous()
    bias = bias.contiguous()
    y = torch.empty_like(x)

    total = x.numel()
    block_size = 256
    grid = (triton.cdiv(total, block_size),)
    _bias_relu_forward_kernel[grid](
        x,
        bias,
        y,
        total,
        x.shape[1],
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return y
