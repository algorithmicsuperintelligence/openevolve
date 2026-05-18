"""Simple Triton forward kernel for row-wise LayerNorm."""

import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_forward_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
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

    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = centered * rstd * weight + bias
    tl.store(y_ptr + row_offsets, y, mask=mask)


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


def layernorm_forward_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute y = layer_norm(x, weight, bias) for contiguous 2D CUDA tensors."""
    if x.ndim != 2:
        raise ValueError(f"expected x to be 2D, got shape {tuple(x.shape)}")
    if weight.ndim != 1 or bias.ndim != 1:
        raise ValueError("weight and bias must be 1D tensors")
    if weight.shape[0] != x.shape[1] or bias.shape[0] != x.shape[1]:
        raise ValueError("weight and bias length must match x hidden dimension")
    if not x.is_cuda or not weight.is_cuda or not bias.is_cuda:
        raise ValueError("layernorm_forward_triton requires CUDA tensors")

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    y = torch.empty_like(x)

    n_rows, n_cols = x.shape
    block_size, num_warps, num_stages = _row_launch_params(n_cols)
    _layernorm_forward_kernel[(n_rows,)](
        x,
        weight,
        bias,
        y,
        n_cols,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y
