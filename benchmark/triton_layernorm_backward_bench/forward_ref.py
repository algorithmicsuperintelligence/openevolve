"""PyTorch reference for row-wise LayerNorm."""

import torch


def layernorm_forward_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute LayerNorm over the last dimension for x shaped [rows, hidden]."""
    if x.ndim != 2:
        raise ValueError(f"expected x to be 2D, got shape {tuple(x.shape)}")
    if weight.ndim != 1 or bias.ndim != 1:
        raise ValueError("weight and bias must be 1D tensors")
    if weight.shape[0] != x.shape[1] or bias.shape[0] != x.shape[1]:
        raise ValueError("weight and bias length must match x hidden dimension")

    mean = torch.mean(x.float(), dim=-1, keepdim=True)
    var = torch.mean((x.float() - mean) ** 2, dim=-1, keepdim=True)
    x_hat = (x.float() - mean) * torch.rsqrt(var + eps)
    return (x_hat * weight.float() + bias.float()).to(x.dtype)
