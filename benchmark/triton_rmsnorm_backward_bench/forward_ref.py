"""PyTorch reference for row-wise RMSNorm."""

import torch


def rmsnorm_forward_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute RMSNorm over the last dimension for x shaped [rows, hidden]."""
    if x.ndim != 2:
        raise ValueError(f"expected x to be 2D, got shape {tuple(x.shape)}")
    if weight.ndim != 1:
        raise ValueError("weight must be a 1D tensor")
    if weight.shape[0] != x.shape[1]:
        raise ValueError("weight length must match x hidden dimension")

    ms = torch.mean(x.float() * x.float(), dim=-1, keepdim=True)
    x_hat = x.float() * torch.rsqrt(ms + eps)
    return (x_hat * weight.float()).to(x.dtype)
