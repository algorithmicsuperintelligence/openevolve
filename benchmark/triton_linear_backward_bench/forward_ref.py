"""PyTorch reference for a Linear layer: y = x @ weight.T + bias."""

import torch
import torch.nn.functional as F


def linear_forward_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Compute y = x @ weight.T + bias for x [M, K], weight [N, K], bias [N]."""
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("x and weight must be 2D tensors")
    if weight.shape[1] != x.shape[1]:
        raise ValueError("weight.shape[1] must match x.shape[1] (K)")
    if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
        raise ValueError("bias must be a 1D tensor of length N (= weight.shape[0])")
    return F.linear(x, weight, bias)
