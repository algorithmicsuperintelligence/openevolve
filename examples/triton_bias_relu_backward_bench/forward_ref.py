"""PyTorch reference for the bias + ReLU forward operator."""

import torch


def bias_relu_forward_ref(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Compute y = relu(x + bias) for x shaped [rows, cols] and bias [cols]."""
    if x.ndim != 2:
        raise ValueError(f"expected x to be 2D, got shape {tuple(x.shape)}")
    if bias.ndim != 1:
        raise ValueError(f"expected bias to be 1D, got shape {tuple(bias.shape)}")
    if x.shape[1] != bias.shape[0]:
        raise ValueError(f"bias length {bias.shape[0]} must match x cols {x.shape[1]}")
    return torch.relu(x + bias)
