"""PyTorch reference for the fused LayerNorm -> Linear operator.

This mirrors MegaFold's ``LayernormLinear`` (arXiv:2506.20686), which replaces a
``LayerNorm(K)`` followed by ``Linear(K, N, bias=False)`` with a single fused op.
The linear weight is stored as ``[K, N]`` (so the matmul is ``y_hat @ B`` directly,
matching MegaFold's ``F.linear(x_hat, linear_weight.T)`` convention).

    x:             [M, K]   (or [*, K]; flattened to [M, K])
    weight (gamma):[K]      LayerNorm scale
    bias   (beta): [K]      LayerNorm shift
    linear_weight: [K, N]   Linear weight (no linear bias, as in AF3 Transition)
    out:           [M, N]
"""

import torch
import torch.nn.functional as F


def layernorm_linear_forward_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    linear_weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute out = layer_norm(x, gamma, beta) @ linear_weight."""
    if x.ndim != 2:
        raise ValueError(f"expected 2D x [M, K], got {tuple(x.shape)}")
    k = x.shape[-1]
    if weight.shape != (k,) or bias.shape != (k,):
        raise ValueError("weight and bias must be 1D tensors of length K")
    if linear_weight.shape[0] != k:
        raise ValueError("linear_weight.shape[0] must match x.shape[1] (K)")
    y_hat = F.layer_norm(x, (k,), weight, bias, eps)
    return y_hat @ linear_weight
