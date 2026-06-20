"""Naive (unfused) backward baseline for the fused LayerNorm -> Linear operator.

This is the deliberately simple, easy-to-verify seed for OpenEvolve. It computes
each gradient with plain PyTorch ops in float32, materializing the intermediate
normalized activation ``y_hat`` and the upstream-of-linear gradient ``dy_hat``.
The evolve target is MegaFold's fused strategy (arXiv:2506.20686): fuse the
LayerNorm into the matmul epilogue, recompute ``x_hat`` in the backward instead of
storing it, and atomic-add the weight gradients.

Forward:  y_hat = (x - mean) * rstd * gamma + beta ;  out = y_hat @ B
Backward (B is linear_weight [K, N]):

    dy_hat        = dout @ B^T                      [M, K]
    dB            = y_hat^T @ dout                  [K, N]
    dgamma        = sum(dy_hat * x_hat, axis=0)     [K]
    dbeta         = sum(dy_hat, axis=0)             [K]
    wdy           = dy_hat * gamma
    c1            = mean(x_hat * wdy, axis=-1)       [M]
    c2            = mean(wdy, axis=-1)               [M]
    dx            = (wdy - x_hat * c1 - c2) * rstd  [M, K]
"""

import torch


def layernorm_linear_backward_naive(
    dout: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    linear_weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (dx, dlinear_weight, dweight, dbias) for LayerNorm -> Linear."""
    if x.ndim != 2:
        raise ValueError(f"expected 2D x [M, K], got {tuple(x.shape)}")
    if not (x.is_cuda and dout.is_cuda and linear_weight.is_cuda):
        raise ValueError("layernorm_linear_backward_naive requires CUDA tensors")

    k = x.shape[-1]
    x_f = x.float()
    gamma = weight.float()
    dout_f = dout.float()
    b_f = linear_weight.float()  # [K, N]

    # Recompute the LayerNorm statistics and normalized activation.
    mean = x_f.mean(dim=-1, keepdim=True)
    var = ((x_f - mean) ** 2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    x_hat = (x_f - mean) * rstd                      # [M, K]
    y_hat = x_hat * gamma + bias.float()             # [M, K]

    # Linear backward.
    dy_hat = dout_f @ b_f.t()                        # [M, K]
    d_linear_weight = y_hat.t() @ dout_f             # [K, N]

    # LayerNorm affine-parameter gradients.
    dgamma = (dy_hat * x_hat).sum(dim=0)             # [K]
    dbeta = dy_hat.sum(dim=0)                        # [K]

    # LayerNorm input gradient.
    wdy = dy_hat * gamma                             # [M, K]
    c1 = (x_hat * wdy).mean(dim=-1, keepdim=True)    # [M, 1]
    c2 = wdy.mean(dim=-1, keepdim=True)              # [M, 1]
    dx = (wdy - (x_hat * c1 + c2)) * rstd            # [M, K]

    return (
        dx.to(x.dtype),
        d_linear_weight.to(linear_weight.dtype),
        dgamma.to(weight.dtype),
        dbeta.to(bias.dtype),
    )
