"""Triton fused LayerNorm -> Linear backward optimization target for OpenEvolve.

The fixed public API is
``layernorm_linear_backward_triton(dout, x, weight, bias, linear_weight, eps)``
returning ``(dx, dlinear_weight, dweight, dbias)`` for
``out = layer_norm(x, weight, bias) @ linear_weight``.

The seed delegates to a manually verified naive *unfused* Triton backward
(``backward_naive_triton.py``; ``backward_naive.py`` is the readable PyTorch
reference). Agents should replace the EVOLVE-BLOCK with a faster, fused Triton
backward that matches
MegaFold's strategy (arXiv:2506.20686): fuse the LayerNorm into the matmul
epilogue, recompute x_hat in the backward instead of storing it, and atomic-add
the weight gradients. See benchmark_strong_baselines.py for the expert reference.
"""

import torch


# EVOLVE-BLOCK-START
from backward_naive_triton import layernorm_linear_backward_naive_triton as _seed_backward
# EVOLVE-BLOCK-END


def layernorm_linear_backward_torch(
    dout: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    linear_weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference formula used for manual checks."""
    k = x.shape[-1]
    x_f = x.float()
    gamma = weight.float()
    mean = x_f.mean(dim=-1, keepdim=True)
    var = ((x_f - mean) ** 2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    x_hat = (x_f - mean) * rstd
    y_hat = x_hat * gamma + bias.float()
    dy_hat = dout.float() @ linear_weight.float().t()
    d_linear_weight = y_hat.t() @ dout.float()
    dgamma = (dy_hat * x_hat).sum(dim=0)
    dbeta = dy_hat.sum(dim=0)
    wdy = dy_hat * gamma
    c1 = (x_hat * wdy).mean(dim=-1, keepdim=True)
    c2 = wdy.mean(dim=-1, keepdim=True)
    dx = (wdy - (x_hat * c1 + c2)) * rstd
    return (
        dx.to(x.dtype),
        d_linear_weight.to(linear_weight.dtype),
        dgamma.to(weight.dtype),
        dbeta.to(bias.dtype),
    )


def layernorm_linear_backward_triton(
    dout: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    linear_weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (dx, dlinear_weight, dweight, dbias) for LayerNorm -> Linear backward."""
    return _seed_backward(dout, x, weight, bias, linear_weight, eps)


def run_example():
    """Small manual smoke test for GPU nodes."""
    m, k, n = 128, 128, 256
    dtype = torch.float16
    x = torch.randn((m, k), device="cuda", dtype=dtype)
    weight = torch.randn((k,), device="cuda", dtype=dtype)
    bias = torch.randn((k,), device="cuda", dtype=dtype)
    linear_weight = (torch.randn((k, n), device="cuda", dtype=dtype) * k ** -0.5)
    dout = torch.randn((m, n), device="cuda", dtype=dtype)

    dx, dlw, dg, db = layernorm_linear_backward_triton(dout, x, weight, bias, linear_weight)
    rx, rlw, rg, rb = layernorm_linear_backward_torch(dout, x, weight, bias, linear_weight)
    return {
        "dx_max_abs_error": torch.max(torch.abs(dx - rx)).item(),
        "dlinear_weight_max_abs_error": torch.max(torch.abs(dlw - rlw)).item(),
        "dweight_max_abs_error": torch.max(torch.abs(dg - rg)).item(),
        "dbias_max_abs_error": torch.max(torch.abs(db - rb)).item(),
    }


if __name__ == "__main__":
    print(run_example())
