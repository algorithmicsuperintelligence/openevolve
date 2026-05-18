"""Triton LayerNorm backward optimization target for OpenEvolve.

The fixed public API is ``layernorm_backward_triton(dy, x, weight, bias, eps)``.
The seed delegates to the manually verified naive decomposed baseline. Agents
may replace the EVOLVE-BLOCK with a faster implementation that preserves the API.
"""

import torch


# EVOLVE-BLOCK-START
from backward_naive_triton import layernorm_backward_naive_triton as _seed_backward
# EVOLVE-BLOCK-END


def layernorm_backward_torch(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference formula used for manual checks."""
    x_f32 = x.float()
    dy_f32 = dy.float()
    weight_f32 = weight.float()
    mean = torch.mean(x_f32, dim=-1, keepdim=True)
    var = torch.mean((x_f32 - mean) ** 2, dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    xhat = (x_f32 - mean) * rstd
    one = dy_f32 * weight_f32
    mean_one = torch.mean(one, dim=-1, keepdim=True)
    mean_xhat_one = torch.mean(xhat * one, dim=-1, keepdim=True)
    dx = (one - mean_one - xhat * mean_xhat_one) * rstd
    dweight = torch.sum(dy_f32 * xhat, dim=0)
    dbias = torch.sum(dy_f32, dim=0)
    return dx.to(x.dtype), dweight.to(weight.dtype), dbias.to(bias.dtype)


def layernorm_backward_triton(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (dx, dweight, dbias) for row-wise LayerNorm."""
    return _seed_backward(dy, x, weight, bias, eps)


def run_example():
    """Small manual smoke test for GPU nodes."""
    x = torch.randn((64, 256), device="cuda", dtype=torch.float32)
    weight = torch.randn((256,), device="cuda", dtype=torch.float32)
    bias = torch.randn((256,), device="cuda", dtype=torch.float32)
    dy = torch.randn_like(x)
    dx, dweight, dbias = layernorm_backward_triton(dy, x, weight, bias)
    ref_dx, ref_dweight, ref_dbias = layernorm_backward_torch(dy, x, weight, bias)
    return {
        "dx_max_abs_error": torch.max(torch.abs(dx - ref_dx)).item(),
        "dweight_max_abs_error": torch.max(torch.abs(dweight - ref_dweight)).item(),
        "dbias_max_abs_error": torch.max(torch.abs(dbias - ref_dbias)).item(),
    }


if __name__ == "__main__":
    print(run_example())
