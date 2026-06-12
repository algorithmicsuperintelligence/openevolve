"""Triton RMSNorm backward optimization target for OpenEvolve.

The fixed public API is ``rmsnorm_backward_triton(dy, x, weight, eps)``.
The seed delegates to the manually verified naive decomposed baseline. Agents
may replace the EVOLVE-BLOCK with a faster implementation that preserves the API.
"""

import torch


# EVOLVE-BLOCK-START
from backward_naive_triton import rmsnorm_backward_naive_triton as _seed_backward
# EVOLVE-BLOCK-END


def rmsnorm_backward_torch(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference formula used for manual checks."""
    x_f32 = x.float()
    dy_f32 = dy.float()
    weight_f32 = weight.float()
    ms = torch.mean(x_f32 * x_f32, dim=-1, keepdim=True)
    rrms = torch.rsqrt(ms + eps)
    xhat = x_f32 * rrms
    g = dy_f32 * weight_f32
    mean_g_xhat = torch.mean(g * xhat, dim=-1, keepdim=True)
    dx = (g - xhat * mean_g_xhat) * rrms
    dweight = torch.sum(dy_f32 * xhat, dim=0)
    return dx.to(x.dtype), dweight.to(weight.dtype)


def rmsnorm_backward_triton(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (dx, dweight) for row-wise RMSNorm."""
    return _seed_backward(dy, x, weight, eps)


def run_example():
    """Small manual smoke test for GPU nodes."""
    x = torch.randn((64, 256), device="cuda", dtype=torch.float32)
    weight = torch.randn((256,), device="cuda", dtype=torch.float32)
    dy = torch.randn_like(x)
    dx, dweight = rmsnorm_backward_triton(dy, x, weight)
    ref_dx, ref_dweight = rmsnorm_backward_torch(dy, x, weight)
    return {
        "dx_max_abs_error": torch.max(torch.abs(dx - ref_dx)).item(),
        "dweight_max_abs_error": torch.max(torch.abs(dweight - ref_dweight)).item(),
    }


if __name__ == "__main__":
    print(run_example())
