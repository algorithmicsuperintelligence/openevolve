"""Triton Linear backward optimization target for OpenEvolve.

The fixed public API is ``linear_backward_triton(dy, x, weight)`` returning
(dx, dweight, dbias) for y = x @ weight.T + bias. The seed delegates to the
manually verified naive tiled baseline. Agents may replace the EVOLVE-BLOCK
with a faster implementation that preserves the API.
"""

import torch


# EVOLVE-BLOCK-START
from backward_naive_triton import linear_backward_naive_triton as _seed_backward
# EVOLVE-BLOCK-END


def linear_backward_torch(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference formula used for manual checks."""
    dx = dy @ weight
    dweight = dy.transpose(0, 1) @ x
    dbias = torch.sum(dy, dim=0)
    return dx.to(x.dtype), dweight.to(weight.dtype), dbias.to(dy.dtype)


def linear_backward_triton(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (dx, dweight, dbias) for a Linear layer backward."""
    return _seed_backward(dy, x, weight)


def run_example():
    """Small manual smoke test for GPU nodes."""
    x = torch.randn((128, 256), device="cuda", dtype=torch.float32)
    weight = torch.randn((64, 256), device="cuda", dtype=torch.float32)
    dy = torch.randn((128, 64), device="cuda", dtype=torch.float32)
    dx, dweight, dbias = linear_backward_triton(dy, x, weight)
    ref_dx, ref_dweight, ref_dbias = linear_backward_torch(dy, x, weight)
    return {
        "dx_max_abs_error": torch.max(torch.abs(dx - ref_dx)).item(),
        "dweight_max_abs_error": torch.max(torch.abs(dweight - ref_dweight)).item(),
        "dbias_max_abs_error": torch.max(torch.abs(dbias - ref_dbias)).item(),
    }


if __name__ == "__main__":
    print(run_example())
