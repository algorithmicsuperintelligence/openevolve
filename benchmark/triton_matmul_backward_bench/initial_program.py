"""Triton matmul backward optimization target for OpenEvolve.

The fixed public API is ``matmul_backward_triton(dc, a, b)`` returning (da, db)
for c = a @ b. The seed delegates to the manually verified naive tiled baseline.
Agents may replace the EVOLVE-BLOCK with a faster implementation that preserves
the API.
"""

import torch


# EVOLVE-BLOCK-START
from backward_naive_triton import matmul_backward_naive_triton as _seed_backward
# EVOLVE-BLOCK-END


def matmul_backward_torch(
    dc: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference formula used for manual checks."""
    da = dc @ b.transpose(0, 1)
    db = a.transpose(0, 1) @ dc
    return da.to(a.dtype), db.to(b.dtype)


def matmul_backward_triton(
    dc: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (da, db) for c = a @ b."""
    return _seed_backward(dc, a, b)


def run_example():
    """Small manual smoke test for GPU nodes."""
    a = torch.randn((128, 256), device="cuda", dtype=torch.float32)
    b = torch.randn((256, 64), device="cuda", dtype=torch.float32)
    dc = torch.randn((128, 64), device="cuda", dtype=torch.float32)
    da, db = matmul_backward_triton(dc, a, b)
    ref_da, ref_db = matmul_backward_torch(dc, a, b)
    return {
        "da_max_abs_error": torch.max(torch.abs(da - ref_da)).item(),
        "db_max_abs_error": torch.max(torch.abs(db - ref_db)).item(),
    }


if __name__ == "__main__":
    print(run_example())
