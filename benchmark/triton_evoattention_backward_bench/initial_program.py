"""Triton EvoformerAttention backward optimization target for OpenEvolve.

The fixed public API is
``evoattention_backward_triton(do, q, k, v, res_mask, pair_bias)`` returning
``(dq, dk, dv, d_pair_bias)`` for AlphaFold3-style EvoformerAttention.

The seed delegates to a manually verified naive single-kernel Triton backward
(``backward_naive_triton.py``: online-softmax recompute + atomic-add dK/dV/d_pair_bias;
``backward_naive.py`` is the readable materialized PyTorch reference). Agents should
replace the EVOLVE-BLOCK with a faster, fused, flash-attention-style Triton backward
(atomic-free two-pass), while preserving the API and matching the gradients.
MegaFold's fused kernel (arXiv:2506.20686) is the reference design to study and
beat; see benchmark_strong_baselines.py.
"""

import torch


# EVOLVE-BLOCK-START
from backward_naive_triton import evoattention_backward_naive_triton as _seed_backward
# EVOLVE-BLOCK-END


def evoattention_backward_torch(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    res_mask: torch.Tensor,
    pair_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference formula used for manual checks (materialized softmax backward)."""
    dim = q.shape[-1]
    scale = dim ** -0.5
    q_t = q.transpose(-2, -3).float()
    k_t = k.transpose(-2, -3).float()
    v_t = v.transpose(-2, -3).float()
    do_t = do.transpose(-2, -3).float()
    scores = torch.matmul(q_t * scale, k_t.transpose(-1, -2)) + pair_bias.float() + res_mask.float()
    probs = torch.softmax(scores, dim=-1)
    dv_t = torch.matmul(probs.transpose(-1, -2), do_t)
    dp = torch.matmul(do_t, v_t.transpose(-1, -2))
    ds = probs * (dp - (dp * probs).sum(dim=-1, keepdim=True))
    dq_t = scale * torch.matmul(ds, k_t)
    dk_t = scale * torch.matmul(ds.transpose(-1, -2), q_t)
    d_pair_bias = ds.sum(dim=1, keepdim=True)
    return (
        dq_t.transpose(-2, -3).contiguous().to(q.dtype),
        dk_t.transpose(-2, -3).contiguous().to(k.dtype),
        dv_t.transpose(-2, -3).contiguous().to(v.dtype),
        d_pair_bias.contiguous().to(pair_bias.dtype),
    )


def evoattention_backward_triton(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    res_mask: torch.Tensor,
    pair_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (dq, dk, dv, d_pair_bias) for EvoformerAttention backward."""
    return _seed_backward(do, q, k, v, res_mask, pair_bias)


def run_example():
    """Small manual smoke test for GPU nodes."""
    b, s, h, r, d = 1, 2, 4, 64, 16
    dtype = torch.float16
    q = torch.randn((b, s, r, h, d), device="cuda", dtype=dtype) * 0.5
    k = torch.randn((b, s, r, h, d), device="cuda", dtype=dtype) * 0.5
    v = torch.randn((b, s, r, h, d), device="cuda", dtype=dtype) * 0.5
    do = torch.randn((b, s, r, h, d), device="cuda", dtype=dtype) * 0.5
    res_mask = torch.zeros((b, s, 1, 1, r), device="cuda", dtype=torch.float32)
    pair_bias = torch.randn((b, 1, h, r, r), device="cuda", dtype=torch.float32) * 0.5

    dq, dk, dv, dpb = evoattention_backward_triton(do, q, k, v, res_mask, pair_bias)
    rq, rk, rv, rpb = evoattention_backward_torch(do, q, k, v, res_mask, pair_bias)
    return {
        "dq_max_abs_error": torch.max(torch.abs(dq - rq)).item(),
        "dk_max_abs_error": torch.max(torch.abs(dk - rk)).item(),
        "dv_max_abs_error": torch.max(torch.abs(dv - rv)).item(),
        "d_pair_bias_max_abs_error": torch.max(torch.abs(dpb - rpb)).item(),
    }


if __name__ == "__main__":
    print(run_example())
