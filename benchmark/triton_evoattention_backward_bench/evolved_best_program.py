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
import triton
import triton.language as tl
from backward_naive_triton import evoattention_backward_naive_triton as _seed_backward


@triton.jit
def _pre_kernel(Q, K, V, DO, RM, PB, LSE, DELTA,
                S: tl.constexpr, R: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                SCALE: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr, BD: tl.constexpr):
    pid_q = tl.program_id(0)
    pid = tl.program_id(1)
    h = pid % H
    t = pid // H
    s = t % S
    b = t // S
    oq = pid_q * BQ + tl.arange(0, BQ)
    od = tl.arange(0, BD)
    q = tl.load(Q + ((b * S + s) * R * H + oq[:, None] * H + h) * D + od[None, :],
                mask=(oq[:, None] < R) & (od[None, :] < D), other=0.0)
    m = tl.full((BQ,), -float("inf"), tl.float32)
    l = tl.zeros((BQ,), tl.float32)
    acc = tl.zeros((BQ, BD), tl.float32)
    for st in range(0, R, BK):
        ok = st + tl.arange(0, BK)
        k = tl.load(K + ((b * S + s) * R * H + ok[:, None] * H + h) * D + od[None, :],
                    mask=(ok[:, None] < R) & (od[None, :] < D), other=0.0)
        v = tl.load(V + ((b * S + s) * R * H + ok[:, None] * H + h) * D + od[None, :],
                    mask=(ok[:, None] < R) & (od[None, :] < D), other=0.0)
        sc = tl.dot(q, tl.trans(k)) * SCALE
        sc += tl.load(PB + (b * H + h) * R * R + oq[:, None] * R + ok[None, :],
                      mask=(oq[:, None] < R) & (ok[None, :] < R), other=-float("inf"))
        sc += tl.load(RM + (b * S + s) * R + ok[None, :], mask=ok[None, :] < R, other=-float("inf"))
        sc = tl.where((oq[:, None] < R) & (ok[None, :] < R), sc, -float("inf"))
        nm = tl.maximum(m, tl.max(sc, 1))
        p = tl.exp(sc - nm[:, None])
        a = tl.exp(m - nm)
        acc = acc * a[:, None] + tl.dot(p, v.to(tl.float32), input_precision="tf32")
        l = l * a + tl.sum(p, 1)
        m = nm
    o = acc / l[:, None]
    do = tl.load(DO + ((b * S + s) * R * H + oq[:, None] * H + h) * D + od[None, :],
                 mask=(oq[:, None] < R) & (od[None, :] < D), other=0.0)
    tl.store(LSE + ((b * S + s) * H + h) * R + oq, m + tl.log(l), mask=oq < R)
    tl.store(DELTA + ((b * S + s) * H + h) * R + oq, tl.sum(o * do, 1), mask=oq < R)


@triton.jit
def _dq_kernel(Q, K, V, DO, RM, PB, LSE, DELTA, DQ, DPB,
               S: tl.constexpr, R: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
               SCALE: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr, BD: tl.constexpr):
    pid_q = tl.program_id(0)
    pid = tl.program_id(1)
    h = pid % H
    t = pid // H
    s = t % S
    b = t // S
    oq = pid_q * BQ + tl.arange(0, BQ)
    od = tl.arange(0, BD)
    q = tl.load(Q + ((b * S + s) * R * H + oq[:, None] * H + h) * D + od[None, :],
                mask=(oq[:, None] < R) & (od[None, :] < D), other=0.0)
    do = tl.load(DO + ((b * S + s) * R * H + oq[:, None] * H + h) * D + od[None, :],
                 mask=(oq[:, None] < R) & (od[None, :] < D), other=0.0)
    lse = tl.load(LSE + ((b * S + s) * H + h) * R + oq, mask=oq < R, other=0.0)
    delta = tl.load(DELTA + ((b * S + s) * H + h) * R + oq, mask=oq < R, other=0.0)
    dq = tl.zeros((BQ, BD), tl.float32)
    for st in range(0, R, BK):
        ok = st + tl.arange(0, BK)
        k = tl.load(K + ((b * S + s) * R * H + ok[:, None] * H + h) * D + od[None, :],
                    mask=(ok[:, None] < R) & (od[None, :] < D), other=0.0)
        v = tl.load(V + ((b * S + s) * R * H + ok[:, None] * H + h) * D + od[None, :],
                    mask=(ok[:, None] < R) & (od[None, :] < D), other=0.0)
        sc = tl.dot(q, tl.trans(k)) * SCALE
        sc += tl.load(PB + (b * H + h) * R * R + oq[:, None] * R + ok[None, :],
                      mask=(oq[:, None] < R) & (ok[None, :] < R), other=-float("inf"))
        sc += tl.load(RM + (b * S + s) * R + ok[None, :], mask=ok[None, :] < R, other=-float("inf"))
        p = tl.exp(sc - lse[:, None])
        p = tl.where((oq[:, None] < R) & (ok[None, :] < R), p, 0.0)
        dp = tl.dot(do.to(tl.float32), tl.trans(v.to(tl.float32)), input_precision="tf32")
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds, k.to(tl.float32), input_precision="tf32") * SCALE
        if S == 1:
            tl.store(DPB + (b * H + h) * R * R + oq[:, None] * R + ok[None, :], ds,
                     mask=(oq[:, None] < R) & (ok[None, :] < R))
        else:
            tl.atomic_add(DPB + (b * H + h) * R * R + oq[:, None] * R + ok[None, :], ds,
                          mask=(oq[:, None] < R) & (ok[None, :] < R), sem="relaxed")
    tl.store(DQ + ((b * S + s) * R * H + oq[:, None] * H + h) * D + od[None, :], dq,
             mask=(oq[:, None] < R) & (od[None, :] < D))


@triton.jit
def _dkv_kernel(Q, K, V, DO, RM, PB, LSE, DELTA, DK, DV,
                S: tl.constexpr, R: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                SCALE: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr, BD: tl.constexpr):
    pid_k = tl.program_id(0)
    pid = tl.program_id(1)
    h = pid % H
    t = pid // H
    s = t % S
    b = t // S
    ok = pid_k * BK + tl.arange(0, BK)
    od = tl.arange(0, BD)
    k = tl.load(K + ((b * S + s) * R * H + ok[:, None] * H + h) * D + od[None, :],
                mask=(ok[:, None] < R) & (od[None, :] < D), other=0.0)
    v = tl.load(V + ((b * S + s) * R * H + ok[:, None] * H + h) * D + od[None, :],
                mask=(ok[:, None] < R) & (od[None, :] < D), other=0.0)
    dk = tl.zeros((BK, BD), tl.float32)
    dv = tl.zeros((BK, BD), tl.float32)
    for st in range(0, R, BQ):
        oq = st + tl.arange(0, BQ)
        q = tl.load(Q + ((b * S + s) * R * H + oq[:, None] * H + h) * D + od[None, :],
                    mask=(oq[:, None] < R) & (od[None, :] < D), other=0.0)
        do = tl.load(DO + ((b * S + s) * R * H + oq[:, None] * H + h) * D + od[None, :],
                     mask=(oq[:, None] < R) & (od[None, :] < D), other=0.0)
        lse = tl.load(LSE + ((b * S + s) * H + h) * R + oq, mask=oq < R, other=0.0)
        delta = tl.load(DELTA + ((b * S + s) * H + h) * R + oq, mask=oq < R, other=0.0)
        sc = tl.dot(q, tl.trans(k)) * SCALE
        sc += tl.load(PB + (b * H + h) * R * R + oq[:, None] * R + ok[None, :],
                      mask=(oq[:, None] < R) & (ok[None, :] < R), other=-float("inf"))
        sc += tl.load(RM + (b * S + s) * R + ok[None, :], mask=ok[None, :] < R, other=-float("inf"))
        p = tl.exp(sc - lse[:, None])
        p = tl.where((oq[:, None] < R) & (ok[None, :] < R), p, 0.0)
        dv += tl.dot(tl.trans(p), do.to(tl.float32), input_precision="tf32")
        dp = tl.dot(do.to(tl.float32), tl.trans(v.to(tl.float32)), input_precision="tf32")
        ds = p * (dp - delta[:, None])
        dk += tl.dot(tl.trans(ds), q.to(tl.float32), input_precision="tf32") * SCALE
    tl.store(DK + ((b * S + s) * R * H + ok[:, None] * H + h) * D + od[None, :], dk,
             mask=(ok[:, None] < R) & (od[None, :] < D))
    tl.store(DV + ((b * S + s) * R * H + ok[:, None] * H + h) * D + od[None, :], dv,
             mask=(ok[:, None] < R) & (od[None, :] < D))


def _flash_backward(do, q, k, v, res_mask, pair_bias):
    if not (q.is_cuda and q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and do.is_contiguous()):
        return _seed_backward(do, q, k, v, res_mask, pair_bias)
    B, S, R, H, D = q.shape
    if D > 128:
        return _seed_backward(do, q, k, v, res_mask, pair_bias)
    BQ = 32
    BK = 64 if D <= 32 else 32
    BD = max(16, triton.next_power_of_2(D))
    nw = 4 if BD <= 64 else 8
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dpb = torch.empty((B, 1, H, R, R), device=q.device, dtype=torch.float32) if S == 1 else torch.zeros((B, 1, H, R, R), device=q.device, dtype=torch.float32)
    lse = torch.empty((B, S, H, R), device=q.device, dtype=torch.float32)
    delta = torch.empty_like(lse)
    grid_q = (triton.cdiv(R, BQ), B * S * H)
    grid_k = (triton.cdiv(R, BK), B * S * H)
    scale = D ** -0.5
    _pre_kernel[grid_q](q, k, v, do, res_mask, pair_bias, lse, delta,
                        S, R, H, D, scale, BQ, BK, BD, num_warps=nw, num_stages=3)
    _dkv_kernel[grid_k](q, k, v, do, res_mask, pair_bias, lse, delta, dk, dv,
                        S, R, H, D, scale, BQ, BK, BD, num_warps=nw, num_stages=3)
    _dq_kernel[grid_q](q, k, v, do, res_mask, pair_bias, lse, delta, dq, dpb,
                       S, R, H, D, scale, BQ, BK, BD, num_warps=nw, num_stages=3)
    return dq, dk, dv, dpb
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
    return _flash_backward(do, q, k, v, res_mask, pair_bias)


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
