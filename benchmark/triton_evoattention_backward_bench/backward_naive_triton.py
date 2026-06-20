"""Naive single-kernel Triton backward baseline for EvoformerAttention.

This is a deliberately simple, hand-written Triton backward. Unlike MegaFold's
optimized three-pass flash kernel (arXiv:2506.20686), it uses ONE kernel that, per
(batch, msa, head, query-block):

  * recomputes the row softmax statistics online (flash forward) to get O and the
    logsumexp M, then D = rowsum(dO * O);
  * makes a second streaming pass over key blocks to accumulate dQ (in registers)
    and atomic-adds dK, dV, and d_pair_bias.

The atomic-adds on dK/dV/d_pair_bias and the full recompute are the "naive" part;
the evolve target is MegaFold's atomic-free two-pass flash design. Block sizes are
fixed (no autotuning). Correctness is verified against PyTorch autograd.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _evoattn_bwd_kernel(
    Q, K, V, RES_MASK, PAIR_BIAS, DO,          # inputs (workspace layout)
    DQ, DK, DV, DPB,                            # outputs (fp32 accumulators)
    scale,
    S, H, R, D,
    MAT: tl.constexpr,                          # R * D, the (b,s,h) matrix stride
    BQ: tl.constexpr, BK: tl.constexpr, BD: tl.constexpr,
):
    q_block = tl.program_id(0).to(tl.int64)
    bsh = tl.program_id(1).to(tl.int64)

    b = bsh // (S * H)
    rem = bsh % (S * H)
    s = rem // H
    h = rem % H

    base = bsh * MAT                            # offset of this (b,s,h) [R, D] block
    base_pb = (b * H + h) * R * R               # pair_bias [B,1,H,R,R], broadcast over s
    base_rm = (b * S + s) * R                   # res_mask  [B,S,1,1,R]

    offs_q = q_block * BQ + tl.arange(0, BQ)
    offs_d = tl.arange(0, BD)
    q_mask = offs_q < R
    d_mask = offs_d < D

    q_ptrs = base + offs_q[:, None] * D + offs_d[None, :]
    Q_blk = tl.load(Q + q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
    dO_blk = tl.load(DO + q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)

    # ---- pass 1: online softmax to get M (logsumexp) and O ----
    m_i = tl.zeros((BQ,), dtype=tl.float32) - float("inf")
    l_i = tl.zeros((BQ,), dtype=tl.float32)
    acc_o = tl.zeros((BQ, BD), dtype=tl.float32)

    kv = 0
    while kv < R:
        offs_kv = kv + tl.arange(0, BK)
        kv_mask = offs_kv < R
        kv_ptrs = base + offs_kv[:, None] * D + offs_d[None, :]
        K_blk = tl.load(K + kv_ptrs, mask=kv_mask[:, None] & d_mask[None, :], other=0.0)
        V_blk = tl.load(V + kv_ptrs, mask=kv_mask[:, None] & d_mask[None, :], other=0.0)
        pb = tl.load(PAIR_BIAS + base_pb + offs_q[:, None] * R + offs_kv[None, :],
                     mask=q_mask[:, None] & kv_mask[None, :], other=0.0)
        rm = tl.load(RES_MASK + base_rm + offs_kv, mask=kv_mask, other=0.0)

        sij = scale * tl.dot(Q_blk, tl.trans(K_blk)) + pb + rm[None, :]
        sij = tl.where(kv_mask[None, :], sij, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(sij, axis=1))
        p = tl.exp(sij - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc_o = acc_o * alpha[:, None] + tl.dot(p.to(V_blk.dtype), V_blk)
        m_i = m_new
        kv += BK

    o_i = acc_o / l_i[:, None]
    m_lse = m_i + tl.log(l_i)
    d_i = tl.sum(dO_blk.to(tl.float32) * o_i, axis=1)

    # ---- pass 2: dQ (registers) + atomic dK/dV/d_pair_bias ----
    dq_acc = tl.zeros((BQ, BD), dtype=tl.float32)
    kv = 0
    while kv < R:
        offs_kv = kv + tl.arange(0, BK)
        kv_mask = offs_kv < R
        kv_ptrs = base + offs_kv[:, None] * D + offs_d[None, :]
        K_blk = tl.load(K + kv_ptrs, mask=kv_mask[:, None] & d_mask[None, :], other=0.0)
        V_blk = tl.load(V + kv_ptrs, mask=kv_mask[:, None] & d_mask[None, :], other=0.0)
        pb = tl.load(PAIR_BIAS + base_pb + offs_q[:, None] * R + offs_kv[None, :],
                     mask=q_mask[:, None] & kv_mask[None, :], other=0.0)
        rm = tl.load(RES_MASK + base_rm + offs_kv, mask=kv_mask, other=0.0)

        sij = scale * tl.dot(Q_blk, tl.trans(K_blk)) + pb + rm[None, :]
        p = tl.exp(sij - m_lse[:, None])
        p = tl.where(q_mask[:, None] & kv_mask[None, :], p, 0.0)

        # dV += p^T @ dO
        dv_contrib = tl.dot(tl.trans(p).to(dO_blk.dtype), dO_blk)
        tl.atomic_add(DV + kv_ptrs, dv_contrib.to(tl.float32),
                      mask=kv_mask[:, None] & d_mask[None, :])

        # dP = dO @ V^T ; dS = p * (dP - D_i)
        dp = tl.dot(dO_blk, tl.trans(V_blk))
        ds = p * (dp - d_i[:, None])
        ds = tl.where(q_mask[:, None] & kv_mask[None, :], ds, 0.0)

        # d_pair_bias += dS
        tl.atomic_add(DPB + base_pb + offs_q[:, None] * R + offs_kv[None, :], ds,
                      mask=q_mask[:, None] & kv_mask[None, :])

        # dK += scale * dS^T @ Q
        dk_contrib = scale * tl.dot(tl.trans(ds).to(Q_blk.dtype), Q_blk)
        tl.atomic_add(DK + kv_ptrs, dk_contrib.to(tl.float32),
                      mask=kv_mask[:, None] & d_mask[None, :])

        # dQ += scale * dS @ K
        dq_acc += scale * tl.dot(ds.to(K_blk.dtype), K_blk)
        kv += BK

    tl.store(DQ + q_ptrs, dq_acc, mask=q_mask[:, None] & d_mask[None, :])


def evoattention_backward_naive_triton(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    res_mask: torch.Tensor,
    pair_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (dq, dk, dv, d_pair_bias) for EvoformerAttention backward."""
    if q.ndim != 5:
        raise ValueError(f"expected 5D q [B, N_seq, N_res, Head, Dim], got {tuple(q.shape)}")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("evoattention_backward_naive_triton requires CUDA tensors")

    out_dtype = q.dtype
    dim = q.shape[-1]
    scale = dim ** -0.5

    # Workspace layout [B, S, H, R, D], contiguous.
    q_ws = q.transpose(-2, -3).contiguous()
    k_ws = k.transpose(-2, -3).contiguous()
    v_ws = v.transpose(-2, -3).contiguous()
    do_ws = do.transpose(-2, -3).contiguous()
    B, S, H, R, Dm = q_ws.shape

    res_mask_c = res_mask.contiguous().to(torch.float32)        # [B, S, 1, 1, R]
    pair_bias_c = pair_bias.contiguous().to(torch.float32)      # [B, 1, H, R, R]

    dq = torch.zeros_like(q_ws, dtype=torch.float32)
    dk = torch.zeros_like(k_ws, dtype=torch.float32)
    dv = torch.zeros_like(v_ws, dtype=torch.float32)
    d_pair_bias = torch.zeros((B, 1, H, R, R), device=q.device, dtype=torch.float32)

    BQ, BK = 16, 32
    BD = max(triton.next_power_of_2(Dm), 16)
    grid = (triton.cdiv(R, BQ), B * S * H)

    _evoattn_bwd_kernel[grid](
        q_ws, k_ws, v_ws, res_mask_c, pair_bias_c, do_ws,
        dq, dk, dv, d_pair_bias,
        scale,
        S, H, R, Dm,
        MAT=R * Dm,
        BQ=BQ, BK=BK, BD=BD,
        num_warps=4, num_stages=2,
    )

    dq = dq.transpose(-2, -3).contiguous().to(out_dtype)
    dk = dk.transpose(-2, -3).contiguous().to(out_dtype)
    dv = dv.transpose(-2, -3).contiguous().to(out_dtype)
    d_pair_bias = d_pair_bias.to(pair_bias.dtype)
    return dq, dk, dv, d_pair_bias
