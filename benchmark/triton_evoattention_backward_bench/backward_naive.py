"""Naive (materialized) EvoformerAttention backward baseline.

This is the deliberately slow, easy-to-verify seed for OpenEvolve. It recomputes
the full attention probability matrix ``P`` (shape [B, N_seq, Head, N_res, N_res])
and applies the textbook attention-backward formulas with plain PyTorch ops in
float32. Materializing ``P``/``dP`` is exactly the memory bottleneck that
MegaFold's fused flash-style Triton kernel removes, so this baseline leaves large
headroom for evolution.

The evolve target is to replace this with a fused, flash-attention-style Triton
backward (no [N_res, N_res] materialization), matching the gradients below.

Math (per batch/msa/head, attention contracts over the residue axis):

    S      = scale * Q @ K^T + pair_bias + res_mask
    P      = softmax(S, axis=key)
    dV     = P^T @ dO
    dP     = dO @ V^T
    dS     = P * (dP - rowsum(dP * P))
    dQ     = scale * dS @ K
    dK     = scale * dS^T @ Q
    d_pair = sum over the N_seq axis of dS
"""

import torch


def evoattention_backward_naive(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    res_mask: torch.Tensor,
    pair_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (dq, dk, dv, d_pair_bias) for EvoformerAttention.

    Inputs use the [B, N_seq, N_res, Head, Dim] layout; res_mask is additive
    [B, N_seq, 1, 1, N_res]; pair_bias is [B, 1, Head, N_res, N_res].
    """
    if q.ndim != 5:
        raise ValueError(f"expected 5D q [B, N_seq, N_res, Head, Dim], got {tuple(q.shape)}")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("evoattention_backward_naive requires CUDA tensors")

    out_dtype = q.dtype
    dim = q.shape[-1]
    scale = dim ** -0.5

    # Work in the [B, N_seq, Head, N_res, Dim] attention layout, float32.
    q_t = q.transpose(-2, -3).float()
    k_t = k.transpose(-2, -3).float()
    v_t = v.transpose(-2, -3).float()
    do_t = do.transpose(-2, -3).float()

    scores = torch.matmul(q_t * scale, k_t.transpose(-1, -2))  # [B, S, H, R, R]
    scores = scores + pair_bias.float() + res_mask.float()
    probs = torch.softmax(scores, dim=-1)  # over key residue axis

    # dV = P^T @ dO
    dv_t = torch.matmul(probs.transpose(-1, -2), do_t)  # [B, S, H, R, Dim]

    # dP = dO @ V^T ; dS = P * (dP - rowsum(dP * P))
    dp = torch.matmul(do_t, v_t.transpose(-1, -2))  # [B, S, H, R, R]
    ds = probs * (dp - (dp * probs).sum(dim=-1, keepdim=True))

    # dQ = scale * dS @ K ; dK = scale * dS^T @ Q
    dq_t = scale * torch.matmul(ds, k_t)
    dk_t = scale * torch.matmul(ds.transpose(-1, -2), q_t)

    # pair_bias is broadcast over the N_seq axis -> sum the per-msa grads.
    d_pair_bias = ds.sum(dim=1, keepdim=True)  # [B, 1, H, R, R]

    dq = dq_t.transpose(-2, -3).contiguous().to(out_dtype)
    dk = dk_t.transpose(-2, -3).contiguous().to(out_dtype)
    dv = dv_t.transpose(-2, -3).contiguous().to(out_dtype)
    d_pair_bias = d_pair_bias.contiguous().to(pair_bias.dtype)
    return dq, dk, dv, d_pair_bias
