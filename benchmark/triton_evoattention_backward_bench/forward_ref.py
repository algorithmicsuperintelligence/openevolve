"""PyTorch reference for AlphaFold3-style EvoformerAttention (3D EvoAttention).

This mirrors MegaFold's ``EvoformerAttention`` semantics (arXiv:2506.20686). It is
flash-attention over a 5D tensor with an extra MSA (``N_seq``) axis, a trainable
pairwise bias broadcast over that axis, and an additive per-key residue mask.

Tensor layout (candidate-facing):

    q, k, v:   [B, N_seq, N_res, Head, Dim]
    res_mask:  [B, N_seq, 1, 1, N_res]     additive (0 keep, large-negative drop)
    pair_bias: [B, 1, Head, N_res, N_res]  broadcast over N_seq, trainable
    output:    [B, N_seq, N_res, Head, Dim]

Internally the attention contracts over the residue axis ``N_res`` per (batch,
msa, head), so q/k/v are transposed to [B, N_seq, Head, N_res, Dim] first.
"""

import torch


def evoattention_forward_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    res_mask: torch.Tensor,
    pair_bias: torch.Tensor,
) -> torch.Tensor:
    """Compute EvoformerAttention output in the [B, N_seq, N_res, Head, Dim] layout.

    softmax(scale * Q @ K^T + pair_bias + res_mask) @ V, with the softmax taken
    over the key residue axis in float32 for numerical stability.
    """
    if q.ndim != 5:
        raise ValueError(f"expected 5D q [B, N_seq, N_res, Head, Dim], got {tuple(q.shape)}")
    dtype = q.dtype
    dim = q.shape[-1]
    scale = dim ** -0.5

    # [B, N_seq, N_res, Head, Dim] -> [B, N_seq, Head, N_res, Dim]
    q_t = q.transpose(-2, -3)
    k_t = k.transpose(-2, -3)
    v_t = v.transpose(-2, -3)

    scores = torch.matmul(q_t * scale, k_t.transpose(-1, -2))  # [B, N_seq, Head, N_res, N_res]
    scores = scores + pair_bias + res_mask
    probs = torch.softmax(scores.float(), dim=-1).to(dtype)
    out_t = torch.matmul(probs, v_t)  # [B, N_seq, Head, N_res, Dim]
    return out_t.transpose(-2, -3).contiguous()  # back to [B, N_seq, N_res, Head, Dim]
