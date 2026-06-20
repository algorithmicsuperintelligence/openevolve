"""PyTorch autograd oracle for EvoformerAttention backward.

Runs autograd over ``evoattention_forward_ref`` to produce the reference
gradients (dq, dk, dv, d_pair_bias). The residue mask is additive and carries no
gradient. ``pair_bias`` is broadcast over the N_seq axis, so its gradient is the
sum over that axis (shape [B, 1, Head, N_res, N_res]).
"""

import torch

try:
    from forward_ref import evoattention_forward_ref
except ImportError:  # pragma: no cover - supports package-style imports
    from .forward_ref import evoattention_forward_ref


def evoattention_backward_ref(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    res_mask: torch.Tensor,
    pair_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return reference gradients (dq, dk, dv, d_pair_bias) via PyTorch autograd."""
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    pair_bias_ref = pair_bias.detach().clone().requires_grad_(True)

    out = evoattention_forward_ref(q_ref, k_ref, v_ref, res_mask, pair_bias_ref)
    out.backward(do.detach().clone())

    if any(t.grad is None for t in (q_ref, k_ref, v_ref, pair_bias_ref)):
        raise RuntimeError("PyTorch autograd did not produce expected gradients")
    return q_ref.grad, k_ref.grad, v_ref.grad, pair_bias_ref.grad
