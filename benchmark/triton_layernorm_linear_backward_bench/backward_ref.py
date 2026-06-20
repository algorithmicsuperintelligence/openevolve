"""PyTorch autograd oracle for the fused LayerNorm -> Linear backward.

Runs autograd over ``layernorm_linear_forward_ref`` to produce the reference
gradients (dx, dlinear_weight, dweight, dbias). There is no linear bias (matching
the AF3 Transition path), so the upstream linear has no bias gradient.
"""

import torch

try:
    from forward_ref import layernorm_linear_forward_ref
except ImportError:  # pragma: no cover - supports package-style imports
    from .forward_ref import layernorm_linear_forward_ref


def layernorm_linear_backward_ref(
    dout: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    linear_weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return reference gradients (dx, dlinear_weight, dweight, dbias) via autograd."""
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)
    lw_ref = linear_weight.detach().clone().requires_grad_(True)

    out = layernorm_linear_forward_ref(x_ref, weight_ref, bias_ref, lw_ref, eps)
    out.backward(dout.detach().clone())

    if any(t.grad is None for t in (x_ref, lw_ref, weight_ref, bias_ref)):
        raise RuntimeError("PyTorch autograd did not produce expected gradients")
    return x_ref.grad, lw_ref.grad, weight_ref.grad, bias_ref.grad
