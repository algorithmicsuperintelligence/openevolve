"""PyTorch autograd oracle for row-wise LayerNorm backward."""

import torch

try:
    from forward_ref import layernorm_forward_ref
except ImportError:  # pragma: no cover - supports package-style imports
    from .forward_ref import layernorm_forward_ref


def layernorm_backward_ref(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return reference gradients (dx, dweight, dbias) using PyTorch autograd."""
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)
    y_ref = layernorm_forward_ref(x_ref, weight_ref, bias_ref, eps)
    y_ref.backward(dy.detach().clone())
    if x_ref.grad is None or weight_ref.grad is None or bias_ref.grad is None:
        raise RuntimeError("PyTorch autograd did not produce expected gradients")
    return x_ref.grad, weight_ref.grad, bias_ref.grad
