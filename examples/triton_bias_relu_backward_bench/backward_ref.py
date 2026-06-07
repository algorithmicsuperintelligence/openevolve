"""PyTorch autograd oracle for bias + ReLU backward."""

import torch

try:
    from forward_ref import bias_relu_forward_ref
except ImportError:  # pragma: no cover - supports package-style imports
    from .forward_ref import bias_relu_forward_ref


def bias_relu_backward_ref(
    dy: torch.Tensor,
    x: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return reference gradients (dx, dbias) using PyTorch autograd."""
    x_ref = x.detach().clone().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)
    y_ref = bias_relu_forward_ref(x_ref, bias_ref)
    y_ref.backward(dy.detach().clone())
    if x_ref.grad is None or bias_ref.grad is None:
        raise RuntimeError("PyTorch autograd did not produce expected gradients")
    return x_ref.grad, bias_ref.grad
