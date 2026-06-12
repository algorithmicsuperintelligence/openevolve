"""PyTorch autograd oracle for Linear backward.

Gradients of y = x @ weight.T + bias:
    dx     = dy @ weight        [M, K]
    dweight = dy.T @ x          [N, K]
    dbias   = sum(dy, dim=0)    [N]

The bias value does not affect any gradient (dbias = sum(dy)), so the oracle
synthesizes a zero bias purely to let autograd produce dbias.
"""

import torch

try:
    from forward_ref import linear_forward_ref
except ImportError:  # pragma: no cover - supports package-style imports
    from .forward_ref import linear_forward_ref


def linear_backward_ref(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return reference gradients (dx, dweight, dbias) using PyTorch autograd."""
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    bias_ref = torch.zeros(weight.shape[0], device=x.device, dtype=x.dtype, requires_grad=True)
    y_ref = linear_forward_ref(x_ref, weight_ref, bias_ref)
    y_ref.backward(dy.detach().clone())
    if x_ref.grad is None or weight_ref.grad is None or bias_ref.grad is None:
        raise RuntimeError("PyTorch autograd did not produce expected gradients")
    return x_ref.grad, weight_ref.grad, bias_ref.grad
