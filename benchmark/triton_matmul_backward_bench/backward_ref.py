"""PyTorch autograd oracle for matmul backward.

Gradients of c = a @ b:
    da = dc @ b.T     [M, K]
    db = a.T @ dc     [K, N]
"""

import torch


def matmul_backward_ref(
    dc: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return reference gradients (da, db) using PyTorch autograd."""
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    c_ref = a_ref @ b_ref
    c_ref.backward(dc.detach().clone())
    if a_ref.grad is None or b_ref.grad is None:
        raise RuntimeError("PyTorch autograd did not produce expected gradients")
    return a_ref.grad, b_ref.grad
