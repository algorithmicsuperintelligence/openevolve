"""Optional Liger LayerNorm baseline for post-OpenEvolve benchmarking."""

from __future__ import annotations

from typing import Callable

import torch


def liger_available() -> bool:
    try:
        import liger_kernel.ops.layer_norm  # noqa: F401

        return True
    except ImportError:
        return False


def make_liger_layernorm_backward_fn() -> Callable:
    """Standalone backward with the same API as layernorm_backward_triton."""
    from liger_kernel.ops.layer_norm import LigerLayerNormFunction

    def layernorm_backward_liger(
        dy: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_req = x.detach().clone().requires_grad_(True)
        weight_req = weight.detach().clone().requires_grad_(True)
        bias_req = bias.detach().clone().requires_grad_(True)
        y = LigerLayerNormFunction.apply(x_req, weight_req, bias_req, eps)
        return torch.autograd.grad(y, (x_req, weight_req, bias_req), dy)

    return layernorm_backward_liger


def make_liger_layernorm_train_step_fn(eps: float):
    """Forward + backward training step using LigerLayerNormFunction."""
    from liger_kernel.ops.layer_norm import LigerLayerNormFunction

    def train_step(
        x_req: torch.Tensor,
        weight_req: torch.Tensor,
        bias_req: torch.Tensor,
        dy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for tensor in (x_req, weight_req, bias_req):
            tensor.grad = None
        y = LigerLayerNormFunction.apply(x_req, weight_req, bias_req, eps)
        loss = torch.sum(y * dy)
        loss.backward()
        return x_req.grad, weight_req.grad, bias_req.grad

    return train_step
