"""Autograd-pair LayerNorm seed with an evolvable saved-tensor contract.

Public contract:

    layernorm_forward_with_saved(x, weight, bias, eps) -> (y, saved_tensors)
    layernorm_backward_from_saved(dy, saved_tensors, eps) -> (dx, dweight, dbias)

The initial seed is deliberately conservative: it only saves the original
forward inputs needed by a standalone backward.  OpenEvolve is expected to
modify the EVOLVE-BLOCK if saving additional forward intermediates improves the
forward+backward tradeoff.
"""

from __future__ import annotations

import torch


def layernorm_forward_torch(x, weight, bias, eps=1e-5):
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = ((xf - mean) * (xf - mean)).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + float(eps))
    y = ((xf - mean) * rstd * weight.float() + bias.float()).to(x.dtype)
    return y


def layernorm_backward_torch(dy, x, weight, bias, eps=1e-5):
    xf = x.float()
    dyf = dy.float()
    wf = weight.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = ((xf - mean) * (xf - mean)).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + float(eps))
    xhat = (xf - mean) * rstd
    g = dyf * wf
    dx = (g - g.mean(dim=-1, keepdim=True) - xhat * (g * xhat).mean(dim=-1, keepdim=True)) * rstd.float()
    dweight = (dyf * xhat).sum(dim=0).to(weight.dtype)
    dbias = dyf.sum(dim=0).to(bias.dtype)
    return dx.to(x.dtype), dweight, dbias


# EVOLVE-BLOCK-START
from backward_naive_triton import layernorm_backward_naive_triton as _seed_backward
from forward_triton import layernorm_forward_triton as _seed_forward


def _forward_with_saved_impl(x, weight, bias, eps=1e-5):
    """Initial evolvable forward: save only original inputs.

    OpenEvolve may add forward-computed intermediates to this tuple and update
    ``_backward_from_saved_impl`` to consume them.
    """
    y = _seed_forward(x, weight, bias, eps)
    return y, (x.contiguous(), weight.contiguous(), bias.contiguous())


def _backward_from_saved_impl(dy, saved_tensors, eps=1e-5):
    """Initial evolvable backward: standalone backward over saved inputs."""
    x, weight, bias = saved_tensors
    return _seed_backward(dy, x, weight, bias, eps)
# EVOLVE-BLOCK-END


def layernorm_forward_with_saved(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
):
    """Return forward output plus an evolvable saved tensor tuple."""
    if not (x.is_cuda and weight.is_cuda and bias.is_cuda):
        y = layernorm_forward_torch(x, weight, bias, eps)
        return y, (x, weight, bias)
    return _forward_with_saved_impl(x, weight, bias, eps)


def layernorm_backward_from_saved(
    dy: torch.Tensor,
    saved_tensors,
    eps: float = 1e-5,
):
    """Consume saved tensors and return ``dx, dweight, dbias``."""
    x, weight, bias = saved_tensors[:3]
    if not (dy.is_cuda and x.is_cuda and weight.is_cuda and bias.is_cuda):
        return layernorm_backward_torch(dy, x, weight, bias, eps)
    return _backward_from_saved_impl(dy, saved_tensors, eps)
