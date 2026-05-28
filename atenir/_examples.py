"""Example forward callables used to exercise atenir.extract --fn (autograd mode).

All forwards take only Tensor arguments (the autograd-driven mode treats every
positional input as a tensor placeholder); any scalar hyperparameters are baked
in as Python constants.
"""

import torch


def square_sum(x: torch.Tensor) -> torch.Tensor:
    """y = sum(x * x, dim=-1)."""
    return (x * x).sum(dim=-1)


def attention_block(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Manual scaled dot-product attention so make_fx sees the decomposition,
    not the fused aten.scaled_dot_product_attention op."""
    scale = 1.0 / (q.shape[-1] ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """RMSNorm with eps baked in as a constant."""
    eps = 1e-5
    rstd = torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps)
    return (x * rstd) * weight


def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """LayerNorm over the last dimension with eps baked in as a constant."""
    eps = 1e-5
    mean = x.float().mean(dim=-1, keepdim=True)
    var = ((x.float() - mean) ** 2).mean(dim=-1, keepdim=True)
    xhat = (x.float() - mean) * torch.rsqrt(var + eps)
    return (xhat * weight.float() + bias.float()).to(x.dtype)


def topk_gather(x: torch.Tensor) -> torch.Tensor:
    """Top-k indices then gather — MoE routing-shaped pattern.  k baked in."""
    k = 4
    _, idx = torch.topk(x, k=k, dim=-1)
    return torch.gather(x, dim=-1, index=idx)


def swiglu(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor) -> torch.Tensor:
    """SwiGLU block: silu(x @ w_gate) * (x @ w_up).

    Example shapes:
      x      : [B, D]
      w_gate : [D, H]
      w_up   : [D, H]
      output : [B, H]
    """
    gate = torch.matmul(x, w_gate)
    up = torch.matmul(x, w_up)
    return torch.nn.functional.silu(gate) * up


def mlp(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """Two-layer MLP with GELU activation.

    Example shapes:
      x  : [B, D]
      w1 : [D, H]
      w2 : [H, O]
      output : [B, O]
    """
    hidden = torch.matmul(x, w1)
    hidden = torch.nn.functional.gelu(hidden)
    return torch.matmul(hidden, w2)
