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


def silu_mlp(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """Two-layer MLP with SiLU (Swish) activation — common in LLaMA/Mistral FFN layers.

    Example shapes:
      x  : [B, D]
      w1 : [D, H]
      w2 : [H, O]
      output : [B, O]
    """
    hidden = torch.matmul(x, w1)
    hidden = torch.nn.functional.silu(hidden)
    return torch.matmul(hidden, w2)


def mobilenet_block(x: torch.Tensor) -> torch.Tensor:
    """HardSwish activation block — standard in MobileNetV3.

    HardSwish = x * clamp((x + 3) / 6, 0, 1).
    """
    return torch.nn.functional.hardswish(x)


def mish_block(x: torch.Tensor) -> torch.Tensor:
    """Mish activation: x * tanh(softplus(x)).

    Softplus + tanh path, both dispatched to Triton.
    """
    return x * torch.tanh(torch.nn.functional.softplus(x))


def lerp_blend(a: torch.Tensor, b: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Element-wise linear interpolation: a + weight * (b - a).

    Common in EMA updates, diffusion-model blending, and style transfer.
    """
    return torch.lerp(a, b, weight)


def erfc_gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU via erfc: 0.5 * x * erfc(-x / sqrt(2)).

    Alternative GELU formula that exercises the erfc kernel instead of erf.
    """
    return 0.5 * x * torch.special.erfc(-x * 0.7071067811865476)


def exp2_scale(x: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
    """Element-wise 2^exponent scaling applied to x.

    Shows exp2 kernel usage; common in fixed-point and quantisation-aware ops.
    """
    return x * torch.exp2(exponent)
