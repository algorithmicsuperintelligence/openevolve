"""AtenIR lowering of the fused LayerNorm -> Linear backward.

This bridges the bench's frozen API::

    layernorm_linear_backward_triton(dout, x, weight, bias, linear_weight, eps)
        -> (dx, dlinear_weight, dweight, dbias)

to the op-agnostic AtenIR runtime (``atenir.compose.run_graph``) + the general
primitive-Triton dispatch (``atenir.primitive_triton.dispatch.make_registry``).

The backward graph is produced by autograd-tracing a *primitive* LayerNorm ->
Linear forward (explicit mean/var, NOT the fused ``aten.native_layer_norm`` /
``aten.var_mean`` ops, whose required non-tensor args the generic dispatch cannot
feed). ``make_fx`` specialises the graph to the traced shape, so — following the
"symbolic-mode extraction" fix described in ``atenir/README.md`` — the adapter
extracts (and caches) one graph per (M, K, N, eps) signature instead of relying on
shape-specialised scalar overrides. A reference graph at one shape is committed at
``atenir/layernorm_linear_bwd_graph.json`` for inspection.

The graph is traced/run in fp32; inputs are up-cast and outputs cast back.
"""

from __future__ import annotations

import functools
import json
import os
import tempfile

import torch
from torch._decomp import core_aten_decompositions
from torch.fx.experimental.proxy_tensor import make_fx

from atenir.compose import run_graph
from atenir.extract import _serialise
from atenir.primitive_triton.dispatch import make_registry

_DECOMPS = dict(core_aten_decompositions())
_CACHE_DIR = os.path.join(tempfile.gettempdir(), "atenir_graphs_layernorm_linear")


def _ln_linear_forward(x, weight, bias, linear_weight, eps):
    """Primitive (un-fused) LayerNorm -> Linear forward."""
    mean = x.mean(dim=-1, keepdim=True)
    centered = x - mean
    var = (centered * centered).mean(dim=-1, keepdim=True)
    x_hat = centered * torch.rsqrt(var + eps)
    return (x_hat * weight + bias) @ linear_weight


@functools.lru_cache(maxsize=None)
def _compiled(M: int, K: int, N: int, eps: float):
    ex = [
        torch.randn(M, K, device="cuda"),
        torch.randn(K, device="cuda"),
        torch.randn(K, device="cuda"),
        torch.randn(K, N, device="cuda"),
    ]
    fwd = lambda x, w, b, lw: _ln_linear_forward(x, w, b, lw, eps)  # noqa: E731

    with torch.no_grad():
        sample = fwd(*ex)
    grad_out = torch.randn_like(sample)

    def bwd(grad_out, *fwd_inputs):
        ins = [t.detach().requires_grad_(True) for t in fwd_inputs]
        return torch.autograd.grad(fwd(*ins), ins, grad_outputs=grad_out)

    gm = make_fx(bwd, decomposition_table=_DECOMPS)(grad_out, *ex)
    graph = _serialise(gm)

    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, f"ln_linear_{M}_{K}_{N}_{eps!r}.json")
    with open(path, "w") as f:
        json.dump(graph, f)

    placeholders = [n["name"] for n in graph["nodes"] if n["op"] == "placeholder"]
    return path, make_registry(graph), placeholders


def layernorm_linear_backward_triton(
    dout: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    linear_weight: torch.Tensor,
    eps: float = 1e-5,
):
    """Compute (dx, dlinear_weight, dweight, dbias) via the AtenIR graph."""
    M, K = x.shape
    N = linear_weight.shape[1]
    path, registry, ph = _compiled(M, K, N, float(eps))

    # Placeholder order: (grad_out, x, weight, bias, linear_weight).
    env = {
        ph[0]: dout.float().contiguous(),
        ph[1]: x.float().contiguous(),
        ph[2]: weight.float().contiguous(),
        ph[3]: bias.float().contiguous(),
        ph[4]: linear_weight.float().contiguous(),
    }
    # Output order from autograd-mode extraction: (dx, dweight, dbias, dlinear_weight).
    dx, dweight, dbias, dlinear_weight = run_graph(path, env, registry)
    return (
        dx.to(x.dtype),
        dlinear_weight.to(linear_weight.dtype),
        dweight.to(weight.dtype),
        dbias.to(bias.dtype),
    )
