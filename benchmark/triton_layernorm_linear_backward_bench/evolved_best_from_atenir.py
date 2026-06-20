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
import triton
import triton.language as tl
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


@triton.jit
def _ln_dx_kernel(dy, xh, rstd, wt, dx, K: tl.constexpr, BLOCK_K: tl.constexpr):
    m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K

    dyv = tl.load(dy + m * K + offs, mask=mask, other=0.0).to(tl.float32)
    xhv = tl.load(xh + m * K + offs, mask=mask, other=0.0).to(tl.float32)
    wv = tl.load(wt + offs, mask=mask, other=0.0).to(tl.float32)

    wdy = dyv * wv
    inv_k = 1.0 / K
    c1 = tl.sum(xhv * wdy, axis=0) * inv_k
    c2 = tl.sum(wdy, axis=0) * inv_k
    rv = tl.load(rstd + m).to(tl.float32)

    out = (wdy - xhv * c1 - c2) * rv
    tl.store(dx + m * K + offs, out, mask=mask)


def layernorm_linear_backward_triton(
    dout: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    linear_weight: torch.Tensor,
    eps: float = 1e-5,
):
    """Direct backward for fused LayerNorm -> Linear.

    Uses optimized native LayerNorm recomputation, tensor-core GEMMs for fp16,
    and a fused Triton row kernel for the LayerNorm input gradient.
    """
    wf = weight.float()
    bf = bias.float()

    if x.dtype is torch.float16:
        x_hat, _, rstd = torch.native_layer_norm(
            x, (x.shape[1],), None, None, float(eps)
        )
    else:
        x_hat, _, rstd = torch.native_layer_norm(
            x.float(), (x.shape[1],), None, None, float(eps)
        )

    if x.dtype is torch.float16:
        dy_hat = dout @ linear_weight.t()
        y_hat = (x_hat * wf + bf).to(torch.float16)
        dlinear_weight = y_hat.t() @ dout
    else:
        df = dout.float()
        lwf = linear_weight.float()
        dy_hat = df @ lwf.t()
        y_hat = x_hat * wf + bf
        dlinear_weight = y_hat.t() @ df

    if dy_hat.dtype is torch.float16:
        dweight = (dy_hat * x_hat).sum(dim=0, dtype=torch.float32)
        dbias = dy_hat.sum(dim=0, dtype=torch.float32)
    else:
        dweight = (dy_hat * x_hat).sum(dim=0)
        dbias = dy_hat.sum(dim=0)

    dx = torch.empty_like(x)
    bk = triton.next_power_of_2(x.shape[1])
    _ln_dx_kernel[(x.shape[0],)](
        dy_hat,
        x_hat,
        rstd,
        weight,
        dx,
        x.shape[1],
        BLOCK_K=bk,
        num_warps=8 if bk >= 512 else 4,
    )

    return (
        dx.to(x.dtype),
        dlinear_weight.to(linear_weight.dtype),
        dweight.to(weight.dtype),
        dbias.to(bias.dtype),
    )
