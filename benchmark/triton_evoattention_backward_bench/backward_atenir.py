"""AtenIR lowering of the EvoformerAttention (3D EvoAttention) backward.

This bridges the bench's frozen API::

    evoattention_backward_triton(do, q, k, v, res_mask, pair_bias)
        -> (dq, dk, dv, d_pair_bias)

to the op-agnostic AtenIR runtime (``atenir.compose.run_graph``) + the general
primitive-Triton dispatch (``atenir.primitive_triton.dispatch.make_registry``).

The backward graph is produced by autograd-tracing the reference attention forward
(scale*Q@K^T + pair_bias + res_mask -> softmax -> @V). The softmax and its backward
decompose into primitives (amax/exp/sum/div, mul/sub/sum); the batched matmuls
become ``aten.bmm`` (dispatched to ``torch.bmm``). res_mask is a forward input, so
the traced graph also produces ``d_res_mask`` — which the bench discards (the mask
is additive and not a learnable parameter).

``make_fx`` specialises the 5D ``view``/``expand`` reshapes to the traced shape, so
— following the "symbolic-mode extraction" fix described in ``atenir/README.md`` —
the adapter extracts (and caches) one graph per (B, S, H, R, D) signature rather
than overriding the many shape-specialised reshape constants. A reference graph at
one shape is committed at ``atenir/evoattention_bwd_graph.json`` for inspection.

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
_CACHE_DIR = os.path.join(tempfile.gettempdir(), "atenir_graphs_evoattention")


def _evoattention_forward(q, k, v, res_mask, pair_bias):
    """Reference EvoformerAttention forward (primitive matmul + softmax)."""
    scale = q.shape[-1] ** -0.5
    q_t = q.transpose(-2, -3)
    k_t = k.transpose(-2, -3)
    v_t = v.transpose(-2, -3)
    scores = torch.matmul(q_t * scale, k_t.transpose(-1, -2)) + pair_bias + res_mask
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v_t).transpose(-2, -3).contiguous()


@functools.lru_cache(maxsize=None)
def _compiled(B: int, S: int, H: int, R: int, D: int):
    ex = [torch.randn(B, S, R, H, D, device="cuda") for _ in range(3)]
    ex.append(torch.zeros(B, S, 1, 1, R, device="cuda"))   # res_mask
    ex.append(torch.randn(B, 1, H, R, R, device="cuda"))   # pair_bias

    with torch.no_grad():
        sample = _evoattention_forward(*ex)
    grad_out = torch.randn_like(sample)

    def bwd(grad_out, *fwd_inputs):
        ins = [t.detach().requires_grad_(True) for t in fwd_inputs]
        return torch.autograd.grad(_evoattention_forward(*ins), ins, grad_outputs=grad_out)

    gm = make_fx(bwd, decomposition_table=_DECOMPS)(grad_out, *ex)
    graph = _serialise(gm)

    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, f"evoattn_{B}_{S}_{H}_{R}_{D}.json")
    with open(path, "w") as f:
        json.dump(graph, f)

    placeholders = [n["name"] for n in graph["nodes"] if n["op"] == "placeholder"]
    return path, make_registry(graph), placeholders


def evoattention_backward_triton(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    res_mask: torch.Tensor,
    pair_bias: torch.Tensor,
):
    """Compute (dq, dk, dv, d_pair_bias) via the AtenIR graph."""
    B, S, R, H, D = q.shape
    path, registry, ph = _compiled(B, S, H, R, D)

    # Placeholder order: (grad_out, q, k, v, res_mask, pair_bias).
    env = {
        ph[0]: do.float().contiguous(),
        ph[1]: q.float().contiguous(),
        ph[2]: k.float().contiguous(),
        ph[3]: v.float().contiguous(),
        ph[4]: res_mask.float().contiguous(),
        ph[5]: pair_bias.float().contiguous(),
    }
    # Output order from autograd-mode extraction: (dq, dk, dv, d_res_mask, d_pair_bias).
    dq, dk, dv, _d_res_mask, d_pair_bias = run_graph(path, env, registry)
    return (
        dq.to(q.dtype),
        dk.to(k.dtype),
        dv.to(v.dtype),
        d_pair_bias.to(pair_bias.dtype),
    )
