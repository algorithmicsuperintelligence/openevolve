"""LayerNorm bench glue between the bench's frozen API and the AtenIR runtime.

The bench's frozen API is::

    layernorm_backward_triton(dy, x, weight, bias, eps) -> (dx, dweight, dbias)

The AtenIR graph extracted from ``aten.native_layer_norm_backward.default``
expects ``mean`` and ``rstd`` as explicit placeholders (because that aten op's
signature does), but the bench never passes them.  Everything that bridges
the two — and that is therefore LayerNorm-specific — lives here:

* recompute ``mean`` and ``rstd`` from ``x`` in fp32;
* cast all inputs up to fp32 before dispatch, then cast outputs back to each
  input's original dtype;
* override the integer literal ``N=256`` baked into ``mul_2`` / ``div``
  ``scalar_args`` at trace time, swapping it for the runtime ``cols`` dim.
  This is a workaround for shape-specialised constants captured by
  ``make_fx``; the proper long-term fix is symbolic-mode extraction (see
  ``atenir/README.md``).

``atenir.compose`` knows none of this — it just walks the graph.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Callable

import torch

from atenir.compose import run_graph

_HERE = Path(__file__).resolve().parent
_KERNELS_DIR = _HERE / "atenir" / "primitive_kernels"
if str(_KERNELS_DIR) not in sys.path:
    sys.path.insert(0, str(_KERNELS_DIR))

_GRAPH_PATH = _HERE / "atenir" / "layernorm_bwd_graph.json"
_GRAPH = json.loads(_GRAPH_PATH.read_text())

_REGISTRY: dict[str, Callable] = {}
for _node in _GRAPH["nodes"]:
    if _node.get("op") == "call_function":
        _name = _node["name"]
        _REGISTRY[_name] = importlib.import_module(_name).run


def layernorm_backward_triton(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
):
    orig_x_dtype = x.dtype
    orig_w_dtype = weight.dtype
    orig_b_dtype = bias.dtype

    x_f32 = x.float().contiguous()
    dy_f32 = dy.float().contiguous()
    weight_f32 = weight.float().contiguous()
    bias_f32 = bias.float().contiguous()

    mean = x_f32.mean(dim=-1, keepdim=True)
    var = ((x_f32 - mean) ** 2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)

    env = {
        "grad_out_1": dy_f32,
        "x_1": x_f32,
        "mean_1": mean.contiguous(),
        "rstd_1": rstd.contiguous(),
        "weight_1": weight_f32,
        "bias_1": bias_f32,
    }
    # N (= feature dim) was baked into mul_2/div as a constant at trace time;
    # swap it for the runtime cols dim until symbolic-mode extraction lands.
    overrides = {"mul_2": [x.shape[-1]], "div": [x.shape[-1]]}

    dx, dweight, dbias = run_graph(str(_GRAPH_PATH), env, _REGISTRY, overrides)
    return (
        dx.to(orig_x_dtype),
        dweight.to(orig_w_dtype),
        dbias.to(orig_b_dtype),
    )
