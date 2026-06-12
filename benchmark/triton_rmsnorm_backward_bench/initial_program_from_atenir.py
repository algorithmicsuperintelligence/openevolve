"""RMSNorm backward seed for OpenEvolve, composed from the AtenIR graph.

Unlike ``initial_program.py`` (which delegates to a hand-written naive kernel),
this seed runs the *automatically extracted* ATen backward graph
(``atenir/rmsnorm_bwd_graph.json``, produced by ``python -m atenir.extract``)
through the op-agnostic AtenIR runtime (``atenir.compose.run_graph``) + the
general primitive-Triton dispatch (``atenir.primitive_triton.dispatch``).

It is the end-to-end AtenIR story: a naive, auto-lowered composition of primitive
Triton kernels that OpenEvolve is meant to fuse/optimize. There are no
EVOLVE-BLOCK markers, so OpenEvolve evolves the whole file.

Notes / caveats:
* ``make_fx`` baked trace-time constants into the graph (the reduction size N and
  the broadcast/reshape shapes, plus eps). They are restored to runtime values
  via ``scalar_overrides`` below, so the kernel generalizes across shapes.
* The graph is traced in fp32; inputs are up-cast to fp32 and outputs cast back,
  which keeps the primitive reductions numerically stable on fp16 inputs.
"""

import json
import os

import torch

from atenir.compose import run_graph
from atenir.primitive_triton.dispatch import make_registry


def _find_graph_path() -> str:
    """Locate rmsnorm_bwd_graph.json next to this bench, robust to temp-dir runs."""
    here = os.path.dirname(os.path.abspath(__file__))
    local = os.path.join(here, "atenir", "rmsnorm_bwd_graph.json")
    if os.path.exists(local):
        return local
    # OpenEvolve may execute the candidate from a checkpoint/temp dir; fall back
    # to the installed bench package location (repo root is on sys.path via the
    # evaluator).
    import benchmark.triton_rmsnorm_backward_bench as _pkg

    return os.path.join(os.path.dirname(_pkg.__file__), "atenir", "rmsnorm_bwd_graph.json")


_GRAPH_PATH = _find_graph_path()
with open(_GRAPH_PATH) as _f:
    _GRAPH = json.load(_f)
# Placeholder order from autograd-mode extraction: (grad_out, x, weight).
_PLACEHOLDERS = [n["name"] for n in _GRAPH["nodes"] if n["op"] == "placeholder"]
_REGISTRY = make_registry(_GRAPH)


def rmsnorm_backward_triton(dy, x, weight, eps=1e-5):
    """Compute (dx, dweight) for row-wise RMSNorm via the AtenIR graph."""
    orig_x_dtype = x.dtype
    orig_w_dtype = weight.dtype

    dy32 = dy.float().contiguous()
    x32 = x.float().contiguous()
    w32 = weight.float().contiguous()

    env = {
        _PLACEHOLDERS[0]: dy32,
        _PLACEHOLDERS[1]: x32,
        _PLACEHOLDERS[2]: w32,
    }

    rows, cols = x.shape
    # Restore shape/size constants that make_fx specialized at trace time.
    overrides = {
        "add": [float(eps)],          # eps (forward default captured at trace time)
        "div": [cols],                # N for the mean-derivative term
        "expand": [[rows, cols]],     # broadcast-to shape
        "view": [[cols]],             # dweight reshape [1, N] -> [N]
    }

    dx, dweight = run_graph(_GRAPH_PATH, env, _REGISTRY, overrides)
    return dx.to(orig_x_dtype), dweight.to(orig_w_dtype)
