"""Linear backward seed for OpenEvolve, composed from the AtenIR graph.

Unlike ``initial_program.py`` (which delegates to a hand-written naive tiled
matmul), this seed runs the *automatically extracted* ATen backward graph
(``atenir/linear_bwd_graph.json``, produced by ``python -m atenir.extract``)
through the op-agnostic AtenIR runtime (``atenir.compose.run_graph``) + the
general primitive-Triton dispatch (``atenir.primitive_triton.dispatch``).

It is the end-to-end AtenIR story for a compute-bound op. There are no
EVOLVE-BLOCK markers, so OpenEvolve evolves the whole file.

IMPORTANT caveat (main branch): the general dispatch currently routes
``aten.mm`` -> ``torch.mm`` and ``aten.addmm`` -> the torch fallback (i.e.
cuBLAS), NOT a Triton GEMM. The Triton GEMM primitive lives on the
``atenir-verification`` branch (``atenir/primitive_triton/gemm.py``). So on main
this seed is correct but its matmuls run on cuBLAS; it becomes a true
primitive-Triton seed once that GEMM primitive is on main. The autograd-traced
graph also recomputes the forward (an ``addmm`` node) whose result is unused by
the gradients — extra work OpenEvolve can prune.

The graph is traced in fp32; inputs are up-cast to fp32 and outputs cast back.
"""

import json
import os

import torch

from atenir.compose import run_graph
from atenir.primitive_triton.dispatch import make_registry


def _find_graph_path() -> str:
    """Locate linear_bwd_graph.json next to this bench, robust to temp-dir runs."""
    here = os.path.dirname(os.path.abspath(__file__))
    local = os.path.join(here, "atenir", "linear_bwd_graph.json")
    if os.path.exists(local):
        return local
    import benchmark.triton_linear_backward_bench as _pkg

    return os.path.join(os.path.dirname(_pkg.__file__), "atenir", "linear_bwd_graph.json")


_GRAPH_PATH = _find_graph_path()
with open(_GRAPH_PATH) as _f:
    _GRAPH = json.load(_f)
# Placeholder order from autograd-mode extraction: (grad_out, x, weight, bias).
_PLACEHOLDERS = [n["name"] for n in _GRAPH["nodes"] if n["op"] == "placeholder"]
_REGISTRY = make_registry(_GRAPH)


def linear_backward_triton(dy, x, weight):
    """Compute (dx, dweight, dbias) for a Linear layer via the AtenIR graph."""
    orig_x_dtype = x.dtype
    orig_w_dtype = weight.dtype

    dy32 = dy.float().contiguous()
    x32 = x.float().contiguous()
    w32 = weight.float().contiguous()
    N = weight.shape[0]
    # bias is a forward placeholder in the autograd-traced graph; its value does
    # not affect any gradient (dbias = sum(dy)), so a zero bias is sufficient.
    b32 = torch.zeros(N, device=x.device, dtype=torch.float32)

    env = {
        _PLACEHOLDERS[0]: dy32,
        _PLACEHOLDERS[1]: x32,
        _PLACEHOLDERS[2]: w32,
        _PLACEHOLDERS[3]: b32,
    }

    # dbias reshape [1, N] -> [N]; restored to the runtime N.
    overrides = {"view": [[N]]}

    dx, dweight, dbias = run_graph(_GRAPH_PATH, env, _REGISTRY, overrides)
    return dx.to(orig_x_dtype), dweight.to(orig_w_dtype), dbias.to(orig_x_dtype)
