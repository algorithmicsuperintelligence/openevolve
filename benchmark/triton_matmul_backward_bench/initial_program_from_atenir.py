"""Matmul backward seed for OpenEvolve, composed from the AtenIR graph.

Unlike ``initial_program.py`` (which delegates to a hand-written naive tiled
matmul), this seed runs the *automatically extracted* ATen backward graph
(``atenir/matmul_bwd_graph.json``, produced by ``python -m atenir.extract``)
through the op-agnostic AtenIR runtime (``atenir.compose.run_graph``) + the
general primitive-Triton dispatch (``atenir.primitive_triton.dispatch``).

There are no EVOLVE-BLOCK markers, so OpenEvolve evolves the whole file. The
graph has no trace-time shape/size constants (only axis permutes), so no
scalar_overrides are needed.

IMPORTANT caveat (main branch): the general dispatch routes ``aten.mm`` ->
``torch.mm`` (cuBLAS), NOT a Triton GEMM (``gemm.py`` lives on the
``atenir-verification`` branch). So on main this seed is correct but its matmuls
run on cuBLAS; it becomes a true primitive-Triton seed once that GEMM is on main.
The graph is traced in fp32; inputs are up-cast to fp32 and outputs cast back.
"""

import json
import os

import torch

from atenir.compose import run_graph
from atenir.primitive_triton.dispatch import make_registry


def _find_graph_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    local = os.path.join(here, "atenir", "matmul_bwd_graph.json")
    if os.path.exists(local):
        return local
    import benchmark.triton_matmul_backward_bench as _pkg

    return os.path.join(os.path.dirname(_pkg.__file__), "atenir", "matmul_bwd_graph.json")


_GRAPH_PATH = _find_graph_path()
with open(_GRAPH_PATH) as _f:
    _GRAPH = json.load(_f)
# Placeholder order from autograd-mode extraction: (grad_out, a, b).
_PLACEHOLDERS = [n["name"] for n in _GRAPH["nodes"] if n["op"] == "placeholder"]
_REGISTRY = make_registry(_GRAPH)


def matmul_backward_triton(dc, a, b):
    """Compute (da, db) for c = a @ b via the AtenIR graph."""
    orig_a_dtype = a.dtype
    orig_b_dtype = b.dtype

    dc32 = dc.float().contiguous()
    a32 = a.float().contiguous()
    b32 = b.float().contiguous()

    env = {
        _PLACEHOLDERS[0]: dc32,
        _PLACEHOLDERS[1]: a32,
        _PLACEHOLDERS[2]: b32,
    }

    da, db = run_graph(_GRAPH_PATH, env, _REGISTRY)
    return da.to(orig_a_dtype), db.to(orig_b_dtype)
