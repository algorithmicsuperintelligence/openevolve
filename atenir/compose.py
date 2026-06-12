"""AtenIR runtime — pure, op-agnostic graph walker.

The runtime knows nothing about any particular op, dtype, shape, or downstream
API.  It loads an AtenIR JSON graph (produced by :mod:`atenir.extract`), walks
the call_function nodes in topological order, and dispatches each one to a
caller-supplied kernel.  All op-specific glue (input recompute, dtype casting,
shape-specialised constants, output ordering) lives in adapter modules — see
``atenir.bench_adapters.layernorm`` for the worked example.

Public surface is a single function, :func:`run_graph`.  Contract:

* ``graph_json_path`` — path to AtenIR JSON.  Loaded and cached per path.
* ``env`` — dict mapping placeholder ``name`` strings to caller-prepared
  tensors.  The caller is responsible for dtype / device / contiguity.
* ``kernel_registry`` — dict mapping ``node["name"]`` to a callable that
  takes ``(*tensor_inputs, *scalar_args)`` and returns one tensor.  Every
  ``call_function`` node in the graph must have a registered kernel; if a
  node is missing, ``run_graph`` raises ``KeyError`` with a clear message.
* ``scalar_overrides`` — optional dict mapping node names to replacement
  scalar-arg lists.  Used today to swap shape-specialised integer literals
  (e.g. ``N`` baked in at trace time) until symbolic-mode extraction lands.

Returns a tuple of tensors in the order declared by the graph's output node
(``graph["nodes"][-1]["args"][0]``).
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable


@lru_cache(maxsize=None)
def _load_graph(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _kernel_label(fn: Callable) -> str:
    """Return a short human-readable label for a kernel, e.g. 'triton/elementwise:sin_'."""
    module = getattr(fn, "__module__", "") or ""
    qname = getattr(fn, "__qualname__", "") or getattr(fn, "__name__", repr(fn))
    for seg in ("elementwise", "gemm", "reduction", "scatter_gather"):
        if seg in module:
            short = seg.replace("_gather", "")
            return f"triton/{short}:{qname}"
    tag = getattr(fn, "_dispatch_tag", None)
    if tag:
        return tag
    return f"pytorch:{qname}"


def run_graph(
    graph_json_path: str | Path,
    env: dict[str, Any],
    kernel_registry: dict[str, Callable],
    scalar_overrides: dict[str, list] | None = None,
    verbose: bool = False,
) -> tuple:
    """Walk an AtenIR graph and dispatch each call_function node.

    See module docstring for the full contract.

    When ``verbose=True``, prints each dispatched op and which kernel handles it.
    """
    graph = _load_graph(str(graph_json_path))
    env = dict(env)
    overrides = scalar_overrides or {}

    if verbose:
        print(f"\n[AtenIR] execution trace  ({graph_json_path})", file=sys.stderr, flush=True)
        print(f"  {'node':45s} {'target':50s}  kernel", file=sys.stderr, flush=True)
        print(f"  {'-'*45} {'-'*50}  {'-'*30}", file=sys.stderr, flush=True)

    for node in graph["nodes"]:
        op = node.get("op")
        if op in ("placeholder", "output"):
            continue
        if op != "call_function":
            raise ValueError(
                f"unexpected node op {op!r} for node {node.get('name')!r}"
            )
        name = node["name"]
        if name not in kernel_registry:
            raise KeyError(
                f"No kernel registered for {name!r} (target {node['target']!r})"
            )

        args_ordered = node.get("args_ordered")
        if args_ordered is not None:
            # Only pass tensor inputs; scalars are baked into kernels at dispatch
            # time (make_kernel reads node["scalar_args"]).  Passing traced
            # scalar kwargs (device=cpu, dtype=float32, …) at GPU runtime would
            # silently override device/dtype to CPU trace-time values.
            args = [env[entry["name"]] for entry in args_ordered if entry["kind"] == "node"]
        else:
            # Backward compat: graphs serialised before args_ordered was added.
            args = [env[p] for p in (node.get("predecessor_ids") or [])]

        if verbose:
            label = _kernel_label(kernel_registry[name])
            print(f"  {name:45s} {node['target']:50s}  {label}", file=sys.stderr, flush=True)

        env[name] = kernel_registry[name](*args)

    out_node = graph["nodes"][-1]
    if out_node.get("op") != "output":
        raise ValueError("last node is not an output node; cannot determine return tuple")
    return tuple(env[n] for n in out_node["args"][0])
