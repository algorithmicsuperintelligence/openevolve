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
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable


@lru_cache(maxsize=None)
def _load_graph(path: str) -> dict:
    return json.loads(Path(path).read_text())


def run_graph(
    graph_json_path: str | Path,
    env: dict[str, Any],
    kernel_registry: dict[str, Callable],
    scalar_overrides: dict[str, list] | None = None,
) -> tuple:
    """Walk an AtenIR graph and dispatch each call_function node.

    See module docstring for the full contract.
    """
    graph = _load_graph(str(graph_json_path))
    env = dict(env)
    overrides = scalar_overrides or {}

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
            # Reconstruct args in original interleaved order; apply scalar_overrides
            # positionally (override[i] replaces the i-th scalar slot).
            scalar_override = iter(overrides.get(name, []))
            args = []
            for entry in args_ordered:
                if entry["kind"] == "node":
                    args.append(env[entry["name"]])
                else:
                    try:
                        args.append(next(scalar_override))
                    except StopIteration:
                        args.append(entry["value"])
        else:
            # Backward compat: graphs serialised before args_ordered was added.
            scalar_args = list(overrides.get(name, node.get("scalar_args") or []))
            args = [env[p] for p in (node.get("predecessor_ids") or [])] + scalar_args

        env[name] = kernel_registry[name](*args)

    out_node = graph["nodes"][-1]
    if out_node.get("op") != "output":
        raise ValueError("last node is not an output node; cannot determine return tuple")
    return tuple(env[n] for n in out_node["args"][0])
