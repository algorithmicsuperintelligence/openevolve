"""Structural diff between two AtenIR JSON graphs.

Names are mapped to topological positions before comparison so that placeholder
renaming or auto-suffixing (e.g. ``grad_out_1`` vs ``grad_output_1``) does not
register as a structural difference.  Compares:
  - placeholder count and (shape, dtype) multiset
  - distinct_ops set
  - call_function node count
  - per-position (target, output_shape/dtype, scalar_args, reduction_dims,
    keepdim, predecessor positions) tuple
  - output node position references
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def _canon(graph: dict):
    nodes = graph["nodes"]
    name_to_pos = {n["name"]: i for i, n in enumerate(nodes) if "name" in n}
    canon = []

    def to_pos(a):
        if isinstance(a, list):
            return tuple(to_pos(x) for x in a)
        return name_to_pos.get(a, a)

    for i, n in enumerate(nodes):
        op = n["op"]
        if op == "placeholder":
            canon.append(("P", i,
                          tuple(n["shape"]) if n["shape"] is not None else None,
                          n["dtype"]))
        elif op == "call_function":
            args_ordered = n.get("args_ordered")
            if args_ordered is not None:
                # Represent each arg as either ("N", position) for node refs or
                # ("S", value) for scalars so names don't affect structural equality.
                def _arg_canon(entry):
                    if entry["kind"] == "node":
                        return ("N", name_to_pos.get(entry["name"], entry["name"]))
                    v = entry["value"]
                    return ("S", tuple(v) if isinstance(v, list) else v)
                args_sig = tuple(_arg_canon(e) for e in args_ordered)
            else:
                # Backward compat: reconstruct from separate buckets (order not preserved).
                args_sig = (
                    tuple(name_to_pos.get(p, p) for p in n["predecessor_ids"]),
                    tuple(n.get("scalar_args") or ()),
                )
            canon.append(("C", i, n["target"],
                          tuple(n["output_shape"]) if n["output_shape"] else None,
                          n["output_dtype"],
                          args_sig,
                          tuple(n.get("reduction_dims") or ()) if n.get("reduction_dims") else None,
                          n.get("keepdim")))
        elif op == "output":
            canon.append(("O", tuple(to_pos(a) for a in n["args"])))
    return canon


def diff(graph_a: dict, graph_b: dict) -> list[str]:
    findings = []
    nodes_a = graph_a["nodes"]
    nodes_b = graph_b["nodes"]

    ph_a = [n for n in nodes_a if n["op"] == "placeholder"]
    ph_b = [n for n in nodes_b if n["op"] == "placeholder"]
    if len(ph_a) != len(ph_b):
        findings.append(f"placeholder count: {len(ph_a)} != {len(ph_b)}")

    ph_a_sig = Counter((tuple(p["shape"]) if p["shape"] else None, p["dtype"]) for p in ph_a)
    ph_b_sig = Counter((tuple(p["shape"]) if p["shape"] else None, p["dtype"]) for p in ph_b)
    if ph_a_sig != ph_b_sig:
        findings.append(
            f"placeholder (shape,dtype) multiset differs:\n  A={dict(ph_a_sig)}\n  B={dict(ph_b_sig)}"
        )

    call_a = [n for n in nodes_a if n["op"] == "call_function"]
    call_b = [n for n in nodes_b if n["op"] == "call_function"]
    if len(call_a) != len(call_b):
        findings.append(f"call_function count: {len(call_a)} != {len(call_b)}")

    ops_a = set(graph_a.get("distinct_ops", []))
    ops_b = set(graph_b.get("distinct_ops", []))
    if ops_a != ops_b:
        findings.append(
            f"distinct_ops differ:\n  only in A: {sorted(ops_a - ops_b)}\n  only in B: {sorted(ops_b - ops_a)}"
        )

    ca, cb = _canon(graph_a), _canon(graph_b)
    if len(ca) != len(cb):
        findings.append(f"total node count: {len(ca)} != {len(cb)}")
    else:
        for i, (a, b) in enumerate(zip(ca, cb)):
            if a != b:
                findings.append(f"node[{i}] differs:\n  A: {a}\n  B: {b}")

    return findings


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="atenir._diff")
    p.add_argument("a")
    p.add_argument("b")
    args = p.parse_args(argv)

    ga = json.loads(Path(args.a).read_text())
    gb = json.loads(Path(args.b).read_text())
    findings = diff(ga, gb)
    if not findings:
        print(f"OK: structurally identical ({args.a} vs {args.b})")
        return 0
    print(f"DIFF: {args.a} vs {args.b}")
    for f in findings:
        print(f"  - {f}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
