"""AtenIR graph extraction CLI.

Two modes — both produce JSON matching the schema of ``layernorm_bwd_graph.json``
(placeholders, call_function nodes with input_nodes / scalar_args / reduction_dims /
keepdim / predecessor_ids, and a single output node).

Named-aten-op mode:
    python -m atenir.extract \\
        --op aten._softmax_backward_data \\
        --example-args "[(64,128) f32, (64,128) f32, -1, f32]" \\
        --out atenir/test_graphs/softmax_bwd.json

Autograd-driven mode (for ops without a named aten backward):
    python -m atenir.extract \\
        --fn atenir._examples:square_sum \\
        --example-input "[(32,64) f32]" \\
        --out atenir/test_graphs/square_sum_bwd.json

Spec grammar for --example-args / --example-input:
    spec   := '[' item (',' item)* ']'
    item   := tensor | list | scalar | bool | dtype
    tensor := '(' int (',' int)* [','] ')' dtype
    list   := '[' item (',' item)* ']'
    scalar := int | float
    dtype  := f16 | f32 | f64 | bf16 | i32 | i64 | bool
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path

import torch
import torch.fx as fx
from torch._decomp import core_aten_decompositions
from torch.fx.experimental.proxy_tensor import make_fx

_DTYPES = {
    "f16": torch.float16,
    "float16": torch.float16,
    "f32": torch.float32,
    "float32": torch.float32,
    "f64": torch.float64,
    "float64": torch.float64,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "i32": torch.int32,
    "int32": torch.int32,
    "i64": torch.int64,
    "int64": torch.int64,
    "bool": torch.bool,
}

_TOK_RE = re.compile(r"\s*([()\[\],]|[^()\[\],\s]+)")


def _tokenize(s: str) -> list[str]:
    tokens, pos = [], 0
    while pos < len(s):
        if s[pos].isspace():
            pos += 1
            continue
        m = _TOK_RE.match(s, pos)
        if not m:
            raise ValueError(f"tokenize: unexpected char {s[pos]!r} at pos {pos}")
        tokens.append(m.group(1))
        pos = m.end()
    return tokens


def _parse_spec(s: str):
    toks = _tokenize(s)
    i = [0]

    def peek():
        return toks[i[0]] if i[0] < len(toks) else None

    def take(expect=None):
        if i[0] >= len(toks):
            raise ValueError("spec: unexpected end")
        t = toks[i[0]]
        if expect is not None and t != expect:
            raise ValueError(f"spec: expected {expect!r}, got {t!r}")
        i[0] += 1
        return t

    def parse_atom():
        t = peek()
        if t == "[":
            return parse_list()
        if t == "(":
            return parse_tensor()
        take()
        if t == "True":
            return True
        if t == "False":
            return False
        if t in _DTYPES:
            return {"kind": "dtype", "value": _DTYPES[t]}
        try:
            return int(t)
        except ValueError:
            pass
        try:
            return float(t)
        except ValueError:
            pass
        raise ValueError(f"spec: unrecognised token {t!r}")

    def parse_shape():
        take("(")
        shape = []
        while peek() != ")":
            shape.append(int(take()))
            if peek() == ",":
                take(",")
        take(")")
        return shape

    def parse_tensor():
        shape = parse_shape()
        dt = peek()
        if dt not in _DTYPES:
            raise ValueError(f"spec: expected dtype after shape, got {dt!r}")
        take()
        return {"kind": "tensor", "shape": shape, "dtype": _DTYPES[dt]}

    def parse_list():
        take("[")
        items = []
        while peek() != "]":
            items.append(parse_atom())
            if peek() == ",":
                take(",")
        take("]")
        return items

    out = parse_list()
    if i[0] != len(toks):
        raise ValueError(f"spec: trailing tokens after parse: {toks[i[0]:]}")
    return out


def _materialise(item, device: str):
    if isinstance(item, dict):
        if item["kind"] == "tensor":
            dt = item["dtype"]
            if dt.is_floating_point:
                return torch.randn(item["shape"], device=device, dtype=dt)
            return torch.zeros(item["shape"], device=device, dtype=dt)
        if item["kind"] == "dtype":
            return item["value"]
    if isinstance(item, list):
        return [_materialise(x, device) for x in item]
    return item


def _resolve_aten(op_name: str):
    rest = op_name[len("aten.") :] if op_name.startswith("aten.") else op_name
    parts = rest.split(".")
    base = parts[0]
    overload = ".".join(parts[1:]) if len(parts) > 1 else "default"
    packet = getattr(torch.ops.aten, base, None)
    if packet is None:
        raise ValueError(f"aten.{base} does not exist")
    if not hasattr(packet, overload):
        avail = [o for o in dir(packet) if not o.startswith("_")]
        raise ValueError(
            f"overload {overload!r} not found for aten.{base}; available: {sorted(avail)}"
        )
    return getattr(packet, overload)


def _import_callable(spec: str):
    mod_name, fn_name = spec.split(":", 1) if ":" in spec else spec.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    if not callable(fn):
        raise TypeError(f"{spec!r} is not callable")
    return fn


# ── Serialization (mirrors layernorm_bwd_graph.json schema) ──────────────────


def _node_meta(gm: fx.GraphModule):
    meta = {}
    for n in gm.graph.nodes:
        v = n.meta.get("val")
        if isinstance(v, torch.Tensor):
            meta[n.name] = {"shape": list(v.shape), "dtype": str(v.dtype)}
    return meta


def _classify_args(node: fx.Node, meta):
    """Split a node's args+kwargs into structured fields.

    Returns:
        input_nodes   -- list of {name, shape, dtype} dicts for tensor predecessors
        scalar_args   -- flat list of non-tensor args in *original* positional order
                         (kept for backward compat; args_ordered is authoritative)
        reduction_dims, keepdim -- extracted from reduction ops
        args_ordered  -- full ordered arg list: each entry is either
                         {"kind": "node", "name": "..."} or
                         {"kind": "scalar", "value": <json-serialisable>}
                         Reduction-specific args (dims list, keepdim bool) are
                         consumed as metadata and do NOT appear here.
    """
    input_nodes, scalar_args = [], []
    reduction_dims, keepdim = None, None
    args_ordered = []

    target = str(node.target)
    is_reduction = "sum" in target or "mean" in target

    def _process(a):
        nonlocal keepdim, reduction_dims
        if isinstance(a, fx.Node):
            m = meta.get(a.name, {})
            input_nodes.append({"name": a.name, "shape": m.get("shape"), "dtype": m.get("dtype")})
            args_ordered.append({"kind": "node", "name": a.name})

        elif isinstance(a, bool):
            if is_reduction:
                keepdim = a
                # consumed as metadata — not added to args_ordered
            else:
                scalar_args.append(a)
                args_ordered.append({"kind": "scalar", "value": a})

        elif isinstance(a, (int, float)):
            scalar_args.append(a)
            args_ordered.append({"kind": "scalar", "value": a})

        elif isinstance(a, (list, tuple)):
            vals, ok = [], True
            for x in a:
                if isinstance(x, fx.Node):
                    ok = False
                    break
                vals.append(x)
            if ok:
                if is_reduction:
                    reduction_dims = vals
                    # consumed as metadata — not added to args_ordered
                else:
                    scalar_args.append(vals)
                    args_ordered.append({"kind": "scalar", "value": vals})

    for a in node.args:
        _process(a)
    for v in (node.kwargs or {}).values():
        _process(v)

    return input_nodes, scalar_args, reduction_dims, keepdim, args_ordered


def _serialise(gm: fx.GraphModule) -> dict:
    meta = _node_meta(gm)
    distinct_ops, nodes_json = set(), []

    def arg_to_str(a):
        if isinstance(a, fx.Node):
            return a.name
        if isinstance(a, (list, tuple)):
            return [arg_to_str(x) for x in a]
        return a

    for n in gm.graph.nodes:
        if n.op == "placeholder":
            m = meta.get(n.name, {})
            nodes_json.append(
                {
                    "op": "placeholder",
                    "name": n.name,
                    "shape": m.get("shape"),
                    "dtype": m.get("dtype"),
                }
            )
        elif n.op == "call_function":
            m = meta.get(n.name, {})
            input_nodes, scalar_args, red_dims, keepdim, args_ordered = _classify_args(n, meta)
            distinct_ops.add(str(n.target))
            nodes_json.append(
                {
                    "op": "call_function",
                    "name": n.name,
                    "target": str(n.target),
                    "output_shape": m.get("shape"),
                    "output_dtype": m.get("dtype"),
                    "input_nodes": input_nodes,
                    "scalar_args": scalar_args,
                    "reduction_dims": red_dims,
                    "keepdim": keepdim,
                    "predecessor_ids": [x["name"] for x in input_nodes],
                    "args_ordered": args_ordered,
                }
            )
        elif n.op == "output":
            nodes_json.append({"op": "output", "args": [arg_to_str(a) for a in n.args]})

    return {"nodes": nodes_json, "distinct_ops": sorted(distinct_ops)}


# ── Extraction modes ─────────────────────────────────────────────────────────


def extract_named_op(op_name: str, parsed_args, device: str = "cpu") -> fx.GraphModule:
    target = _resolve_aten(op_name)
    materialised = [_materialise(x, device) for x in parsed_args]
    # Only tensor positions become FX placeholders; lists/scalars/dtypes are
    # closed over by the wrapper so make_fx keeps them as inline constants.
    # Without this, e.g. ``[256]`` and ``[True,True,True]`` get pytree-flattened
    # into one shapeless placeholder per element.
    tensor_positions = [i for i, v in enumerate(materialised) if isinstance(v, torch.Tensor)]
    tensor_args = [materialised[i] for i in tensor_positions]
    tensor_set = set(tensor_positions)

    def wrapped(*tensors):
        full, ti = [], 0
        for j in range(len(materialised)):
            if j in tensor_set:
                full.append(tensors[ti])
                ti += 1
            else:
                full.append(materialised[j])
        return target(*full)

    return make_fx(wrapped, decomposition_table=core_aten_decompositions())(*tensor_args)


def extract_autograd(fn_spec: str, parsed_inputs, device: str = "cpu") -> fx.GraphModule:
    forward_fn = _import_callable(fn_spec)
    fwd_in = [_materialise(x, device) for x in parsed_inputs]
    if not all(isinstance(t, torch.Tensor) for t in fwd_in):
        raise ValueError("autograd mode: --example-input must be a flat list of tensors")

    with torch.no_grad():
        sample_out = forward_fn(*[t.detach() for t in fwd_in])
    if isinstance(sample_out, (tuple, list)):
        if len(sample_out) != 1:
            raise NotImplementedError(
                f"autograd mode currently supports single-output forwards; got {len(sample_out)}"
            )
        sample_out = sample_out[0]
    grad_out = torch.randn_like(sample_out)

    def bwd(grad_out, *fwd_inputs):
        ins = [t.detach().requires_grad_(True) for t in fwd_inputs]
        out = forward_fn(*ins)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return torch.autograd.grad(out, ins, grad_outputs=grad_out)

    return make_fx(bwd, decomposition_table=core_aten_decompositions())(grad_out, *fwd_in)


# ── CLI ──────────────────────────────────────────────────────────────────────


def _print_summary(graph: dict, path: Path):
    placeholders = [n for n in graph["nodes"] if n["op"] == "placeholder"]
    call_nodes = [n for n in graph["nodes"] if n["op"] == "call_function"]
    print(f"Wrote {path}")
    print(f"  placeholders ({len(placeholders)}):")
    for p in placeholders:
        print(f"    {p['name']:24s} shape={p['shape']} dtype={p['dtype']}")
    print(f"  call_function nodes: {len(call_nodes)}")
    print(f"  distinct ops ({len(graph['distinct_ops'])}):")
    for op in graph["distinct_ops"]:
        print(f"    {op}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="atenir.extract",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--op", help="aten op name, e.g. aten._softmax_backward_data")
    mode.add_argument("--fn", help="forward callable, e.g. 'pkg.mod:fn'")
    p.add_argument("--example-args", help="spec for --op (CLI list)")
    p.add_argument("--example-input", help="spec for --fn (CLI list of forward inputs)")
    p.add_argument("--out", required=True, help="output JSON path")
    p.add_argument("--device", default="cpu", help="device for trace tensors (default: cpu)")
    args = p.parse_args(argv)

    if args.op:
        if not args.example_args:
            p.error("--op requires --example-args")
        parsed = _parse_spec(args.example_args)
        gm = extract_named_op(args.op, parsed, device=args.device)
    else:
        if not args.example_input:
            p.error("--fn requires --example-input")
        parsed = _parse_spec(args.example_input)
        gm = extract_autograd(args.fn, parsed, device=args.device)

    graph = _serialise(gm)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(graph, indent=2))
    _print_summary(graph, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
