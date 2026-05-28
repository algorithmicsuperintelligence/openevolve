"""
Feasibility check: AtenIR backward graph for nn.LayerNorm(256).

Two complementary extraction paths:

  A) aot_export_module + trace_joint=True  (joint graph, all inputs as params)
  B) make_fx directly on native_layer_norm_backward  (full backward only)

Pass criterion: backward decomposes into ~10 primitive aten ops (mul, mean,
sum, sub, rsqrt, div, etc.), NOT the fused aten.native_layer_norm_backward.

Path B graph + op set are also written to atenir/layernorm_bwd_graph.json.
"""

import json
import torch
import torch.nn as nn
from torch._functorch.aot_autograd import aot_export_module
from torch._decomp import core_aten_decompositions
from torch.fx.experimental.proxy_tensor import make_fx
import torch.fx as fx

HIDDEN = 256
BATCH = 64


# ── Path A: aot_export_module ────────────────────────────────────────────────

class LayerNormAllParams(nn.Module):
    """All tensors are parameters so the joint graph computes grad_input too."""

    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(HIDDEN)
        # Register x as a buffer so it appears as a primal input, but we want
        # it to be a parameter so the backward includes grad_input.
        self.register_parameter("x", nn.Parameter(torch.randn(BATCH, HIDDEN)))

    def forward(self):
        # aot_export_module requires tuple output
        return (self.ln(self.x).sum(),)


def detect_backward_nodes(gm: fx.GraphModule) -> list[fx.Node]:
    """
    Identify backward nodes in a joint graph produced with output_loss_index.

    With output_loss_index the seed gradient (full_like or similar) is the
    first backward node; there are no explicit tangent placeholders.
    Strategy: the last node before the final 'output' node is a gradient
    accumulation result. Walk backwards from output, collecting the 'gradient
    cluster' — nodes that DO NOT appear as inputs to earlier (forward) nodes.

    Simpler heuristic used here: collect nodes that appear AFTER the first
    aten.sum (the loss), since the loss computation is the forward/backward
    boundary in a loss-seeded joint graph.
    """
    nodes = list(gm.graph.nodes)
    # Find the loss node: first aten.sum.dim_IntList with output shape []
    loss_idx = None
    for i, n in enumerate(nodes):
        if n.op == "call_function" and "sum" in str(n.target):
            # Check shape — the loss is a scalar
            if hasattr(n, "meta") and "val" in n.meta:
                if n.meta["val"].shape == torch.Size([]):
                    loss_idx = i
                    break
    if loss_idx is None:
        # Fallback: first aten.full_like is the gradient seed
        for i, n in enumerate(nodes):
            if n.op == "call_function" and "full_like" in str(n.target):
                loss_idx = i - 1
                break

    if loss_idx is None:
        return []

    bwd = [n for n in nodes[loss_idx + 1 :] if n.op != "output"]
    return bwd


# ── Path B: make_fx on native_layer_norm_backward ───────────────────────────

def layernorm_backward_fn(grad_out, x, mean, rstd, weight, bias):
    """Wraps native_layer_norm_backward with output_mask=[True, True, True].

    Signature: (grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask)
    """
    return torch.ops.aten.native_layer_norm_backward.default(
        grad_out, x, [HIDDEN], mean, rstd, weight, bias,
        [True, True, True],  # grad_input, grad_weight, grad_bias
    )


def extract_with_make_fx():
    """Trace native_layer_norm_backward through core_aten_decompositions."""
    x = torch.randn(BATCH, HIDDEN)
    weight = torch.randn(HIDDEN)
    bias = torch.randn(HIDDEN)
    grad_out = torch.randn(BATCH, HIDDEN)
    mean = torch.zeros(BATCH, 1)
    rstd = torch.ones(BATCH, 1)

    decomps = core_aten_decompositions()
    gm = make_fx(layernorm_backward_fn, decomposition_table=decomps)(
        grad_out, x, mean, rstd, weight, bias
    )
    return gm


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    decomps = core_aten_decompositions()

    # ── A: Joint graph via aot_export_module ──────────────────────────────
    print("=" * 70)
    print("PATH A — FULL JOINT FX GRAPH  (aot_export_module, trace_joint=True)")
    print("=" * 70)

    model = LayerNormAllParams()
    result = aot_export_module(
        model,
        (),           # no external args — everything is a parameter
        decompositions=decomps,
        trace_joint=True,
        output_loss_index=0,
    )
    gm_joint = result[0] if isinstance(result, tuple) else result
    gm_joint.print_readable()

    bwd_nodes_a = detect_backward_nodes(gm_joint)
    print("\n--- Backward nodes detected (Path A) ---")
    for n in bwd_nodes_a:
        if n.op == "call_function":
            print(f"  {n.name:40s}  {n.target}")
        else:
            print(f"  {n.name:40s}  [{n.op}]")

    ops_a = {str(n.target) for n in bwd_nodes_a if n.op == "call_function"}

    # ── B: Direct make_fx on native_layer_norm_backward ───────────────────
    print("\n" + "=" * 70)
    print("PATH B — DECOMPOSED native_layer_norm_backward  (make_fx)")
    print("=" * 70)

    gm_bwd = extract_with_make_fx()
    gm_bwd.print_readable()

    ops_b = {
        str(n.target)
        for n in gm_bwd.graph.nodes
        if n.op == "call_function"
    }

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"ATEN OPS IN BACKWARD — Path A (joint grad cluster): {len(ops_a)}")
    print("=" * 70)
    for op in sorted(ops_a):
        print(f"  {op}")

    print(f"\nATEN OPS IN BACKWARD — Path B (make_fx full backward): {len(ops_b)}")
    print("=" * 70)
    for op in sorted(ops_b):
        print(f"  {op}")

    # ── Dump Path B to JSON ────────────────────────────────────────────────
    # Build shape/dtype lookup for every node in the graph.
    node_meta: dict[str, dict] = {}
    for n in gm_bwd.graph.nodes:
        if "val" in n.meta:
            node_meta[n.name] = {
                "shape": list(n.meta["val"].shape),
                "dtype": str(n.meta["val"].dtype),
            }

    def _classify_args(node: fx.Node):
        """Split a node's args into typed buckets for downstream lowering."""
        input_nodes = []
        scalar_args = []
        reduction_dims = None
        keepdim = None
        is_sum = "sum" in str(node.target)

        for a in node.args:
            if isinstance(a, fx.Node):
                m = node_meta.get(a.name, {})
                input_nodes.append({
                    "name": a.name,
                    "shape": m.get("shape"),
                    "dtype": m.get("dtype"),
                })
            elif isinstance(a, bool):          # bool before int — bool is subclass of int
                if is_sum:
                    keepdim = a
            elif isinstance(a, (int, float)):
                scalar_args.append(a)
            elif isinstance(a, (list, tuple)):
                if is_sum:
                    reduction_dims = list(a)

        return input_nodes, scalar_args, reduction_dims, keepdim

    def _arg_to_str(a):
        if isinstance(a, fx.Node):
            return a.name
        if isinstance(a, (list, tuple)):
            return [_arg_to_str(x) for x in a]
        return a

    nodes_json = []
    for node in gm_bwd.graph.nodes:
        if node.op == "placeholder":
            m = node_meta.get(node.name, {})
            nodes_json.append({
                "op": "placeholder",
                "name": node.name,
                "shape": m.get("shape"),
                "dtype": m.get("dtype"),
            })
        elif node.op == "call_function":
            m = node_meta.get(node.name, {})
            input_nodes, scalar_args, reduction_dims, keepdim = _classify_args(node)
            predecessor_ids = [n["name"] for n in input_nodes]
            nodes_json.append({
                "op": "call_function",
                "name": node.name,
                "target": str(node.target),
                "output_shape": m.get("shape"),
                "output_dtype": m.get("dtype"),
                "input_nodes": input_nodes,
                "scalar_args": scalar_args,
                "reduction_dims": reduction_dims,
                "keepdim": keepdim,
                "predecessor_ids": predecessor_ids,
            })
        elif node.op == "output":
            nodes_json.append({"op": "output", "args": [_arg_to_str(a) for a in node.args]})

    out_path = "atenir/layernorm_bwd_graph.json"
    with open(out_path, "w") as f:
        json.dump({"nodes": nodes_json, "distinct_ops": sorted(ops_b)}, f, indent=2)
    print(f"\nPath B graph written to {out_path}")

    # ── Pass / fail ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    fused_tag = "aten.native_layer_norm_backward"
    a_fused = any(fused_tag in op for op in ops_a)
    b_fused = any(fused_tag in op for op in ops_b)

    if b_fused:
        print("FAIL (Path B): native_layer_norm_backward NOT decomposed.")
        from torch._decomp import decomposition_table
        key = next((k for k in decomposition_table if "native_layer_norm_backward" in str(k)), None)
        print(f"  decomposition_table entry: {key}")
    else:
        print(f"PASS (Path B): backward decomposes into {len(ops_b)} primitive aten ops")
        print("  No fused native_layer_norm_backward — ready for kernel synthesis.")

    if a_fused:
        print("NOTE  (Path A): joint-graph grad cluster still contains fused op.")
    elif not ops_a:
        print("NOTE  (Path A): grad cluster empty — loss is scalar, no tangent nodes.")
    else:
        print(f"PASS  (Path A): joint grad cluster has {len(ops_a)} ops, all primitive.")
    print("=" * 70)


if __name__ == "__main__":
    main()
