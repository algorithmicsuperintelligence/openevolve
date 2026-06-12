"""AtenIR graph summarization for per-op lowering prompts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_graph(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_graph(graph: dict[str, Any], max_nodes: int | None = None) -> str:
    placeholders = [node for node in graph["nodes"] if node.get("op") == "placeholder"]
    calls = [node for node in graph["nodes"] if node.get("op") == "call_function"]
    output = graph["nodes"][-1]
    selected_calls = calls if max_nodes is None else calls[:max_nodes]

    lines = ["# AtenIR Graph Summary", ""]
    lines.append(f"- placeholders: {len(placeholders)}")
    lines.append(f"- call_function nodes: {len(calls)}")
    lines.append(f"- distinct ops: {len(graph.get('distinct_ops', []))}")
    lines.append("")
    lines.append("## Placeholders")
    for node in placeholders:
        lines.append(f"- `{node['name']}`: shape={node.get('shape')} dtype={node.get('dtype')}")
    lines.append("")
    lines.append("## Outputs")
    lines.append(f"- `{output.get('args')}`")
    lines.append("")
    lines.append("## Call Nodes")
    for index, node in enumerate(selected_calls):
        ao = node.get("args_ordered") or []
        tensor_inputs = [e["name"] for e in ao if e.get("kind") == "node"]
        scalar_values = [e["value"] for e in ao if e.get("kind") == "scalar" and e.get("value") is not None]
        lines.append(
            f"{index}. `{node['name']}` target=`{node.get('target')}` "
            f"inputs={tensor_inputs} scalars={scalar_values} "
            f"shape={node.get('output_shape')} dtype={node.get('output_dtype')}"
        )
    if max_nodes is not None and len(calls) > max_nodes:
        lines.append(f"... omitted {len(calls) - max_nodes} call nodes ...")
    return "\n".join(lines) + "\n"


def summarize_graph_file(path: Path, max_nodes: int | None = None) -> str:
    return summarize_graph(load_graph(path), max_nodes=max_nodes)


def list_call_nodes(graph: dict[str, Any]) -> list[dict[str, Any]]:
    return [node for node in graph["nodes"] if node.get("op") == "call_function"]


def summarize_node(node: dict[str, Any]) -> str:
    """Return a detailed description of one call_function node for an LLM prompt."""
    lines = [
        f"Node name: `{node['name']}`",
        f"Target op: `{node['target']}`",
        "",
        "Arguments (positional order used when calling the kernel):",
    ]
    for i, arg in enumerate(node.get("args_ordered") or []):
        if arg["kind"] == "node":
            input_info = next(
                (n for n in (node.get("input_nodes") or []) if n["name"] == arg["name"]),
                {},
            )
            lines.append(
                f"  {i}. tensor `{arg['name']}`: "
                f"shape={input_info.get('shape')} dtype={input_info.get('dtype')}"
            )
        else:
            lines.append(f"  {i}. scalar  value={arg['value']!r}")

    if node.get("reduction_dims") is not None:
        lines.append(f"reduction_dims: {node['reduction_dims']}")
    if node.get("keepdim") is not None:
        lines.append(f"keepdim: {node['keepdim']}")

    lines.append("")
    lines.append(f"Output: shape={node.get('output_shape')} dtype={node.get('output_dtype')}")
    return "\n".join(lines)
