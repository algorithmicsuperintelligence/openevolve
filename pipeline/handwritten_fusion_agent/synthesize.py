"""Orchestration for hand-written context kernel fusion synthesis (Pipeline E).

Pipeline:
  1. Extract the AtenIR backward graph.
  2. Build per-node dispatch context from dispatch.py (no LLM — same as Pipeline D step 2).
  3. Pass that verified context to kernel_fusion_agent for LLM-driven kernel synthesis.

The difference from Pipeline C is the source of the lowering context:
  C: context comes from LLM-generated per-op kernels (Pipeline B output).
  E: context comes from the hand-written dispatch table (dispatch.py), which is
     already verified against autograd. The LLM gets a guaranteed-correct reference
     for each op in the graph before attempting fusion.
"""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

from pipeline.fusion_agent.synthesize import FusionConfig
from pipeline.handwritten_dispatch.synthesize import generate_dispatch_context
from pipeline.kernel_fusion_agent.synthesize import synthesize_kernel_fusion


def _extract_graph(
    *,
    python: str,
    forward: str,
    example_input: str,
    output_dir: Path,
) -> Path:
    """Extract AtenIR graph with a caller-supplied example_input spec."""
    graph_path = output_dir / "atenir_graph.json"
    cmd = [
        python, "-m", "atenir.extract",
        "--fn", forward,
        "--example-input", example_input,
        "--out", str(graph_path),
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parents[2]),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    (output_dir / "extract_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "extract_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"AtenIR extraction failed:\n{completed.stderr}")
    return graph_path


def synthesize_handwritten_fusion(config: FusionConfig, example_input: str) -> int:
    """Synthesize a fused Triton kernel using hand-written dispatch as grounding context.

    Args:
        config: FusionConfig (same as kernel_fusion_agent; lowering_context is ignored).
        example_input: spec string for graph extraction, e.g. "[(8,64) f32, (64) f32]".
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[E] Extract: {config.forward}")
    graph_path = _extract_graph(
        python=config.python,
        forward=config.forward,
        example_input=example_input,
        output_dir=config.output_dir,
    )
    print(f"  → {graph_path}")

    print("[E] Generate hand-written dispatch context")
    context = generate_dispatch_context(graph_path)
    context_path = config.output_dir / "lowering_context.md"
    context_path.write_text(context, encoding="utf-8")
    print(f"  → {context_path} ({len(context)} chars)")

    print("[E] Synthesize fused kernel (kernel_fusion_agent with hand-written context)")
    # synthesize_kernel_fusion will re-extract the graph (idempotent write to the same
    # path) then run its plan→codegen→verify loop with our context injected.
    return synthesize_kernel_fusion(replace(config, lowering_context=context))
