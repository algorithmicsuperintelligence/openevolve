"""Deterministic verification pipeline using hand-written dispatch.py kernels (Pipeline D).

Pipeline:
  1. Extract the AtenIR backward graph via atenir.extract.
  2. Build a kernel registry from atenir.primitive_triton.dispatch.make_registry() — no LLM.
  3. Run the graph in-process with atenir.compose.run_graph.
  4. Compare outputs against torch.autograd.grad.
  5. Write verification_report.json and lowering_context.md.

lowering_context.md is consumed by Pipeline E (handwritten_fusion_agent) as grounding
context for the LLM kernel fusion step.
"""

from __future__ import annotations

import importlib
import inspect
import json
import re as _re
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


@dataclass(frozen=True)
class HandwrittenDispatchConfig:
    forward: str        # callable spec "pkg.mod:fn"
    example_input: str  # spec string "[(8,64) f32, (64) f32, (64) f32]"
    output_dir: Path
    dtypes: tuple[str, ...]
    atol: float
    rtol: float
    fp16_atol: float
    fp16_rtol: float
    python: str


# ── graph extraction ──────────────────────────────────────────────────────────


def _extract_graph(config: HandwrittenDispatchConfig) -> Path:
    graph_path = config.output_dir / "atenir_graph.json"
    cmd = [
        config.python, "-m", "atenir.extract",
        "--fn", config.forward,
        "--example-input", config.example_input,
        "--out", str(graph_path),
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parents[2]),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    (config.output_dir / "extract_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (config.output_dir / "extract_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"AtenIR extraction failed:\n{completed.stderr}")
    return graph_path


# ── context generation ────────────────────────────────────────────────────────

_TRITON_SUBMODULES = ("elementwise", "reduction", "gemm", "scatter_gather")


def _get_triton_source(kernel) -> tuple[str, str]:
    """Return (dedup_key, source_text) for a kernel from make_registry.

    Direct Triton module functions  → full function source via inspect.getsource.
    Factory-returned closures       → factory source + referenced helpers (2 levels deep),
                                      surfacing the actual @triton.jit kernel bodies.
    Dispatch.py closures            → closure source (thin wrapper, still informative).
    """
    module_name = getattr(kernel, "__module__", "") or ""
    qualname = getattr(kernel, "__qualname__", "") or ""

    if any(m in module_name for m in _TRITON_SUBMODULES):
        try:
            mod = sys.modules.get(module_name) or importlib.import_module(module_name)
        except ImportError:
            mod = None

        if "<locals>" in qualname and mod is not None:
            # Factory-returned closure (e.g. make_sum_kernel.<locals>.triton_sum).
            # Show the factory definition + all _*-named helpers it transitively calls,
            # which surfaces the @triton.jit kernel bodies.
            factory_name = qualname.split(".<locals>")[0]
            dedup_key = f"{module_name}.{factory_name}"
            factory_fn = getattr(mod, factory_name, None)
            if factory_fn is None:
                return dedup_key, ""
            try:
                factory_src = inspect.getsource(factory_fn)
            except (OSError, TypeError):
                return dedup_key, ""

            parts = [factory_src]
            seen_helpers: set[str] = set()

            def _safe_getsource(fn) -> str | None:
                # Triton JITFunction wraps the original Python fn in .fn or __wrapped__.
                # inspect.getsource on the JITFunction object itself would return the
                # class source, not the kernel body — so unwrap first.
                for candidate in (fn,
                                   getattr(fn, "__wrapped__", None),
                                   getattr(fn, "fn", None)):
                    if candidate is None:
                        continue
                    try:
                        return inspect.getsource(candidate)
                    except (OSError, TypeError):
                        pass
                return None

            def _collect(text: str, depth: int) -> None:
                if depth <= 0:
                    return
                for name in set(_re.findall(r"\b(_[a-zA-Z0-9_]+)\b", text)):
                    if name in seen_helpers:
                        continue
                    fn = getattr(mod, name, None)
                    if fn is not None and callable(fn):
                        seen_helpers.add(name)
                        fn_src = _safe_getsource(fn)
                        if fn_src:
                            parts.append(fn_src)
                            _collect(fn_src, depth - 1)

            _collect(factory_src, depth=2)
            return dedup_key, "\n\n".join(parts)

        # Direct module-level function (e.g. elementwise.mul_tt).
        dedup_key = f"{module_name}.{qualname}"
        try:
            return dedup_key, inspect.getsource(kernel)
        except (OSError, TypeError):
            return dedup_key, ""

    # Dispatch.py closure or unknown: return closure source as-is.
    dedup_key = f"dispatch.{qualname}"
    try:
        return dedup_key, inspect.getsource(kernel)
    except (OSError, TypeError):
        return dedup_key, ""


def generate_dispatch_context(graph_path: Path) -> str:
    """Build a lowering_context.md from dispatch.py for each call_function node.

    Shows the full Triton implementation for each node — not truncated, not just
    the dispatch closure.  For factory-returned kernels (e.g. reduction ops) this
    includes the factory function plus all @triton.jit helper kernels it calls.
    Repeated uses of the same underlying implementation are deduplicated with a
    back-reference so the context stays readable.

    Safe to call without CUDA: make_kernel creates closures but never executes
    Triton kernels, so this works on any machine that has the package installed.
    The resulting file is consumed by Pipeline E as grounding for the LLM.
    """
    from atenir.primitive_triton.dispatch import make_registry, _kernel_label
    from pipeline.primitive_atenir_lowering_agent.graph_summary import summarize_graph

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    registry = make_registry(graph)
    call_nodes = [n for n in graph["nodes"] if n["op"] == "call_function"]

    lines = [
        "# Hand-Written Dispatch Context",
        "",
        "These kernels come from `atenir/primitive_triton/dispatch.py`, a verified",
        "hand-written dispatch table that passes all autograd correctness tests.",
        "Triton kernels are in the `elementwise`, `reduction`, `gemm`, and",
        "`scatter_gather` modules; everything else falls back to PyTorch native.",
        "",
        "## AtenIR Graph",
        "",
        summarize_graph(graph),
        "## Per-Node Kernel Mapping",
        "",
    ]

    # dedup_key -> node name where the source was first shown
    seen_sources: dict[str, str] = {}

    for node in call_nodes:
        name = node["name"]
        target = node["target"]
        kernel = registry.get(name)
        label = _kernel_label(kernel) if kernel else "unregistered"

        lines.append(f"### `{name}` — `{target}`")
        lines.append(f"dispatch: `{label}`")
        lines.append("")

        for entry in node.get("args_ordered") or []:
            if entry["kind"] == "node":
                meta = next(
                    (n for n in (node.get("input_nodes") or []) if n["name"] == entry["name"]),
                    {},
                )
                lines.append(
                    f"- tensor `{entry['name']}`: "
                    f"shape={meta.get('shape')} dtype={meta.get('dtype')}"
                )
            else:
                lines.append(f"- scalar value={entry['value']!r}")
        lines.append(
            f"- output: shape={node.get('output_shape')} dtype={node.get('output_dtype')}"
        )

        if kernel is not None:
            dedup_key, src = _get_triton_source(kernel)
            if src:
                if dedup_key not in seen_sources:
                    seen_sources[dedup_key] = name
                    lines += ["```python", src.strip(), "```"]
                else:
                    lines.append(
                        f"_(same Triton implementation as `{seen_sources[dedup_key]}`"
                        f" — see source above)_"
                    )

        lines.append("")

    return "\n".join(lines) + "\n"


# ── per-dtype verification ────────────────────────────────────────────────────


def _run_dtype(
    *,
    forward: str,
    graph_path: Path,
    dtype_name: str,
    atol: float,
    rtol: float,
    seed: int = 0,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"passed": False, "dtype": dtype_name, "error": "CUDA not available"}

    from atenir.compose import run_graph
    from atenir.primitive_triton.dispatch import make_registry

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    placeholders = [n for n in graph["nodes"] if n["op"] == "placeholder"]

    try:
        registry = make_registry(graph)
    except Exception as exc:
        return {
            "passed": False, "dtype": dtype_name,
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }

    dtype = _DTYPES[dtype_name]
    mod_name, fn_name = (forward.split(":", 1) if ":" in forward else forward.rsplit(".", 1))
    forward_fn = getattr(importlib.import_module(mod_name), fn_name)
    fwd_shapes = [tuple(ph["shape"]) for ph in placeholders[1:]]

    torch.manual_seed(seed)
    try:
        inputs = [torch.randn(s, dtype=dtype, device="cuda") for s in fwd_shapes]
        ref_inputs = [t.detach().clone().requires_grad_(True) for t in inputs]
        out = forward_fn(*ref_inputs)
        if isinstance(out, (tuple, list)):
            out = out[0]
        grad_out = torch.randn_like(out)
        expected_grads = tuple(torch.autograd.grad(out, ref_inputs, grad_outputs=grad_out))

        env_tensors = [grad_out.detach().contiguous()] + [t.detach().contiguous() for t in inputs]
        if len(placeholders) != len(env_tensors):
            return {
                "passed": False, "dtype": dtype_name,
                "error_type": "PlaceholderMismatch",
                "error": (
                    f"graph has {len(placeholders)} placeholders "
                    f"but built {len(env_tensors)} env tensors"
                ),
            }
        env = {ph["name"]: t for ph, t in zip(placeholders, env_tensors)}
        actual_grads = run_graph(str(graph_path), env, registry)
    except Exception as exc:
        return {
            "passed": False, "dtype": dtype_name,
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }

    grad_reports, passed = [], True
    for i, (actual, expected) in enumerate(zip(actual_grads, expected_grads)):
        diff = (actual.float() - expected.float()).abs()
        max_abs = float(diff.max())
        max_rel = float((diff / expected.float().abs().clamp(min=1e-8)).max())
        ok = bool(torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol))
        grad_reports.append({
            "index": i, "passed": ok, "shape": list(actual.shape),
            "max_abs": max_abs, "max_rel": max_rel, "atol": atol, "rtol": rtol,
        })
        passed = passed and ok

    n_passed = sum(1 for r in grad_reports if r["passed"])
    return {
        "passed": passed,
        "passed_cases": n_passed,
        "failed_cases": len(grad_reports) - n_passed,
        "total_cases": len(grad_reports),
        "dtype": dtype_name,
        "grad_reports": grad_reports,
    }


# ── orchestrator ──────────────────────────────────────────────────────────────


def synthesize_handwritten_dispatch(config: HandwrittenDispatchConfig) -> int:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[D] Extract: {config.forward}")
    graph_path = _extract_graph(config)
    print(f"  → {graph_path}")

    print("[D] Generate dispatch context (lowering_context.md for Pipeline E)")
    try:
        context = generate_dispatch_context(graph_path)
        context_path = config.output_dir / "lowering_context.md"
        context_path.write_text(context, encoding="utf-8")
        print(f"  → {context_path} ({len(context)} chars)")
    except Exception as exc:
        print(f"  Warning: context generation failed: {exc}")

    print("[D] Generate dispatch program (dispatch_program.py — the optimized seed)")
    try:
        from pipeline.handwritten_dispatch.program_codegen import generate_dispatch_program

        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        program = generate_dispatch_program(graph)
        program_path = config.output_dir / "dispatch_program.py"
        program_path.write_text(program, encoding="utf-8")
        print(f"  → {program_path} ({len(program)} chars)")
    except Exception as exc:
        print(f"  Warning: dispatch program generation failed: {exc}")

    reports = []
    for dtype_name in config.dtypes:
        is_fp16 = dtype_name in {"float16", "fp16", "bfloat16", "bf16"}
        atol = config.fp16_atol if is_fp16 else config.atol
        rtol = config.fp16_rtol if is_fp16 else config.rtol
        print(f"[D] Verify {dtype_name} ...")
        report = _run_dtype(
            forward=config.forward,
            graph_path=graph_path,
            dtype_name=dtype_name,
            atol=atol,
            rtol=rtol,
        )
        status = "PASS" if report.get("passed") else "FAIL"
        n = f"{report.get('passed_cases', 0)}/{report.get('total_cases', 0)}"
        print(f"  {dtype_name}: {status} ({n} gradients)")
        reports.append(report)

    passed = all(r.get("passed") for r in reports)
    passed_cases = sum(r.get("passed_cases", 0) for r in reports)
    total_cases = sum(r.get("total_cases", 0) for r in reports)

    result = {
        "passed": passed,
        "passed_cases": passed_cases,
        "failed_cases": total_cases - passed_cases,
        "total_cases": total_cases,
        "dtypes": list(config.dtypes),
        "reports": reports,
    }
    report_path = config.output_dir / "verification_report.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\n[D] {'PASS' if passed else 'FAIL'}  ({passed_cases}/{total_cases} gradients)")
    print(f"    report: {report_path}")
    return 0 if passed else 1
