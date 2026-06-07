"""Orchestration for AtenIR per-op Triton kernel lowering.

Pipeline:
  1. Extract AtenIR backward graph via atenir.extract.
  2. Summarize the graph.
  3. For each call_function node, ask the LLM to generate a Triton kernel.
  4. Assemble all kernels into kernels.py, which exports make_kernel_registry().
  5. Verify correctness with run_lowering_correctness (subprocess).
  6. If verification fails, ask the LLM to repair the whole kernels.py and retry.

The final kernels.py is placed in output_dir/best/kernels.py on success.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import re
import subprocess
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.shared.llm_client import (
    generate_with_openai_compatible_api,
)
from pipeline.primitive_atenir_lowering_agent.graph_summary import (
    list_call_nodes,
    load_graph,
    summarize_graph,
    summarize_node,
)
from pipeline.primitive_atenir_lowering_agent.prompts import (
    SYSTEM_MESSAGE,
    render_op_codegen_prompt,
    render_op_repair_prompt,
)


@dataclass(frozen=True)
class LoweringConfig:
    # Graph extraction
    forward: str  # forward callable spec, e.g. "pkg.mod:fn"  (for --fn mode)
    example_input: str  # spec string, e.g. "[(8,64) f32, (64) f32, (64) f32]"
    # Output
    output_dir: Path
    # LLM
    api_base: str
    model: str
    api_key: str | None
    max_tokens: int
    temperature: float | None
    timeout: int
    # Verification
    dtypes: tuple[str, ...]
    atol: float
    rtol: float
    fp16_atol: float
    fp16_rtol: float
    # Misc
    max_attempts: int  # repair attempts after first assembly
    python: str
    # Seconds to sleep between LLM calls. Set based on your TPM limit:
    #   inter_call_delay = 60 * avg_tokens_per_call / tpm_limit
    # e.g. 30k TPM, ~3k tokens/call → 60 * 3000 / 30000 = 6 seconds
    inter_call_delay: float = 0.0
    parallelism: int = 1
    reuse_existing: bool = False
    dry_run: bool = False


# ── helpers ──────────────────────────────────────────────────────────────────


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    # Complete fence: opening ``` ... closing ```
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).rstrip() + "\n"
    # Truncated response: LLM hit max_tokens before writing the closing ```.
    # Strip just the opening fence line and treat the rest as Python source.
    match = re.match(r"```(?:python)?\s*\n(.*)", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).rstrip() + "\n"
    return text + "\n"


def _check_syntax(code: str) -> str | None:
    """Return a human-readable error if code has a SyntaxError, else None."""
    try:
        compile(code, "<kernels.py>", "exec")
        return None
    except SyntaxError as exc:
        return f"SyntaxError at line {exc.lineno}: {exc.msg}"


def _defined_functions(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _has_required_kernel_function(code: str, node_name: str) -> bool:
    return f"kernel_{node_name}" in _defined_functions(code)


def _missing_kernel_functions(code: str, call_nodes: list[dict]) -> list[str]:
    functions = _defined_functions(code)
    return [
        node["name"]
        for node in call_nodes
        if f"kernel_{node['name']}" not in functions
    ]


def _missing_node_from_name_error(exc: Exception, call_nodes: list[dict]) -> str | None:
    match = re.search(r"name 'kernel_(\w+)' is not defined", str(exc))
    if not match:
        return None
    missing = match.group(1)
    expected = {node["name"] for node in call_nodes}
    return missing if missing in expected else None


def _ensure_unique_triton_names(code: str) -> str:
    """Rename duplicate @triton.jit function names so each is unique.

    When a whole-file LLM response reuses generic names like _mul_kernel across
    multiple nodes, Python silently overwrites earlier definitions with later ones.
    This pass:
      1. Finds every @triton.jit-decorated function in document order.
      2. For the 2nd, 3rd, … occurrence of each name, appends _2, _3, …
      3. Also renames the nearest following call site (name[...]) to match.
    """
    lines = code.splitlines(keepends=True)

    # Collect (line_index, function_name) for every @triton.jit-decorated def
    jit_defs: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        if line.strip() == "@triton.jit":
            # Scan forward for the def line
            for j in range(i + 1, min(i + 5, len(lines))):
                m = re.match(r"\s*def\s+(\w+)\s*\(", lines[j])
                if m:
                    jit_defs.append((j, m.group(1)))
                    break

    from collections import Counter

    dupes = {name for name, cnt in Counter(n for _, n in jit_defs).items() if cnt > 1}
    if not dupes:
        return code

    occurrence: dict[str, int] = {}
    new_lines = list(lines)

    for def_line_idx, name in jit_defs:
        if name not in dupes:
            continue
        occurrence[name] = occurrence.get(name, 0) + 1
        if occurrence[name] == 1:
            continue  # keep first occurrence unchanged

        new_name = f"{name}_{occurrence[name]}"
        # Rename the def
        new_lines[def_line_idx] = new_lines[def_line_idx].replace(
            f"def {name}(", f"def {new_name}(", 1
        )
        # Rename the nearest following call site: old_name[grid](...)
        for k in range(def_line_idx + 1, len(new_lines)):
            if f"{name}[" in new_lines[k]:
                new_lines[k] = new_lines[k].replace(f"{name}[", f"{new_name}[", 1)
                break

    return "".join(new_lines)


def _extract_graph(config: LoweringConfig) -> Path:
    graph_path = config.output_dir / "atenir_graph.json"
    cmd = [
        config.python,
        "-m",
        "atenir.extract",
        "--fn",
        config.forward,
        "--example-input",
        config.example_input,
        "--out",
        str(graph_path),
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parents[2]),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (config.output_dir / "extract_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (config.output_dir / "extract_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"AtenIR extraction failed:\n{completed.stderr}")
    return graph_path


def _prefix_triton_kernels(snippet: str, node_name: str) -> str:
    """Force-rename every @triton.jit function in snippet to use _{node_name}_ prefix.

    When per-node snippets are concatenated into one file, generic names like
    _mul_kernel or sum_kernel get defined multiple times and Python silently
    overwrites earlier definitions. Every wrapper then dispatches to the LAST
    definition, which has the wrong signature.  This function makes each helper
    name unique before assembly, preventing silent misdispatch entirely.
    """
    # Match '@triton.jit' (with optional whitespace/newlines) followed by 'def name'
    jit_names = re.findall(r"@triton\.jit\s*\n\s*def\s+(\w+)", snippet)
    result = snippet
    for old in jit_names:
        if node_name in old:
            continue  # already prefixed
        new = f"_{node_name}_{old.lstrip('_')}"
        # rename definition
        result = re.sub(rf"\bdef\s+{re.escape(old)}\b", f"def {new}", result)
        # rename all launch sites:  old_name[grid](...)
        result = re.sub(rf"\b{re.escape(old)}\[", f"{new}[", result)
    return result


def _strip_module_level_imports(snippet: str) -> str:
    """Remove bare import lines that belong at module level, not inside a snippet.

    The LLM sometimes emits 'import torch' or 'import triton' in the middle of a
    node snippet.  These are harmless in isolation but generate redundant noise and
    can confuse the assembler.
    """
    lines = []
    for line in snippet.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        lines.append(line)
    return "".join(lines)


def _assemble_kernels_file(node_kernels: dict[str, str], call_nodes: list[dict]) -> str:
    """Combine per-node kernel snippets into a complete Python module.

    Each snippet is:
      1. stripped of redundant module-level imports, and
      2. has its @triton.jit helper names prefixed with the node name to prevent
         silent Python name-collision overwrites.
    """
    lines = [
        '"""LLM-generated Triton kernels for AtenIR primitive op lowering."""',
        "import math",
        "import torch",
        "",
        "try:",
        "    import triton",
        "    import triton.language as tl",
        "    HAS_TRITON = True",
        "except ImportError:",
        "    HAS_TRITON = False",
        "",
        "# EVOLVE-BLOCK-START",
        "",
    ]

    for node in call_nodes:
        name = node["name"]
        raw = node_kernels.get(
            name,
            f"def kernel_{name}(*args):\n    raise NotImplementedError('missing kernel for {name}')\n",
        )
        code = _prefix_triton_kernels(_strip_module_level_imports(raw), name)
        lines.append(f"# node: {name}  |  target: {node['target']}")
        lines.append(code.rstrip())
        lines.append("")

    lines += [
        "# EVOLVE-BLOCK-END",
        "",
        "",
        "def make_kernel_registry():",
        "    return {",
    ]
    for node in call_nodes:
        lines.append(f'        "{node["name"]}": kernel_{node["name"]},')
    lines += [
        "    }",
        "",
    ]

    return "\n".join(lines)


def _extract_node_kernels(code: str, call_nodes: list[dict]) -> dict[str, str]:
    """Extract per-node kernel source from an assembled kernels.py using the AST.

    For each call_function node, finds every top-level function whose name starts
    with ``kernel_<node_name>`` or is a ``@triton.jit`` helper that precedes it,
    and returns the contiguous source lines from the first such helper through the
    wrapper function.
    """
    lines = code.splitlines(keepends=True)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}

    # Map function name -> (start_line, end_line) — 1-based, end inclusive
    fn_spans: dict[str, tuple[int, int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn_spans[node.name] = (node.lineno, node.end_lineno)

    result: dict[str, str] = {}
    sorted_fns = sorted(fn_spans.items(), key=lambda x: x[1][0])

    for graph_node in call_nodes:
        name = graph_node["name"]
        wrapper = f"kernel_{name}"
        if wrapper not in fn_spans:
            continue

        wrapper_start, wrapper_end = fn_spans[wrapper]

        # Collect any @triton.jit helpers that appear immediately before the
        # wrapper (between the previous wrapper's end and this wrapper's start).
        block_start = wrapper_start
        for fn_name, (fn_start, fn_end) in sorted_fns:
            if fn_end < wrapper_start and fn_start >= block_start - 1:
                block_start = min(block_start, fn_start)

        # Fall back to just the wrapper if no helpers found above it
        snippet_lines = lines[block_start - 1 : wrapper_end]
        result[name] = "".join(snippet_lines).rstrip() + "\n"

    return result


def _save_node_kernels(node_kernels: dict[str, str], dest_dir: Path) -> None:
    """Write each node's extracted kernel snippet to dest_dir/nodes/<name>/kernel.py."""
    nodes_dir = dest_dir / "nodes"
    nodes_dir.mkdir(parents=True, exist_ok=True)
    for name, code in node_kernels.items():
        node_dir = nodes_dir / name
        node_dir.mkdir(exist_ok=True)
        (node_dir / "kernel.py").write_text(code, encoding="utf-8")


def _verify_kernels(config: LoweringConfig, kernels_path: Path, graph_path: Path) -> dict:
    """Run verification as a subprocess; returns a parsed JSON report."""
    reports = []
    for dtype in config.dtypes:
        atol, rtol = (
            (config.fp16_atol, config.fp16_rtol)
            if dtype in {"float16", "fp16", "bfloat16", "bf16"}
            else (config.atol, config.rtol)
        )
        cmd = [
            config.python,
            "-m",
            "pipeline.run_lowering_correctness",
            "--forward",
            config.forward,
            "--graph-json",
            str(graph_path),
            "--kernels-file",
            str(kernels_path),
            "--dtype",
            dtype,
            "--atol",
            str(atol),
            "--rtol",
            str(rtol),
        ]
        completed = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parents[2]),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        try:
            report = json.loads(completed.stdout)
        except json.JSONDecodeError:
            report = {
                "passed": False,
                "error_type": "VerifierInvocationError",
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "returncode": completed.returncode,
            }
        if completed.stderr:
            report["stderr"] = completed.stderr
        reports.append(report)

    passed_cases = sum(r.get("passed_cases", 0) for r in reports)
    total_cases = sum(r.get("total_cases", 0) for r in reports)
    passed = all(r.get("passed") for r in reports)
    return {
        "passed": passed,
        "passed_cases": passed_cases,
        "failed_cases": total_cases - passed_cases,
        "total_cases": total_cases,
        "dtypes": list(config.dtypes),
        "reports": reports,
    }


# ── per-node verification ─────────────────────────────────────────────────────


def _load_callable_spec(spec: str):
    mod, fn = spec.split(":", 1) if ":" in spec else spec.rsplit(".", 1)
    return getattr(importlib.import_module(mod), fn)


def _import_registry_from_file(path: Path) -> dict:
    mod_name = f"atenir_kernels_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_kernel_registry()


def _build_node_args(node: dict[str, Any], env: dict[str, Any]) -> list:
    """Reconstruct the positional arg list for a call_function node from env."""
    args_ordered = node.get("args_ordered")
    if args_ordered:
        result = []
        for entry in args_ordered:
            if entry["kind"] == "node":
                result.append(env.get(entry["name"]))
            else:
                result.append(entry["value"])
        return result
    scalar_args = list(node.get("scalar_args") or [])
    return [env.get(p) for p in (node.get("predecessor_ids") or [])] + scalar_args


def _verify_per_node(
    config: LoweringConfig,
    kernels_path: Path,
    graph_path: Path,
) -> dict[str, Any]:
    """Test every kernel individually against a dispatch.py reference.

    Steps:
      1. Run the full graph with the hand-written dispatch.py registry to get
         all correct intermediate tensors (the reference env).
      2. For each call_function node, call the LLM kernel with the reference
         inputs and compare output shape + values against the reference.

    Returns a report with per-node pass/fail, covering syntax errors, import
    errors, runtime errors (Triton compilation, shape errors) and numerical
    mismatches — all in one pass.
    """
    import torch
    from atenir.primitive_triton.dispatch import make_registry as make_ref_registry

    graph = load_graph(graph_path)
    placeholders = [n for n in graph["nodes"] if n["op"] == "placeholder"]
    call_nodes = list_call_nodes(graph)

    # ── syntax & import check ──────────────────────────────────────────────
    code = kernels_path.read_text(encoding="utf-8")
    syn_err = _check_syntax(code)
    if syn_err:
        return {
            "passed": False,
            "error_type": "SyntaxError",
            "error": syn_err,
            "node_reports": {},
            "failed_nodes": [n["name"] for n in call_nodes],
            "passed_nodes": [],
        }

    missing_functions = _missing_kernel_functions(code, call_nodes)
    if missing_functions:
        return {
            "passed": False,
            "error_type": "MissingKernelFunctions",
            "error": (
                f"Missing wrapper functions: {missing_functions}. "
                "A per-node generation likely returned an empty or malformed response."
            ),
            "node_reports": {
                name: {
                    "passed": False,
                    "error_type": "MissingKernelFunction",
                    "error": f"kernel_{name} is not defined",
                }
                for name in missing_functions
            },
            "failed_nodes": missing_functions,
            "passed_nodes": [],
        }

    try:
        llm_registry = _import_registry_from_file(kernels_path)
    except Exception as exc:
        missing_node = _missing_node_from_name_error(exc, call_nodes)
        if missing_node is not None:
            return {
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "node_reports": {
                    missing_node: {
                        "passed": False,
                        "error_type": type(exc).__name__,
                        "error": f"kernel_{missing_node} is not defined",
                    }
                },
                "failed_nodes": [missing_node],
                "passed_nodes": [],
            }
        return {
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "node_reports": {},
            "failed_nodes": [n["name"] for n in call_nodes],
            "passed_nodes": [],
        }

    # Check completeness: all expected nodes must be in the registry.
    expected = {n["name"] for n in call_nodes}
    missing = sorted(expected - set(llm_registry.keys()))
    if missing:
        # File was truncated — report immediately so the repair prompt knows.
        return {
            "passed": False,
            "error_type": "IncompleteRegistry",
            "error": (
                f"{len(missing)} nodes missing from make_kernel_registry() — "
                "the file was likely truncated before all kernels were written. "
                "Increase max_tokens or simplify kernel bodies."
            ),
            "missing_nodes": missing,
            "node_reports": {
                n: {"passed": False, "error": "missing from make_kernel_registry()"}
                for n in missing
            },
            "failed_nodes": missing,
            "passed_nodes": [],
        }

    if not torch.cuda.is_available():
        return {"passed": False, "error": "CUDA not available", "node_reports": {}}

    # ── build reference env ────────────────────────────────────────────────
    # Test all configured dtypes so float16 issues are caught here, not only in e2e.
    dtype_names = list(config.dtypes)
    all_node_reports: dict[str, Any] = {}
    all_failed: list[str] = []
    all_passed: list[str] = []

    _DTYPE_MAP = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }

    forward_fn = _load_callable_spec(config.forward)
    fwd_shapes = [tuple(ph["shape"]) for ph in placeholders[1:]]

    for dtype_name in dtype_names:
        dtype = _DTYPE_MAP.get(dtype_name, torch.float32)
        torch.manual_seed(42)
        inputs = [torch.randn(shape, dtype=dtype, device="cuda") for shape in fwd_shapes]
        ref_inputs = [t.detach().clone().requires_grad_(True) for t in inputs]
        out = forward_fn(*ref_inputs)
        if isinstance(out, (tuple, list)):
            out = out[0]
        grad_out = torch.randn_like(out)

        env_tensors = [grad_out.detach().contiguous()] + [t.detach().contiguous() for t in inputs]
        ref_env: dict[str, Any] = {ph["name"]: t for ph, t in zip(placeholders, env_tensors)}

        # Run reference graph to populate all intermediate tensors for this dtype
        ref_registry = make_ref_registry(graph)
        for node in graph["nodes"]:
            if node.get("op") != "call_function":
                continue
            name = node["name"]
            try:
                ref_env[name] = ref_registry[name](*_build_node_args(node, ref_env))
            except Exception:
                ref_env[name] = None

        # Test each LLM kernel against the reference for this dtype
        atol = (
            config.fp16_atol
            if dtype_name in {"float16", "fp16", "bfloat16", "bf16"}
            else config.atol
        )
        rtol = (
            config.fp16_rtol
            if dtype_name in {"float16", "fp16", "bfloat16", "bf16"}
            else config.rtol
        )

        for node in call_nodes:
            name = node["name"]
            if name not in llm_registry:
                # Already reported as missing above; don't overwrite
                if name not in all_node_reports:
                    all_node_reports[name] = {
                        "passed": False,
                        "error": "missing from make_kernel_registry()",
                    }
                continue

            node_args = _build_node_args(node, ref_env)
            ref_out = ref_env.get(name)

            try:
                llm_out = llm_registry[name](*node_args)
            except Exception as exc:
                report_entry = {
                    "passed": False,
                    "dtype": dtype_name,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:400],
                    "traceback": traceback.format_exc(limit=5),
                }
                # A failure on any dtype marks the node failed; don't overwrite a failure with pass
                if all_node_reports.get(name, {}).get("passed") is not False:
                    all_node_reports[name] = report_entry
                continue

            if ref_out is None:
                if name not in all_node_reports:
                    all_node_reports[name] = {"passed": None, "note": "reference unavailable"}
                continue

            if not isinstance(llm_out, torch.Tensor):
                report_entry = {
                    "passed": False,
                    "dtype": dtype_name,
                    "error": f"returned {type(llm_out).__name__}, expected Tensor",
                }
                if all_node_reports.get(name, {}).get("passed") is not False:
                    all_node_reports[name] = report_entry
                continue

            if llm_out.shape != ref_out.shape:
                report_entry = {
                    "passed": False,
                    "dtype": dtype_name,
                    "error": "shape mismatch",
                    "got_shape": list(llm_out.shape),
                    "expected_shape": list(ref_out.shape),
                }
                if all_node_reports.get(name, {}).get("passed") is not False:
                    all_node_reports[name] = report_entry
                continue

            try:
                ok = bool(torch.allclose(llm_out.float(), ref_out.float(), atol=atol, rtol=rtol))
            except Exception as exc:
                report_entry = {
                    "passed": False,
                    "dtype": dtype_name,
                    "error": f"comparison failed: {exc}",
                }
                if all_node_reports.get(name, {}).get("passed") is not False:
                    all_node_reports[name] = report_entry
                continue

            if ok:
                # Only mark passed if not already failed on another dtype
                if name not in all_node_reports:
                    all_node_reports[name] = {"passed": True}
            else:
                diff = (llm_out.float() - ref_out.float()).abs()
                report_entry = {
                    "passed": False,
                    "dtype": dtype_name,
                    "error": "numerical mismatch",
                    "max_abs": round(float(diff.max().item()), 6),
                    "max_rel": round(
                        float((diff / ref_out.float().abs().clamp(min=1e-8)).max().item()), 6
                    ),
                    "atol": atol,
                    "rtol": rtol,
                }
                if all_node_reports.get(name, {}).get("passed") is not False:
                    all_node_reports[name] = report_entry

    failed = [n for n, r in all_node_reports.items() if r.get("passed") is False]
    passed = [n for n, r in all_node_reports.items() if r.get("passed") is True]
    return {
        "passed": len(failed) == 0,
        "passed_nodes": passed,
        "failed_nodes": failed,
        "total_nodes": len(call_nodes),
        "node_reports": all_node_reports,
    }


def _predecessor_context(node: dict[str, Any], node_index: dict[str, dict]) -> str:
    """Build a short (3-line) context showing only the node's direct predecessors.

    This replaces the full graph_summary in per-node prompts, cutting per-call
    input tokens from ~500 down to ~50 for a graph with 40 nodes.
    """
    preds = node.get("predecessor_ids") or []
    if not preds:
        return ""
    lines = []
    for pred_name in preds:
        pred = node_index.get(pred_name)
        if pred:
            lines.append(
                f"- `{pred_name}` target=`{pred.get('target')}` "
                f"shape={pred.get('output_shape')} dtype={pred.get('output_dtype')}"
            )
        else:
            lines.append(f"- `{pred_name}` (placeholder input)")
    return "\n".join(lines)


def _generate_per_node(
    config: LoweringConfig,
    call_nodes: list[dict],
    save_dir: Path,
    existing: dict[str, str] | None = None,
    only_nodes: list[str] | None = None,
) -> dict[str, str]:
    """One LLM call per graph node.  Returns node_name → kernel_snippet dict.

    Args:
        save_dir:     directory to write per-node files (nodes/<name>/kernel.py)
        existing:     already-generated snippets to keep (repair: skip passing nodes)
        only_nodes:   if set, only (re)generate these node names
    """
    node_index = {n["name"]: n for n in call_nodes}
    save_dir.mkdir(exist_ok=True)
    node_kernels: dict[str, str] = dict(existing or {})
    targets = [n for n in call_nodes if only_nodes is None or n["name"] in only_nodes]
    if config.reuse_existing:
        remaining = []
        reused = 0
        for node in targets:
            name = node["name"]
            kernel_path = save_dir / name / "kernel.py"
            if kernel_path.exists():
                code = kernel_path.read_text(encoding="utf-8")
                if code.strip() and _has_required_kernel_function(code, name):
                    node_kernels[name] = code
                    reused += 1
                    continue
            remaining.append(node)
        targets = remaining
        if reused:
            print(f"  Reusing {reused} existing per-node kernels")

    def _generate_one(i: int, node: dict[str, Any]) -> tuple[str, str]:
        name = node["name"]
        fn_name = f"kernel_{name}"
        node_dir = save_dir / name
        node_dir.mkdir(exist_ok=True)

        node_summary = summarize_node(node)
        graph_context = _predecessor_context(node, node_index)
        prompt = render_op_codegen_prompt(
            node_summary=node_summary,
            fn_name=fn_name,
            graph_context=graph_context,
        )
        (node_dir / "codegen_prompt.md").write_text(prompt, encoding="utf-8")

        if config.dry_run:
            return name, (
                f"def kernel_{name}(*args):\n"
                f"    # dry-run placeholder for {node['target']}\n"
                f"    raise NotImplementedError\n"
            )

        if i > 0 and config.inter_call_delay > 0:
            time.sleep(config.inter_call_delay)

        last_response = ""
        last_code = ""
        for api_attempt in range(1, 4):
            suffix = "" if api_attempt == 1 else f" retry {api_attempt - 1}/2"
            print(f"  Generate [{i+1}/{len(targets)}]: {name}  ({node['target']}){suffix}")
            response = generate_with_openai_compatible_api(
                prompt=prompt,
                system_message=SYSTEM_MESSAGE,
                model=config.model,
                api_base=config.api_base,
                api_key=config.api_key,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                timeout=config.timeout,
            )
            code = _strip_code_fence(response)
            last_response = response
            last_code = code
            if code.strip() and _has_required_kernel_function(code, name):
                (node_dir / "kernel.py").write_text(code, encoding="utf-8")
                (node_dir / "response.txt").write_text(response, encoding="utf-8")
                return name, code
            (node_dir / f"invalid_response_{api_attempt}.txt").write_text(
                response, encoding="utf-8"
            )

        (node_dir / "kernel.py").write_text(last_code, encoding="utf-8")
        (node_dir / "response.txt").write_text(last_response, encoding="utf-8")
        return name, last_code

    if config.parallelism <= 1 or len(targets) <= 1:
        for i, node in enumerate(targets):
            name, code = _generate_one(i, node)
            node_kernels[name] = code
        return node_kernels

    max_workers = max(1, config.parallelism)
    print(f"  Generating {len(targets)} kernels with parallelism={max_workers}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_generate_one, i, node): node["name"]
            for i, node in enumerate(targets)
        }
        for future in as_completed(futures):
            name, code = future.result()
            node_kernels[name] = code

    return node_kernels


def _repair_per_node(
    config: LoweringConfig,
    call_nodes: list[dict],
    node_kernels: dict[str, str],
    node_reports: dict[str, Any],
    failed_nodes: list[str],
    save_dir: Path,
) -> dict[str, str]:
    """Send one targeted repair LLM call per failed kernel.  Returns updated dict."""
    repaired = dict(node_kernels)
    node_index = {n["name"]: n for n in call_nodes}
    save_dir.mkdir(exist_ok=True)

    def _repair_one(i: int, name: str) -> tuple[str, str] | None:
        node = node_index.get(name)
        if node is None:
            return None

        error_report = json.dumps(node_reports.get(name, {}), indent=2)
        prompt = render_op_repair_prompt(
            node_summary=summarize_node(node),
            fn_name=f"kernel_{name}",
            previous_code=node_kernels.get(name, ""),
            error_report=error_report,
        )
        (save_dir / f"repair_{name}_prompt.md").write_text(prompt, encoding="utf-8")

        if config.inter_call_delay > 0 and i > 0:
            time.sleep(config.inter_call_delay)

        last_code = ""
        for api_attempt in range(1, 4):
            suffix = "" if api_attempt == 1 else f" retry {api_attempt - 1}/2"
            print(f"  Repair [{i+1}/{len(failed_nodes)}]: {name}{suffix}")
            response = generate_with_openai_compatible_api(
                prompt=prompt,
                system_message=SYSTEM_MESSAGE,
                model=config.model,
                api_base=config.api_base,
                api_key=config.api_key,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                timeout=config.timeout,
            )
            code = _strip_code_fence(response)
            last_code = code
            if code.strip() and _has_required_kernel_function(code, name):
                (save_dir / f"repair_{name}.py").write_text(code, encoding="utf-8")
                return name, code
            (save_dir / f"repair_{name}_invalid_response_{api_attempt}.txt").write_text(
                response, encoding="utf-8"
            )
        (save_dir / f"repair_{name}.py").write_text(last_code, encoding="utf-8")
        return name, last_code

    if config.parallelism <= 1 or len(failed_nodes) <= 1:
        for i, name in enumerate(failed_nodes):
            result = _repair_one(i, name)
            if result is not None:
                repaired[result[0]] = result[1]
        return repaired

    max_workers = max(1, config.parallelism)
    print(f"  Repairing {len(failed_nodes)} kernels with parallelism={max_workers}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_repair_one, i, name): name
            for i, name in enumerate(failed_nodes)
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                repaired[result[0]] = result[1]

    return repaired


# ── main entry point ──────────────────────────────────────────────────────────


def synthesize_lowering(config: LoweringConfig) -> int:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: extract graph ──────────────────────────────────────────────
    print("Extract: AtenIR backward graph")
    graph_path = _extract_graph(config)
    graph = load_graph(graph_path)
    graph_summary = summarize_graph(graph)
    (config.output_dir / "graph_summary.md").write_text(graph_summary, encoding="utf-8")
    call_nodes = list_call_nodes(graph)
    print(f"Graph has {len(call_nodes)} call_function nodes")

    # ── Step 2: generate one kernel per node ──────────────────────────────
    nodes_dir = config.output_dir / "nodes"
    node_kernels = _generate_per_node(config, call_nodes, save_dir=nodes_dir)

    # ── Step 3: assemble initial kernels.py ───────────────────────────────
    kernels_code = _assemble_kernels_file(node_kernels, call_nodes)
    (config.output_dir / "kernels.py").write_text(kernels_code, encoding="utf-8")
    print(f"Assembled {len(call_nodes)} kernels → {config.output_dir / 'kernels.py'}")

    if config.dry_run:
        return 0

    # ── Step 4: verify → repair loop ─────────────────────────────────────
    for attempt in range(1, config.max_attempts + 1):
        attempt_dir = config.output_dir / f"attempt_{attempt:03d}"
        attempt_dir.mkdir(exist_ok=True)

        attempt_kernels = attempt_dir / "kernels.py"
        attempt_kernels.write_text(kernels_code, encoding="utf-8")

        # Run per-node check: each kernel tested independently with reference inputs
        print(f"Verify [{attempt}/{config.max_attempts}]: running per-node checker")
        report = _verify_per_node(config, attempt_kernels, graph_path)
        (attempt_dir / "per_node_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )

        failed = report.get("failed_nodes", [])
        passed = report.get("passed_nodes", [])
        print(f"  {len(passed)} passed, {len(failed)} failed")
        if failed:
            print(f"  failed: {failed}")

        if report.get("passed"):
            # All kernels correct individually — confirm with full end-to-end graph run
            print("  All kernels pass; running full end-to-end graph check")
            e2e = _verify_kernels(config, attempt_kernels, graph_path)
            (attempt_dir / "e2e_report.json").write_text(
                json.dumps(e2e, indent=2, sort_keys=True), encoding="utf-8"
            )
            if e2e.get("passed"):
                best_dir = config.output_dir / "best"
                best_dir.mkdir(exist_ok=True)
                (best_dir / "kernels.py").write_text(kernels_code, encoding="utf-8")
                (best_dir / "e2e_report.json").write_text(
                    json.dumps(e2e, indent=2, sort_keys=True), encoding="utf-8"
                )
                _save_node_kernels(node_kernels, best_dir)
                print(f"Synthesis passed.  Best kernels: {best_dir / 'kernels.py'}")
                print(f"  Per-node kernels: {best_dir / 'nodes'}/")
                return 0
            # All per-node checks pass but e2e still fails.
            # dispatch.py (per-node reference) diverges from PyTorch autograd
            # (e2e reference) for some ops — small per-node errors compound
            # across 44 nodes. Per-node testing cannot identify which kernels
            # to repair. Stop and report.
            print(
                "  WARNING: all per-node checks pass but full graph fails.\n"
                "  The per-node reference (dispatch.py) diverges from the autograd\n"
                "  reference for some ops and errors compound across nodes.\n"
                f"  Inspect {attempt_dir / 'e2e_report.json'} to see which\n"
                "  gradient outputs are wrong and by how much."
            )
            return 1

        # Some per-node kernels failed — repair only those, one LLM call each
        print(f"  Repairing {len(failed)} kernels individually")
        node_kernels = _repair_per_node(
            config,
            call_nodes,
            node_kernels=node_kernels,
            node_reports=report.get("node_reports", {}),
            failed_nodes=failed,
            save_dir=attempt_dir,
        )

        # Re-assemble with repaired kernels
        kernels_code = _assemble_kernels_file(node_kernels, call_nodes)

    print(f"Synthesis failed after {config.max_attempts} attempts")
    return 1
