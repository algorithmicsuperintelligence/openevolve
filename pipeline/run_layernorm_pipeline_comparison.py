"""Compare LayerNorm AtenIR agent pipelines under a shared model/budget.

Pipelines:
  A. atenir -> fusion_agent -> OpenEvolve
  B. atenir -> lowering_agent -> OpenEvolve
  C. atenir -> lowering_agent -> kernel_fusion_agent -> OpenEvolve

The script produces one seed program per pipeline, evaluates each seed with the
LayerNorm evaluator, and optionally runs OpenEvolve with the same model for all
pipelines.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

from pipeline.shared.summarize_failure_taxonomy import (
    render_markdown,
    summarize,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_REPO_ROOT = (
    Path("/u/wzhan/openevolve")
    if Path("/u/wzhan/openevolve").exists()
    else REPO_ROOT
)
FORWARD_SPEC = "benchmark.triton_layernorm_backward_bench.forward_ref:layernorm_forward_ref"
EXAMPLE_INPUT = "[(8,64) f32, (64) f32, (64) f32]"
PUBLIC_API = "layernorm_backward_triton"


def _repo_env(primary_repo_root: Path, *extra_repo_roots: Path) -> dict[str, str]:
    env = os.environ.copy()
    paths = [
        str(primary_repo_root),
        str(primary_repo_root / "benchmark"),
        str(primary_repo_root / "examples"),
    ]
    for root in extra_repo_roots:
        paths.extend([str(root), str(root / "benchmark"), str(root / "examples")])
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str],
    continue_on_error: bool = False,
    return_output: bool = False,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logged_cmd = list(cmd)
    for index, value in enumerate(logged_cmd[:-1]):
        if value == "--api-key":
            logged_cmd[index + 1] = "<redacted>"
    start = time.time()
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    write_lock = threading.Lock()

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND:\n" + " ".join(logged_cmd) + "\n\nOUTPUT:\n")
        log_file.flush()

        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )

        def _stream_output(stream, label: str, sink: list[str], terminal) -> None:
            assert stream is not None
            for line in stream:
                sink.append(line)
                terminal.write(line)
                terminal.flush()
                with write_lock:
                    log_file.write(f"[{label}] {line}")
                    log_file.flush()

        stdout_thread = threading.Thread(
            target=_stream_output,
            args=(process.stdout, "stdout", stdout_lines, sys.stdout),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_stream_output,
            args=(process.stderr, "stderr", stderr_lines, sys.stderr),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        returncode = process.wait()
        stdout_thread.join()
        stderr_thread.join()

    elapsed = time.time() - start
    result = {
        "cmd": logged_cmd,
        "returncode": returncode,
        "elapsed_sec": elapsed,
        "log": str(log_path),
    }
    if return_output:
        result["_stdout"] = "".join(stdout_lines)
        result["_stderr"] = "".join(stderr_lines)
    if returncode != 0 and not continue_on_error:
        raise RuntimeError(f"command failed ({returncode}); see {log_path}")
    return result


def _add_agent_common_args(
    cmd: list[str],
    *,
    api_base: str,
    model: str,
    api_key: str | None,
    max_tokens: int,
    temperature: float,
    timeout: int,
    dtypes: list[str],
) -> None:
    cmd.extend(
        [
            "--api-base",
            api_base,
            "--model",
            model,
            "--max-tokens",
            str(max_tokens),
            "--temperature",
            str(temperature),
            "--timeout",
            str(timeout),
        ]
    )
    if api_key:
        cmd.extend(["--api-key", api_key])
    for dtype in dtypes:
        cmd.extend(["--dtype", dtype])


def _run_fusion_agent(
    *,
    out_dir: Path,
    runner_module: str,
    log_name: str,
    api_base: str,
    model: str,
    api_key: str | None,
    max_attempts: int,
    max_tokens: int,
    temperature: float,
    timeout: int,
    dtypes: list[str],
    lowering_context_file: Path | None,
    dry_run: bool,
    env: dict[str, str],
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        runner_module,
        "--forward",
        FORWARD_SPEC,
        "--public-api",
        PUBLIC_API,
        "--op",
        "layernorm",
        "--mode",
        "dynamic",
        "--output-dir",
        str(out_dir),
        "--max-attempts",
        str(max_attempts),
    ]
    _add_agent_common_args(
        cmd,
        api_base=api_base,
        model=model,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        dtypes=dtypes,
    )
    if lowering_context_file:
        cmd.extend(["--lowering-context-file", str(lowering_context_file)])
    if dry_run:
        cmd.append("--dry-run")
    result = _run(
        cmd,
        cwd=REPO_ROOT,
        log_path=out_dir / log_name,
        env=env,
        continue_on_error=True,
    )
    _write_failure_taxonomy_if_available(out_dir)
    return result


def _run_lowering_agent(
    *,
    out_dir: Path,
    api_base: str,
    model: str,
    api_key: str | None,
    max_attempts: int,
    max_tokens: int,
    temperature: float,
    timeout: int,
    dtypes: list[str],
    inter_call_delay: float,
    parallelism: int,
    reuse_existing: bool,
    dry_run: bool,
    env: dict[str, str],
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "pipeline.primitive_atenir_lowering_agent.cli",
        "--forward",
        FORWARD_SPEC,
        "--example-input",
        EXAMPLE_INPUT,
        "--output-dir",
        str(out_dir),
        "--max-attempts",
        str(max_attempts),
        "--inter-call-delay",
        str(inter_call_delay),
        "--parallelism",
        str(parallelism),
    ]
    _add_agent_common_args(
        cmd,
        api_base=api_base,
        model=model,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        dtypes=dtypes,
    )
    if dry_run:
        cmd.append("--dry-run")
    if reuse_existing:
        cmd.append("--reuse-existing")
    result = _run(
        cmd,
        cwd=REPO_ROOT,
        log_path=out_dir / "run_lowering_agent.log",
        env=env,
        continue_on_error=True,
    )
    _write_failure_taxonomy_if_available(out_dir)
    return result


def _write_failure_taxonomy_if_available(out_dir: Path) -> None:
    if not list(out_dir.glob("attempt_*/verification_report.json")):
        return
    summary = summarize(out_dir)
    (out_dir / "failure_taxonomy.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "failure_taxonomy.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )


def _build_scalar_override_source(graph_path: Path) -> str:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    placeholders = [n for n in graph["nodes"] if n.get("op") == "placeholder"]
    x_shape = placeholders[1].get("shape") if len(placeholders) > 1 else None
    trace_rows = x_shape[0] if isinstance(x_shape, list) and len(x_shape) == 2 else 8
    trace_cols = x_shape[1] if isinstance(x_shape, list) and len(x_shape) == 2 else 64
    return f"""
def _scalar_overrides(rows: int, cols: int):
    def replace_traced_dim(value):
        if isinstance(value, int) and not isinstance(value, bool):
            if value == {trace_rows!r}:
                return rows, True
            if value == {trace_cols!r}:
                return cols, True
            return value, False
        if isinstance(value, list):
            changed_any = False
            replaced = []
            for item in value:
                new_item, changed = replace_traced_dim(item)
                replaced.append(new_item)
                changed_any = changed_any or changed
            return replaced, changed_any
        if isinstance(value, tuple):
            changed_any = False
            replaced = []
            for item in value:
                new_item, changed = replace_traced_dim(item)
                replaced.append(new_item)
                changed_any = changed_any or changed
            return tuple(replaced), changed_any
        return value, False

    overrides = {{}}
    for node in _GRAPH["nodes"]:
        if node.get("op") != "call_function":
            continue
        values = []
        changed = False
        for entry in node.get("args_ordered") or []:
            if entry.get("kind") != "scalar":
                continue
            value = entry.get("value")
            new_value, value_changed = replace_traced_dim(value)
            values.append(new_value)
            changed = changed or value_changed
        if changed:
            overrides[node["name"]] = values
    return overrides
"""


def _write_lowering_seed(*, lowering_dir: Path, seed_path: Path) -> Path:
    graph_path = lowering_dir / "atenir_graph.json"
    kernels_path = lowering_dir / "best" / "kernels.py"
    if not graph_path.exists():
        raise FileNotFoundError(f"missing lowering graph: {graph_path}")
    if not kernels_path.exists():
        raise FileNotFoundError(f"missing verified lowering kernels: {kernels_path}")

    override_source = _build_scalar_override_source(graph_path)
    seed = f'''"""OpenEvolve seed generated from per-op AtenIR lowering."""

from __future__ import annotations

import importlib.util
import json
import uuid
from pathlib import Path

import torch

from atenir.compose import run_graph


_GRAPH_PATH = Path({str(graph_path)!r})
_KERNELS_PATH = Path({str(kernels_path)!r})
_GRAPH = json.loads(_GRAPH_PATH.read_text(encoding="utf-8"))


def _load_registry():
    module_name = f"lowering_seed_kernels_{{uuid.uuid4().hex}}"
    spec = importlib.util.spec_from_file_location(module_name, _KERNELS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import lowering kernels from {{_KERNELS_PATH}}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_kernel_registry()


_REGISTRY = _load_registry()
_PLACEHOLDERS = [node["name"] for node in _GRAPH["nodes"] if node.get("op") == "placeholder"]


{textwrap.indent(override_source.strip(), "")}


# EVOLVE-BLOCK-START
def _run_lowered_graph(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
):
    del eps  # eps is already captured in the extracted AtenIR graph.
    values = [
        dy.contiguous(),
        x.contiguous(),
        weight.contiguous(),
        bias.contiguous(),
    ]
    if len(_PLACEHOLDERS) != len(values):
        raise RuntimeError(
            f"expected {{len(_PLACEHOLDERS)}} graph placeholders, got {{len(values)}} runtime inputs"
        )
    env = {{name: value for name, value in zip(_PLACEHOLDERS, values)}}
    rows, cols = x.shape
    return run_graph(str(_GRAPH_PATH), env, _REGISTRY, _scalar_overrides(rows, cols))
# EVOLVE-BLOCK-END


def layernorm_backward_triton(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
):
    dx, dweight, dbias = _run_lowered_graph(dy, x, weight, bias, eps)
    return dx.to(x.dtype), dweight.to(weight.dtype), dbias.to(bias.dtype)
'''
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    seed_path.write_text(seed, encoding="utf-8")
    return seed_path


def _write_lowering_context(
    *,
    lowering_dir: Path,
    context_path: Path,
    max_chars: int,
    per_node_chars: int,
) -> Path:
    graph_summary = (lowering_dir / "graph_summary.md").read_text(encoding="utf-8")
    report_path = lowering_dir / "best" / "e2e_report.json"
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else "{}"
    graph = json.loads((lowering_dir / "atenir_graph.json").read_text(encoding="utf-8"))

    chunks = [
        "# Verified Per-Op Lowering Context",
        "",
        "The per-op lowering agent generated kernels that passed per-node checks and end-to-end autograd verification.",
        "",
        "## AtenIR Graph Summary",
        graph_summary,
        "## E2E Report",
        "```json",
        report[:4000],
        "```",
        "",
        "## Per-Node Kernel Snippets",
    ]
    for node in graph["nodes"]:
        if node.get("op") != "call_function":
            continue
        name = node["name"]
        kernel_path = lowering_dir / "best" / "nodes" / name / "kernel.py"
        if not kernel_path.exists():
            continue
        code = kernel_path.read_text(encoding="utf-8").strip()
        if len(code) > per_node_chars:
            code = code[:per_node_chars] + "\n# ... truncated ..."
        chunks.extend(
            [
                "",
                f"### `{name}` target=`{node.get('target')}`",
                "```python",
                code,
                "```",
            ]
        )

    text = "\n".join(chunks)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n... truncated lowering context ...\n"
    context_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.write_text(text, encoding="utf-8")
    return context_path


def _evaluate_seed(seed_path: Path, out_dir: Path, env: dict[str, str]) -> dict[str, Any]:
    example_dir = Path(env["LAYER_NORM_EVAL_EXAMPLE_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    result = _run(
        [sys.executable, str(example_dir / "evaluator.py"), str(seed_path)],
        cwd=example_dir,
        log_path=out_dir / "initial_evaluator.log",
        env=env,
        continue_on_error=True,
        return_output=True,
    )
    stdout = str(result.pop("_stdout", ""))
    result.pop("_stderr", None)
    try:
        parsed = json.loads(stdout)
    except Exception as exc:
        parsed = {"parse_error": str(exc)}
    result["report"] = parsed
    return result


def _run_openevolve(
    *,
    seed_path: Path,
    out_dir: Path,
    iterations: int,
    api_base: str,
    model: str,
    env: dict[str, str],
) -> dict[str, Any]:
    eval_repo_root = Path(env["LAYER_NORM_EVAL_REPO_ROOT"])
    example_dir = Path(env["LAYER_NORM_EVAL_EXAMPLE_DIR"])
    cmd = [
        sys.executable,
        "-m",
        "openevolve.cli",
        str(seed_path),
        str(example_dir / "evaluator.py"),
        "--config",
        str(example_dir / "config.yaml"),
        "--iterations",
        str(iterations),
        "--output",
        str(out_dir / "openevolve"),
        "--api-base",
        api_base,
        "--primary-model",
        model,
        "--secondary-model",
        model,
    ]
    return _run(
        cmd,
        cwd=eval_repo_root,
        log_path=out_dir / "openevolve.log",
        env=env,
        continue_on_error=True,
    )


def _pipeline_record(name: str, seed_path: Path | None, steps: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "name": name,
        "seed_path": str(seed_path) if seed_path else None,
        "steps": steps,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LayerNorm AtenIR agent pipelines")
    parser.add_argument("--output-dir", default="~/tmp/layernorm_pipeline_comparison")
    parser.add_argument("--api-base", default="https://api.openai.com/v1")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument(
        "--lowering-model",
        default=None,
        help="Model for per-op lowering. Defaults to --model.",
    )
    parser.add_argument("--api-key", default=None, help="default: $OPENAI_API_KEY")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--fusion-max-attempts", type=int, default=10)
    parser.add_argument("--lowering-max-attempts", type=int, default=10)
    parser.add_argument("--fusion-max-tokens", type=int, default=16000)
    parser.add_argument("--lowering-max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--inter-call-delay", type=float, default=2.0)
    parser.add_argument(
        "--lowering-parallelism",
        type=int,
        default=4,
        help="Concurrent per-op LLM calls for the lowering agent (default: 4)",
    )
    parser.add_argument(
        "--reuse-lowering-existing",
        action="store_true",
        help="Reuse valid existing per-op lowering kernels in the output directory",
    )
    parser.add_argument("--dtype", action="append", default=None)
    parser.add_argument("--no-evolve", action="store_true", help="only generate and evaluate seeds")
    parser.add_argument("--dry-run-agents", action="store_true", help="write prompts without LLM calls")
    parser.add_argument(
        "--reuse-existing-seeds",
        action="store_true",
        help=(
            "Skip agent synthesis for pipelines whose seed already exists in output-dir; "
            "still run evaluator/OpenEvolve unless --no-evolve is set."
        ),
    )
    parser.add_argument("--lowering-context-chars", type=int, default=30000)
    parser.add_argument("--lowering-context-per-node-chars", type=int, default=900)
    parser.add_argument(
        "--eval-repo-root",
        default=str(DEFAULT_EVAL_REPO_ROOT),
        help=(
            "Repository root used for final seed evaluation/OpenEvolve. "
            "Defaults to /u/wzhan/openevolve when present; agent synthesis still runs from this worktree."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    out_root = Path(args.output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    eval_repo_root = Path(args.eval_repo_root).expanduser().resolve()
    eval_example_dir = eval_repo_root / "benchmark" / "triton_layernorm_backward_bench"
    synthesis_env = _repo_env(REPO_ROOT)
    eval_env = _repo_env(eval_repo_root, REPO_ROOT)
    eval_env["LAYER_NORM_EVAL_REPO_ROOT"] = str(eval_repo_root)
    eval_env["LAYER_NORM_EVAL_EXAMPLE_DIR"] = str(eval_example_dir)
    api_key = args.api_key or synthesis_env.get("OPENAI_API_KEY")
    if not args.dry_run_agents and not api_key:
        raise RuntimeError(
            "Missing OpenAI-compatible API key. Pass --api-key explicitly or export OPENAI_API_KEY."
        )
    dtypes = args.dtype or ["float32", "float16"]
    lowering_model = args.lowering_model or args.model

    summary: dict[str, Any] = {
        "model": args.model,
        "lowering_model": lowering_model,
        "api_base": args.api_base,
        "iterations": args.iterations,
        "dtypes": dtypes,
        "synthesis_repo_root": str(REPO_ROOT),
        "eval_repo_root": str(eval_repo_root),
        "pipelines": [],
    }

    # A: atenir -> fusion_agent -> OpenEvolve
    fusion_dir = out_root / "A_fusion_agent"
    fusion_seed = fusion_dir / "best" / "initial_program_from_atenir.py"
    steps = []
    if args.reuse_existing_seeds and fusion_seed.exists():
        steps.append({"returncode": 0, "reused_seed": str(fusion_seed)})
    else:
        steps.append(
            _run_fusion_agent(
                out_dir=fusion_dir,
                runner_module="pipeline.fusion_agent.cli",
                log_name="run_fusion_agent.log",
                api_base=args.api_base,
                model=args.model,
                api_key=api_key,
                max_attempts=args.fusion_max_attempts,
                max_tokens=args.fusion_max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
                dtypes=dtypes,
                lowering_context_file=None,
                dry_run=args.dry_run_agents,
                env=synthesis_env,
            )
        )
    if fusion_seed.exists():
        steps.append(_evaluate_seed(fusion_seed, fusion_dir, eval_env))
        if not args.no_evolve:
            steps.append(
                _run_openevolve(
                    seed_path=fusion_seed,
                    out_dir=fusion_dir,
                    iterations=args.iterations,
                    api_base=args.api_base,
                    model=args.model,
                    env=eval_env,
                )
            )
    summary["pipelines"].append(_pipeline_record("A_fusion_agent", fusion_seed, steps))

    # B: atenir -> lowering_agent -> OpenEvolve
    lowering_dir = out_root / "B_lowering_agent"
    lowering_seed = lowering_dir / "initial_program_from_lowering.py"
    lowering_context = lowering_dir / "lowering_context.md"
    steps = []
    if args.reuse_existing_seeds and lowering_seed.exists():
        steps.append({"returncode": 0, "reused_seed": str(lowering_seed)})
    else:
        steps.append(
            _run_lowering_agent(
                out_dir=lowering_dir,
                api_base=args.api_base,
                model=lowering_model,
                api_key=api_key,
                max_attempts=args.lowering_max_attempts,
                max_tokens=args.lowering_max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
                dtypes=dtypes,
                inter_call_delay=args.inter_call_delay,
                parallelism=max(1, args.lowering_parallelism),
                reuse_existing=args.reuse_lowering_existing,
                dry_run=args.dry_run_agents,
                env=synthesis_env,
            )
        )
    if (lowering_dir / "best" / "kernels.py").exists():
        if not (args.reuse_existing_seeds and lowering_seed.exists()):
            _write_lowering_seed(lowering_dir=lowering_dir, seed_path=lowering_seed)
        if not lowering_context.exists():
            _write_lowering_context(
                lowering_dir=lowering_dir,
                context_path=lowering_context,
                max_chars=args.lowering_context_chars,
                per_node_chars=args.lowering_context_per_node_chars,
            )
        steps.append(_evaluate_seed(lowering_seed, lowering_dir, eval_env))
        if not args.no_evolve:
            steps.append(
                _run_openevolve(
                    seed_path=lowering_seed,
                    out_dir=lowering_dir,
                    iterations=args.iterations,
                    api_base=args.api_base,
                    model=args.model,
                    env=eval_env,
                )
            )
    summary["pipelines"].append(_pipeline_record("B_lowering_agent", lowering_seed, steps))

    # C: atenir -> lowering_agent -> kernel_fusion_agent -> OpenEvolve
    hybrid_dir = out_root / "C_lowering_then_kernel_fusion_agent"
    steps = []
    hybrid_seed = hybrid_dir / "best" / "initial_program_from_atenir.py"
    if args.reuse_existing_seeds and hybrid_seed.exists():
        steps.append({"returncode": 0, "reused_seed": str(hybrid_seed)})
        steps.append(_evaluate_seed(hybrid_seed, hybrid_dir, eval_env))
        if not args.no_evolve:
            steps.append(
                _run_openevolve(
                    seed_path=hybrid_seed,
                    out_dir=hybrid_dir,
                    iterations=args.iterations,
                    api_base=args.api_base,
                    model=args.model,
                    env=eval_env,
                )
            )
    elif lowering_context.exists():
        steps.append(
            _run_fusion_agent(
                out_dir=hybrid_dir,
                runner_module="pipeline.kernel_fusion_agent.cli",
                log_name="run_kernel_fusion_agent.log",
                api_base=args.api_base,
                model=args.model,
                api_key=api_key,
                max_attempts=args.fusion_max_attempts,
                max_tokens=args.fusion_max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
                dtypes=dtypes,
                lowering_context_file=lowering_context,
                dry_run=args.dry_run_agents,
                env=synthesis_env,
            )
        )
        if hybrid_seed.exists():
            steps.append(_evaluate_seed(hybrid_seed, hybrid_dir, eval_env))
            if not args.no_evolve:
                steps.append(
                    _run_openevolve(
                        seed_path=hybrid_seed,
                        out_dir=hybrid_dir,
                        iterations=args.iterations,
                        api_base=args.api_base,
                        model=args.model,
                        env=eval_env,
                    )
                )
    else:
        steps.append(
            {
                "returncode": 1,
                "error": "lowering context missing; B_lowering_agent did not produce verified kernels",
            }
        )
    summary["pipelines"].append(
        _pipeline_record("C_lowering_then_kernel_fusion_agent", hybrid_seed, steps)
    )

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote comparison summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
