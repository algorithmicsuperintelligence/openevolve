"""CLI for Stage 1 Triton backward benchmark construction artifacts."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

try:
    from .builder.candidate_verifier import non_gpu_commands, verification_commands
    from .builder.prompt_templates import (
        check_oracle_source,
        render_all_prompts,
        render_oracle_guidance,
    )
    from .builder.spec import load_benchmark_spec
    from .builder.synthesize import SynthesizeConfig, infer_candidate_fn_name, synthesize
except ImportError:  # pragma: no cover - supports direct script execution
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from benchmark.triton_backward_bench_common.builder.candidate_verifier import (
        non_gpu_commands,
        verification_commands,
    )
    from benchmark.triton_backward_bench_common.builder.prompt_templates import (
        check_oracle_source,
        render_all_prompts,
        render_oracle_guidance,
    )
    from benchmark.triton_backward_bench_common.builder.spec import load_benchmark_spec
    from benchmark.triton_backward_bench_common.builder.synthesize import (
        SynthesizeConfig,
        infer_candidate_fn_name,
        synthesize,
    )


def _task_dir(value: str) -> Path:
    path = Path(value).resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"task directory does not exist: {path}")
    if not (path / "meta.yaml").exists():
        raise argparse.ArgumentTypeError(f"task directory must contain meta.yaml: {path}")
    return path


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    print(f"wrote {path}")


def emit_spec(task_dir: Path) -> None:
    spec = load_benchmark_spec(task_dir)
    output = task_dir / "stage1_spec.yaml"
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(spec.to_stage1_dict(), handle, sort_keys=False)
    print(f"wrote {output}")


def emit_prompts(task_dir: Path) -> None:
    spec = load_benchmark_spec(task_dir)
    prompts_dir = task_dir / "prompts"
    for filename, content in render_all_prompts(spec).items():
        _write_text(prompts_dir / filename, content)
    _write_text(prompts_dir / "autograd_oracle_guidance.md", render_oracle_guidance(spec))


def check_oracle(task_dir: Path) -> int:
    oracle_path = task_dir / "backward_ref.py"
    if not oracle_path.exists():
        print(f"ERROR: missing {oracle_path}")
        return 1
    warnings = check_oracle_source(oracle_path.read_text(encoding="utf-8"))
    if warnings:
        print(f"Oracle warnings for {oracle_path}:")
        for warning in warnings:
            print(f"- {warning}")
        return 1
    print(f"oracle check passed: {oracle_path}")
    return 0


def verify(task_dir: Path, python: str, run: bool) -> int:
    commands = non_gpu_commands(task_dir, python) if run else verification_commands(task_dir, python)
    if not run:
        print("Suggested verification commands:")
        for command in commands:
            print("\n" + command)
        return 0

    for command in commands:
        print(f"running: {command}")
        completed = subprocess.run(command, shell=True, cwd=os.getcwd(), check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 Triton backward benchmark builder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ["emit-spec", "emit-prompts", "check-oracle"]:
        subparser = subparsers.add_parser(command)
        subparser.add_argument("task_dir", type=_task_dir)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("task_dir", type=_task_dir)
    verify_parser.add_argument("--python", default=sys.executable)
    verify_parser.add_argument("--run", action="store_true", help="Run non-GPU safety checks")

    synth_parser = subparsers.add_parser("synthesize")
    synth_parser.add_argument("task_dir", type=_task_dir)
    synth_parser.add_argument("--api-base", default="https://api.openai.com/v1")
    synth_parser.add_argument("--model", default="gpt-5.5")
    synth_parser.add_argument("--api-key", default=None)
    synth_parser.add_argument("--max-attempts", type=int, default=3)
    synth_parser.add_argument("--max-tokens", type=int, default=20000)
    synth_parser.add_argument("--temperature", type=float, default=0.2)
    synth_parser.add_argument("--timeout", type=int, default=180)
    synth_parser.add_argument("--output-dir", default="stage1_candidates")
    synth_parser.add_argument("--candidate-fn-name", default=None)
    synth_parser.add_argument("--python", default=sys.executable)
    synth_parser.add_argument("--max-cases", type=int, default=None)
    synth_parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.command == "emit-spec":
        emit_spec(args.task_dir)
        return 0
    if args.command == "emit-prompts":
        emit_prompts(args.task_dir)
        return 0
    if args.command == "check-oracle":
        return check_oracle(args.task_dir)
    if args.command == "verify":
        return verify(args.task_dir, args.python, args.run)
    if args.command == "synthesize":
        spec = load_benchmark_spec(args.task_dir)
        candidate_fn_name = args.candidate_fn_name or infer_candidate_fn_name(spec.operator)
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = args.task_dir / output_dir
        return synthesize(
            SynthesizeConfig(
                task_dir=args.task_dir,
                output_dir=output_dir,
                api_base=args.api_base,
                model=args.model,
                api_key=args.api_key,
                max_attempts=args.max_attempts,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
                candidate_fn_name=candidate_fn_name,
                python=args.python,
                max_cases=args.max_cases,
                dry_run=args.dry_run,
            )
        )
    raise ValueError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
