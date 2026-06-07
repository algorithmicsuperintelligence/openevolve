"""Orchestration for AtenIR backward fusion synthesis (extract -> fuse -> verify)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from pipeline.shared.llm_client import (
    generate_with_openai_compatible_api,
)
from pipeline.fusion_agent.graph_summary import summarize_graph_file
from pipeline.fusion_agent.prompts import (
    SYSTEM_MESSAGE,
    render_codegen_prompt,
    render_fusion_plan_prompt,
    render_repair_prompt,
)


@dataclass(frozen=True)
class FusionConfig:
    forward: str
    public_api: str
    op: str
    mode: str
    output_dir: Path
    api_base: str
    model: str
    api_key: str | None
    max_attempts: int
    max_tokens: int
    temperature: float | None
    timeout: int
    dtypes: tuple[str, ...]
    atol: float
    rtol: float
    fp16_atol: float
    fp16_rtol: float
    scalar_args: tuple[str, ...]
    python: str
    lowering_context: str | None = None
    dry_run: bool = False


def _strip_code_fence(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip() + "\n"
    return text.strip() + "\n"


def _extract_graph(config: FusionConfig) -> Path:
    graph_path = config.output_dir / "atenir_graph.json"
    cmd = [
        config.python,
        "-m",
        "atenir.extract",
        "--fn",
        config.forward,
        "--example-input",
        "[(8,64) f32, (64) f32, (64) f32]",
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
        raise RuntimeError(f"AtenIR extraction failed: {completed.stderr}")
    return graph_path


def _tolerance_for_dtype(config: FusionConfig, dtype: str) -> tuple[float, float]:
    if dtype in {"float16", "fp16", "bfloat16", "bf16"}:
        return config.fp16_atol, config.fp16_rtol
    return config.atol, config.rtol


def _verify_program_for_dtype(config: FusionConfig, program_path: Path, dtype: str) -> dict:
    atol, rtol = _tolerance_for_dtype(config, dtype)
    cmd = [
        config.python,
        "-m",
        "tests.atenir_correctness.run_correctness",
        "--forward",
        config.forward,
        "--backward-file",
        str(program_path),
        "--backward-fn",
        config.public_api,
        "--mode",
        config.mode,
        "--atol",
        str(atol),
        "--rtol",
        str(rtol),
        "--dtype",
        dtype,
    ]
    for scalar in config.scalar_args:
        cmd.extend(["--scalar", scalar])
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
    return report


def _verify_program(config: FusionConfig, program_path: Path) -> dict:
    reports = [_verify_program_for_dtype(config, program_path, dtype) for dtype in config.dtypes]
    passed_cases = sum(report.get("passed_cases", 0) for report in reports)
    total_cases = sum(report.get("total_cases", 0) for report in reports)
    passed = all(report.get("passed") for report in reports)
    return {
        "passed": passed,
        "passed_cases": passed_cases,
        "failed_cases": total_cases - passed_cases,
        "total_cases": total_cases,
        "dtypes": list(config.dtypes),
        "reports": reports,
    }


def synthesize_fusion(config: FusionConfig) -> int:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    print("Extract: AtenIR backward graph")
    graph_path = _extract_graph(config)
    graph_summary = summarize_graph_file(graph_path)
    (config.output_dir / "graph_summary.md").write_text(graph_summary, encoding="utf-8")
    lowering_context = config.lowering_context or ""
    if lowering_context:
        (config.output_dir / "lowering_context.md").write_text(
            lowering_context, encoding="utf-8"
        )

    plan_prompt = render_fusion_plan_prompt(
        forward=config.forward,
        public_api=config.public_api,
        graph_summary=graph_summary,
        lowering_context=lowering_context,
    )
    (config.output_dir / "fusion_plan_prompt.md").write_text(plan_prompt, encoding="utf-8")

    if config.dry_run:
        code_prompt = render_codegen_prompt(
            public_api=config.public_api,
            graph_summary=graph_summary,
            fusion_plan="{FUSION_PLAN_FROM_LLM}",
            lowering_context=lowering_context,
        )
        dry_dir = config.output_dir / "attempt_001"
        dry_dir.mkdir(parents=True, exist_ok=True)
        (dry_dir / "codegen_prompt.md").write_text(code_prompt, encoding="utf-8")
        print(f"dry-run wrote {config.output_dir / 'graph_summary.md'}")
        print(f"dry-run wrote {config.output_dir / 'fusion_plan_prompt.md'}")
        print(f"dry-run wrote {dry_dir / 'codegen_prompt.md'}")
        return 0

    print("Fusion: plan synthesis")
    fusion_plan = generate_with_openai_compatible_api(
        prompt=plan_prompt,
        system_message=SYSTEM_MESSAGE,
        model=config.model,
        api_base=config.api_base,
        api_key=config.api_key,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        timeout=config.timeout,
    )
    (config.output_dir / "fusion_plan.md").write_text(fusion_plan, encoding="utf-8")

    prompt = render_codegen_prompt(
        public_api=config.public_api,
        graph_summary=graph_summary,
        fusion_plan=fusion_plan,
        lowering_context=lowering_context,
    )
    previous_code = ""

    for attempt in range(1, config.max_attempts + 1):
        attempt_dir = config.output_dir / f"attempt_{attempt:03d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "codegen_prompt.md").write_text(prompt, encoding="utf-8")
        print(f"Fusion: codegen attempt {attempt}/{config.max_attempts}")
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
        program_path = attempt_dir / "program.py"
        program_path.write_text(code, encoding="utf-8")
        (attempt_dir / "response.txt").write_text(response, encoding="utf-8")

        report = _verify_program(config, program_path)
        (attempt_dir / "verification_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if report.get("passed"):
            best_dir = config.output_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            best_path = best_dir / "initial_program_from_atenir.py"
            best_path.write_text(code, encoding="utf-8")
            (best_dir / "verification_report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(f"Fusion synthesis passed. Best program: {best_path}")
            return 0

        verifier_report = json.dumps(report, indent=2, sort_keys=True)
        repair_prompt = render_repair_prompt(
            public_api=config.public_api,
            graph_summary=graph_summary,
            fusion_plan=fusion_plan,
            previous_code=code or previous_code,
            verifier_report=verifier_report,
            lowering_context=lowering_context,
        )
        (attempt_dir / "repair_prompt.md").write_text(repair_prompt, encoding="utf-8")
        prompt = repair_prompt
        previous_code = code
        print(f"attempt {attempt} failed; wrote repair prompt")

    print(f"Fusion synthesis failed after {config.max_attempts} attempts")
    return 1
