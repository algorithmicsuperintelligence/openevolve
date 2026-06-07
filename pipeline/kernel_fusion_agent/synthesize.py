"""Orchestration for kernel-aware AtenIR fusion synthesis."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.shared.llm_client import (
    generate_with_openai_compatible_api,
)
from pipeline.fusion_agent.graph_summary import summarize_graph_file
from pipeline.fusion_agent.synthesize import (
    FusionConfig,
    _extract_graph,
    _strip_code_fence,
    _verify_program,
)
from pipeline.kernel_fusion_agent.prompts import (
    SYSTEM_MESSAGE,
    render_kernel_fusion_codegen_prompt,
    render_kernel_fusion_plan_prompt,
    render_kernel_fusion_repair_prompt,
)


def synthesize_kernel_fusion(config: FusionConfig) -> int:
    """Synthesize a fused program using verified per-op Triton kernels as context."""
    lowering_context = config.lowering_context or ""
    if not lowering_context:
        raise ValueError("kernel-aware fusion requires non-empty lowering_context")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    print("Extract: AtenIR backward graph")
    graph_path = _extract_graph(config)
    graph_summary = summarize_graph_file(graph_path)
    (config.output_dir / "graph_summary.md").write_text(graph_summary, encoding="utf-8")
    (config.output_dir / "lowering_context.md").write_text(
        lowering_context, encoding="utf-8"
    )

    plan_prompt = render_kernel_fusion_plan_prompt(
        forward=config.forward,
        public_api=config.public_api,
        graph_summary=graph_summary,
        lowering_context=lowering_context,
    )
    (config.output_dir / "kernel_fusion_plan_prompt.md").write_text(
        plan_prompt, encoding="utf-8"
    )

    if config.dry_run:
        code_prompt = render_kernel_fusion_codegen_prompt(
            public_api=config.public_api,
            graph_summary=graph_summary,
            fusion_plan="{KERNEL_FUSION_PLAN_FROM_LLM}",
            lowering_context=lowering_context,
        )
        dry_dir = config.output_dir / "attempt_001"
        dry_dir.mkdir(parents=True, exist_ok=True)
        (dry_dir / "codegen_prompt.md").write_text(code_prompt, encoding="utf-8")
        print(f"dry-run wrote {config.output_dir / 'graph_summary.md'}")
        print(f"dry-run wrote {config.output_dir / 'kernel_fusion_plan_prompt.md'}")
        print(f"dry-run wrote {dry_dir / 'codegen_prompt.md'}")
        return 0

    print("Kernel fusion: plan synthesis")
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
    (config.output_dir / "kernel_fusion_plan.md").write_text(
        fusion_plan, encoding="utf-8"
    )

    prompt = render_kernel_fusion_codegen_prompt(
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
        print(f"Kernel fusion: codegen attempt {attempt}/{config.max_attempts}")
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
            print(f"Kernel fusion synthesis passed. Best program: {best_path}")
            return 0

        verifier_report = json.dumps(report, indent=2, sort_keys=True)
        repair_prompt = render_kernel_fusion_repair_prompt(
            public_api=config.public_api,
            graph_summary=graph_summary,
            fusion_plan=fusion_plan,
            previous_code=code or previous_code,
            verifier_report=verifier_report,
            lowering_context=lowering_context,
        )
        (attempt_dir / "repair_prompt.md").write_text(
            repair_prompt, encoding="utf-8"
        )
        prompt = repair_prompt
        previous_code = code
        print(f"attempt {attempt} failed; wrote repair prompt")

    print(f"Kernel fusion synthesis failed after {config.max_attempts} attempts")
    return 1
