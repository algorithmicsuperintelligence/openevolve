"""CLI for the AtenIR per-op Triton kernel lowering agent.

Example:

    python -m pipeline.primitive_atenir_lowering_agent.cli \\
        --forward benchmark.triton_layernorm_backward_bench.forward_ref:layernorm_forward_ref \\
        --example-input "[(8,64) f32, (64) f32, (64) f32]" \\
        --output-dir /tmp/lowering_layernorm \\
        --api-base https://api.openai.com/v1 \\
        --model gpt-4o \\
        --dtype float32 --dtype float16

Dry-run (writes prompts and graph summary without calling the LLM):

    python -m pipeline.primitive_atenir_lowering_agent.cli \\
        --forward benchmark.triton_layernorm_backward_bench.forward_ref:layernorm_forward_ref \\
        --example-input "[(8,64) f32, (64) f32, (64) f32]" \\
        --output-dir /tmp/lowering_dry \\
        --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline.primitive_atenir_lowering_agent.synthesize import (
    LoweringConfig,
    synthesize_lowering,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AtenIR per-op Triton kernel lowering agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--forward",
        required=True,
        help="Forward callable spec, e.g. pkg.mod:fn",
    )
    parser.add_argument(
        "--example-input",
        required=True,
        help='Example input spec for graph extraction, e.g. "[(8,64) f32, (64) f32, (64) f32]"',
    )
    parser.add_argument("--output-dir", default="atenir_lowering_output")
    parser.add_argument("--api-base", default="https://api.openai.com/v1")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--api-key", default=None, help="LLM API key (default: $OPENAI_API_KEY)")
    parser.add_argument("--max-attempts", type=int, default=3, help="Repair attempts after first assembly")
    parser.add_argument(
        "--max-tokens", type=int, default=4096,
        help="Max output tokens per LLM call. Reduce to cut TPM usage (default: 4096)",
    )
    parser.add_argument(
        "--inter-call-delay", type=float, default=0.0,
        help=(
            "Seconds to sleep between LLM calls in per_node mode. "
            "Formula: 60 * avg_tokens_per_call / tpm_limit. "
            "Example for 30k TPM: 60 * 3000 / 30000 = 6.0"
        ),
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="Number of concurrent per-op LLM calls for generation/repair (default: 1)",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse non-empty existing nodes/<name>/kernel.py files that define kernel_<name>",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--dtype",
        action="append",
        default=None,
        choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"],
        help="Data type(s) to verify (repeatable; default: float32)",
    )
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--fp16-atol", type=float, default=5e-2)
    parser.add_argument("--fp16-rtol", type=float, default=5e-2)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write prompts and graph summary without calling the LLM",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    dtypes = tuple(args.dtype or ["float32"])
    return synthesize_lowering(
        LoweringConfig(
            forward=args.forward,
            example_input=args.example_input,
            output_dir=Path(args.output_dir).resolve(),
            api_base=args.api_base,
            model=args.model,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            dtypes=dtypes,
            atol=args.atol,
            rtol=args.rtol,
            fp16_atol=args.fp16_atol,
            fp16_rtol=args.fp16_rtol,
            max_attempts=args.max_attempts,
            python=args.python,
            inter_call_delay=args.inter_call_delay,
            parallelism=max(1, args.parallelism),
            reuse_existing=args.reuse_existing,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
