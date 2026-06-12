"""CLI for the hand-written context kernel fusion pipeline (Pipeline E).

Extracts the AtenIR backward graph, generates a lowering context from the
verified hand-written dispatch table (dispatch.py), then runs the
kernel_fusion_agent LLM loop with that context as grounding.

Unlike Pipeline C (which requires Pipeline B to have completed first), Pipeline E
generates its context deterministically from dispatch.py — no prior LLM run needed.

Example:

    python -m pipeline.run_handwritten_fusion_agent \\
        --forward benchmark.triton_layernorm_backward_bench.forward_ref:layernorm_forward_ref \\
        --example-input "[(8,64) f32, (64) f32, (64) f32]" \\
        --output-dir /tmp/E_layernorm \\
        --model gpt-4o \\
        --dtype float32 --dtype float16

Dry-run (writes prompts + context without calling the LLM):

    python -m pipeline.run_handwritten_fusion_agent \\
        --forward ... --example-input "..." --output-dir /tmp/dry --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline.fusion_agent.synthesize import FusionConfig
from pipeline.handwritten_fusion_agent.synthesize import synthesize_handwritten_fusion


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline E — AtenIR kernel fusion with hand-written dispatch context",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--forward", required=True, help="Forward callable spec, e.g. pkg.mod:fn")
    parser.add_argument(
        "--example-input",
        required=True,
        help='Input spec for graph extraction, e.g. "[(8,64) f32, (64) f32, (64) f32]"',
    )
    parser.add_argument("--public-api", default="layernorm_backward_triton")
    parser.add_argument("--op", default="layernorm")
    parser.add_argument("--mode", default="dynamic", choices=["static", "dynamic", "nontile"])
    parser.add_argument("--output-dir", default="atenir_handwritten_fusion")
    parser.add_argument("--api-base", default="https://api.openai.com/v1")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--dtype", action="append", default=None)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--fp16-atol", type=float, default=5e-2)
    parser.add_argument("--fp16-rtol", type=float, default=5e-2)
    parser.add_argument("--scalar", action="append", default=["1e-5"])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    return synthesize_handwritten_fusion(
        FusionConfig(
            forward=args.forward,
            public_api=args.public_api,
            op=args.op,
            mode=args.mode,
            output_dir=Path(args.output_dir).resolve(),
            api_base=args.api_base,
            model=args.model,
            api_key=args.api_key,
            max_attempts=args.max_attempts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            dtypes=tuple(args.dtype or ["float32", "float16"]),
            atol=args.atol,
            rtol=args.rtol,
            fp16_atol=args.fp16_atol,
            fp16_rtol=args.fp16_rtol,
            scalar_args=tuple(args.scalar),
            python=args.python,
            dry_run=args.dry_run,
        ),
        example_input=args.example_input,
    )


if __name__ == "__main__":
    raise SystemExit(main())
