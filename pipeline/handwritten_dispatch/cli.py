"""CLI for the deterministic hand-written dispatch pipeline (Pipeline D).

No LLM required. Extracts the AtenIR backward graph, builds a kernel registry
from atenir.primitive_triton.dispatch.make_registry(), runs the graph, and
compares outputs against torch.autograd.grad.

Also writes lowering_context.md that Pipeline E can consume directly.

Example:

    python -m pipeline.run_handwritten_dispatch \\
        --forward atenir._examples:silu_mlp \\
        --example-input "[(2048,4096) f32, (4096,8192) f32, (8192,2048) f32]" \\
        --output-dir /tmp/D_silu_mlp \\
        --dtype float32 --dtype float16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline.handwritten_dispatch.synthesize import (
    HandwrittenDispatchConfig,
    synthesize_handwritten_dispatch,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline D — deterministic AtenIR verification via hand-written dispatch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--forward", required=True, help="Forward callable spec, e.g. pkg.mod:fn")
    parser.add_argument(
        "--example-input",
        required=True,
        help='Input spec for graph extraction, e.g. "[(8,64) f32, (64) f32, (64) f32]"',
    )
    parser.add_argument("--output-dir", default="atenir_handwritten_dispatch")
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    return synthesize_handwritten_dispatch(
        HandwrittenDispatchConfig(
            forward=args.forward,
            example_input=args.example_input,
            output_dir=Path(args.output_dir).resolve(),
            dtypes=tuple(args.dtype or ["float32"]),
            atol=args.atol,
            rtol=args.rtol,
            fp16_atol=args.fp16_atol,
            fp16_rtol=args.fp16_rtol,
            python=args.python,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
