"""CLI for kernel-aware AtenIR backward fusion synthesis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline.fusion_agent.synthesize import FusionConfig
from pipeline.kernel_fusion_agent.synthesize import (
    synthesize_kernel_fusion,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AtenIR Kernel-Aware Fusion Agent")
    parser.add_argument("--forward", required=True)
    parser.add_argument("--public-api", default="layernorm_backward_triton")
    parser.add_argument("--op", default="layernorm")
    parser.add_argument("--mode", default="dynamic", choices=["static", "dynamic", "nontile"])
    parser.add_argument("--output-dir", default="atenir_kernel_fusion_layernorm")
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
    parser.add_argument(
        "--lowering-context-file",
        required=True,
        help="Markdown/text file with verified per-op Triton lowering context",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    dtypes = tuple(args.dtype or ["float32", "float16"])
    lowering_context = Path(args.lowering_context_file).read_text(encoding="utf-8")
    return synthesize_kernel_fusion(
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
            dtypes=dtypes,
            atol=args.atol,
            rtol=args.rtol,
            fp16_atol=args.fp16_atol,
            fp16_rtol=args.fp16_rtol,
            scalar_args=tuple(args.scalar),
            python=args.python,
            lowering_context=lowering_context,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
