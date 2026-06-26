"""CLI for the autograd-pair saved-tensor fusion agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.autograd_pair_fusion_agent.prompts import LAYERNORM_SPEC, OperatorSpec
from pipeline.autograd_pair_fusion_agent.synthesize import (
    AutogradPairConfig,
    synthesize_autograd_pair,
)


def _load_op_spec(path: str) -> OperatorSpec:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    no_grad = data.get("no_grad_inputs", [])
    return OperatorSpec(
        forward_fn_name=data["forward_fn_name"],
        forward_args=data["forward_args"],
        backward_fn_name=data["backward_fn_name"],
        backward_args=data["backward_args"],
        backward_returns=data["backward_returns"],
        forward_semantics=data["forward_semantics"],
        backward_semantics=data["backward_semantics"],
        no_grad_inputs=tuple(no_grad),
        extra_constraints=data.get("extra_constraints", ""),
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autograd-pair saved-tensor fusion agent")
    parser.add_argument("--forward", required=True)
    parser.add_argument(
        "--example-input",
        default="[(8,64) f32, (64) f32, (64) f32]",
        help="AtenIR extraction input spec for the forward reference",
    )
    parser.add_argument("--output-dir", default="autograd_pair_layernorm")
    parser.add_argument("--api-base", default="https://api.openai.com/v1")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--dtype", action="append", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--lowering-context-file", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--op-spec",
        default=None,
        metavar="JSON_FILE",
        help=(
            "Path to a JSON file describing the target operator. "
            "Required fields: forward_fn_name, forward_args, backward_fn_name, "
            "backward_args, backward_returns, forward_semantics, backward_semantics. "
            "Optional fields: no_grad_inputs (list of str), extra_constraints (str). "
            "Defaults to LayerNorm when omitted."
        ),
    )
    parser.add_argument(
        "--evaluator",
        default=None,
        metavar="EVALUATOR_PATH",
        help=(
            "Path to an evaluator script (relative to repo root) used to verify "
            "generated programs. When omitted, verification is skipped and the "
            "first generated attempt is saved directly."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    lowering_context = None
    if args.lowering_context_file:
        lowering_context = Path(args.lowering_context_file).read_text(encoding="utf-8")
    op_spec = _load_op_spec(args.op_spec) if args.op_spec else None
    return synthesize_autograd_pair(
        AutogradPairConfig(
            forward=args.forward,
            example_input=args.example_input,
            output_dir=Path(args.output_dir).resolve(),
            api_base=args.api_base,
            model=args.model,
            api_key=args.api_key,
            max_attempts=args.max_attempts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            dtypes=tuple(args.dtype or ["float32", "float16", "bfloat16"]),
            python=args.python,
            lowering_context=lowering_context,
            dry_run=args.dry_run,
            op_spec=op_spec,
            evaluator_path=args.evaluator,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
