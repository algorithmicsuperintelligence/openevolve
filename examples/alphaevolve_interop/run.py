"""Evaluate an unchanged AlphaEvolve example; explicitly opt in to model-driven evolution."""

import argparse
import importlib.util
import json
import os
from pathlib import Path

from openevolve import run_evolution
from openevolve.compat.alphaevolve import adapt_evaluator, config_from_alphaevolve

EXAMPLES = {
    "circle_packing": ("circle_packing_evaluation", "sum_of_radii"),
    "tsp": ("tsp_evaluation_function", "neg_tour_length"),
    "signal_processing": ("signal_processing_evaluation", "overall_score"),
}


def load_evaluator(example_dir, *, local_model_types=False):
    """Load the example without importing its cloud client or experiment entrypoint."""
    function_name, objective = EXAMPLES[example_dir.name]
    if local_model_types:
        from openevolve.compat.alpha_evolve import load_evaluator as load_local_evaluator

        evaluator = load_local_evaluator(example_dir / "src" / "evaluate.py", function_name)
        return adapt_evaluator(evaluator, objective_metric=objective)
    spec = importlib.util.spec_from_file_location(
        "_alphaevolve_example", example_dir / "src" / "evaluate.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load evaluator from {example_dir}")
    module = importlib.util.module_from_spec(spec)
    # Deliberately not registered in sys.modules: cloudpickle transports this
    # file-loaded callable by value to OpenEvolve's worker processes.
    spec.loader.exec_module(module)
    return adapt_evaluator(getattr(module, function_name), objective_metric=objective)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example_dir", type=Path, help="Path to the upstream example directory")
    parser.add_argument(
        "--local-model-types",
        action="store_true",
        help="Explicitly load evaluator imports with local envelopes instead of Google's SDK",
    )
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument(
        "--config", type=Path, help="OE YAML config; starts evolution (legacy mode)"
    )
    sources.add_argument("--alphaevolve-config", type=Path, help="AlphaEvolve experiment JSON")
    sources.add_argument("--alphaevolve-env", type=Path, help="Explicit AlphaEvolve .env file")
    parser.add_argument(
        "--instructions", type=Path, help="Problem text file; required with --alphaevolve-env"
    )
    parser.add_argument("--api-base", help="Explicit OE model endpoint for AlphaEvolve imports")
    parser.add_argument(
        "--api-key-env", help="Read the OE API key from this named environment variable"
    )
    parser.add_argument(
        "--evolve",
        action="store_true",
        help="Run imported AlphaEvolve config using the chosen model (may incur charges)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="Positive generation attempts; may only reduce an imported budget (legacy --config default: 1)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("alphaevolve_output"))
    args = parser.parse_args(argv)
    example_dir = args.example_dir.resolve()
    if example_dir.name not in EXAMPLES:
        parser.error(f"Supported example names: {', '.join(EXAMPLES)}")
    imported = args.alphaevolve_config is not None or args.alphaevolve_env is not None
    evolving = args.config is not None or args.evolve
    if args.evolve and not (args.config or imported):
        parser.error("--evolve requires a configuration source")
    if args.iterations is not None and args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.iterations is not None and not evolving:
        parser.error("--iterations requires --evolve or legacy --config")
    if not imported and (
        args.api_base is not None or args.api_key_env is not None or args.instructions is not None
    ):
        parser.error("--api-base, --api-key-env and --instructions require an AlphaEvolve source")
    if imported and not args.api_base:
        parser.error("--api-base is required for AlphaEvolve imports")
    if args.alphaevolve_env is not None and args.instructions is None:
        parser.error("--instructions is required with --alphaevolve-env")
    if args.instructions is not None and args.alphaevolve_env is None:
        parser.error(
            "--instructions is only used with --alphaevolve-env; JSON includes problem_description"
        )

    config = args.config
    if imported:
        api_key = None
        if args.api_key_env is not None:
            api_key = os.environ.get(args.api_key_env)
            if not api_key or not api_key.strip():
                parser.error("--api-key-env must name a non-empty environment variable")
        try:
            if args.alphaevolve_env is not None:
                from openevolve.compat.alphaevolve_env import config_from_alphaevolve_env

                config = config_from_alphaevolve_env(
                    args.alphaevolve_env,
                    problem_description=args.instructions.read_text(encoding="utf-8"),
                    api_base=args.api_base,
                    api_key=api_key,
                )
            else:
                config = config_from_alphaevolve(
                    json.loads(args.alphaevolve_config.read_text(encoding="utf-8-sig")),
                    api_base=args.api_base,
                    api_key=api_key,
                )
        except (ValueError, OSError, ImportError) as exc:
            parser.error(str(exc))
        if args.iterations is not None and args.iterations > config.max_iterations:
            parser.error("--iterations cannot exceed the imported generation budget")

    program_path = example_dir / "src" / "program.py"
    for path in (program_path, example_dir / "src" / "evaluate.py"):
        if not path.is_file():
            parser.error(f"Required example file not found: {path}")
    evaluator = load_evaluator(example_dir, local_model_types=args.local_model_types)
    if not evolving:
        result = evaluator(program_path)
        print(json.dumps({"metrics": result.metrics, "artifacts": result.artifacts}, indent=2))
        return

    result = run_evolution(
        initial_program=program_path,
        evaluator=evaluator,
        config=config,
        iterations=args.iterations if imported else (args.iterations or 1),
        output_dir=str(args.output_dir),
        cleanup=True,
    )
    print(json.dumps({"metrics": result.metrics, "best_score": result.best_score}, indent=2))


if __name__ == "__main__":
    main()
