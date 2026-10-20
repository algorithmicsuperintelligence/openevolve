"""Evaluate an unchanged AlphaEvolve example; optionally evolve it with an OE config."""

import argparse
import importlib.util
import json
from pathlib import Path

from openevolve import run_evolution
from openevolve.compat.alphaevolve import adapt_evaluator

EXAMPLES = {
    "circle_packing": ("circle_packing_evaluation", "sum_of_radii"),
    "tsp": ("tsp_evaluation_function", "neg_tour_length"),
    "signal_processing": ("signal_processing_evaluation", "overall_score"),
}


def load_evaluator(example_dir):
    """Load the example without importing its cloud client or experiment entrypoint."""
    function_name, objective = EXAMPLES[example_dir.name]
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example_dir", type=Path, help="Path to the upstream example directory")
    parser.add_argument(
        "--config", type=Path, help="OE YAML config; omit to evaluate only (no LLM)"
    )
    parser.add_argument(
        "--iterations", type=int, default=1, help="Evolution iterations with --config"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("alphaevolve_output"))
    args = parser.parse_args()
    example_dir = args.example_dir.resolve()
    if example_dir.name not in EXAMPLES:
        parser.error(f"Supported example names: {', '.join(EXAMPLES)}")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    program_path = example_dir / "src" / "program.py"
    evaluator = load_evaluator(example_dir)
    if args.config is None:
        result = evaluator(program_path)
        print(json.dumps({"metrics": result.metrics, "artifacts": result.artifacts}, indent=2))
        return

    result = run_evolution(
        initial_program=program_path,
        evaluator=evaluator,
        config=args.config,
        iterations=args.iterations,
        output_dir=str(args.output_dir),
        cleanup=True,
    )
    print(json.dumps({"metrics": result.metrics, "best_score": result.best_score}, indent=2))


if __name__ == "__main__":
    main()
