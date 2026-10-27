"""Run supported upstream entrypoints locally with explicit configuration swaps."""

import argparse
import contextlib
import importlib
import json
import os
import shutil
import sys
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from openevolve.compat.alpha_evolve import (
    AlphaEvolveClient,
    AlphaEvolveExperiment,
    local_imports,
)

EXAMPLES = {
    "circle_packing": "sum_of_radii",
    "tsp": "neg_tour_length",
    "signal_processing": "overall_score",
}


def _load_entrypoint(root, settings):
    """Import relative helpers, then unregister them for by-value worker transport."""
    try:
        import dotenv
    except ImportError as error:
        raise ImportError("Install python-dotenv and the upstream example dependencies") from error

    name = "_openevolve_ae_entrypoint_" + uuid.uuid4().hex
    package = ModuleType(name)
    package.__path__ = [str(root)]
    package.__package__ = name
    sys.modules[name] = package
    try:
        # The caller supplies OE settings explicitly. Do not discover/load .env
        # files while importing the original entrypoint's configuration globals.
        with (
            local_imports(),
            patch.object(dotenv, "load_dotenv", return_value=False),
            patch.dict(os.environ, settings),
        ):
            module = importlib.import_module(name + ".src.run_evolution")
        return module
    finally:
        for key in list(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]


def _validate_source(example_dir):
    example_dir = Path(example_dir).resolve()
    if example_dir.name not in EXAMPLES:
        raise ValueError("Supported upstream entrypoints: " + ", ".join(EXAMPLES))
    source = example_dir / "src"
    required = [source / name for name in ("run_evolution.py", "program.py", "evaluate.py")]
    if example_dir.name == "tsp":
        required.append(example_dir / "instructions.md")
    for path in required:
        if not path.is_file():
            raise ValueError("Missing required upstream file: " + str(path))
    if source.is_symlink() or any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError("Example src must contain regular files/directories, not symlinks")
    instructions = example_dir / "instructions.md"
    if instructions.is_symlink():
        raise ValueError("instructions.md must not be a symlink")
    return example_dir


def run_entrypoint(example_dir, *, api_base, model, api_key=None, iterations=0, output_dir=None):
    """Run trusted, byte-identical staged sources; default to seed evaluation only.

    Only the three reviewed entrypoints are supported. Imports are process-global;
    use this example runner in a fresh, single-threaded main process. Generation
    uses the chosen endpoint and may incur charges when iterations is positive.
    Reports operate on the facade's best-only result, not a full search history.
    """
    example_dir = _validate_source(example_dir)
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must explicitly name a model served by the OE endpoint")
    if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
        raise ValueError("api_key must be a non-empty string or None")
    # Validate options before copying/importing any trusted upstream code.
    options = dict(
        api_base=api_base,
        objective_metric=EXAMPLES[example_dir.name],
        api_key=api_key if api_key is not None else "local-placeholder",
        iterations=iterations,
    )
    AlphaEvolveClient(**options)
    destination = Path(output_dir).absolute() if output_dir is not None else None
    if destination is not None and destination.resolve().is_relative_to(example_dir):
        raise ValueError("output_dir must be outside the upstream example")

    experiments = []

    class SignalProcessingExperiment(AlphaEvolveExperiment):
        def create_experiment(self, config):
            # This upstream entrypoint has no generation_settings.models. Supply
            # the caller's explicit model as a configuration swap, never a default
            # model or an in-place change to the original experiment dictionary.
            if not isinstance(config, Mapping):
                raise ValueError("experiment must be a mapping")
            generation = config.get("generation_settings", {})
            if not isinstance(generation, Mapping):
                raise ValueError("generation_settings must be a mapping")
            selected = dict(generation) | {"models": [{"name": model, "weight": 1.0}]}
            super().create_experiment(dict(config) | {"generation_settings": selected})

    def experiment_factory(*args, **kwargs):
        experiment_type = (
            SignalProcessingExperiment
            if example_dir.name == "signal_processing"
            else AlphaEvolveExperiment
        )
        experiment = experiment_type(*args, **kwargs)
        experiments.append(experiment)
        return experiment

    with contextlib.ExitStack() as stack:
        if destination is None:
            root = Path(
                stack.enter_context(tempfile.TemporaryDirectory(prefix="oe_ae_entrypoint_"))
            )
        else:
            # Exclusive creation prevents reports or copied sources overwriting
            # an existing run. Keep explicit output, including after failure.
            destination.mkdir()
            root = destination
        shutil.copytree(
            example_dir / "src",
            root / "src",
            ignore=shutil.ignore_patterns(".env*", "__pycache__", "*.pyc"),
        )
        instructions = example_dir / "instructions.md"
        if instructions.is_file():
            shutil.copyfile(instructions, root / "instructions.md")
        settings = {
            "MODEL": model,
            "MODEL_1": model,
            "MODEL_2": "",
            "MODEL_1_WEIGHT": "1",
            "MODEL_2_WEIGHT": "0",
            "MAX_PROGRAMS_GENERATED": str(max(1, iterations)),
            "MAX_PROGRAMS_EVALUATED": str(max(1, iterations)),
            "PARALLEL_EVALUATION": "False",
            "CONCURRENCY": "1",
            "WORKER_CONCURRENCY": "1",
        }
        entrypoint = _load_entrypoint(root, settings)
        required = {
            "main",
            "AlphaEvolveClient",
            "AlphaEvolveExperiment",
            "MAX_PROGRAMS_GENERATED",
            "MAX_PROGRAMS_EVALUATED",
            "CONCURRENCY",
        }
        if example_dir.name == "circle_packing":
            required.update(
                {
                    "MODEL_1",
                    "MODEL_2",
                    "MODEL_1_WEIGHT",
                    "MODEL_2_WEIGHT",
                    "WORKER_CONCURRENCY",
                    "PARALLEL_EVALUATION",
                }
            )
        elif example_dir.name == "tsp":
            required.update({"MODEL", "PARALLEL_EVALUATION"})
        else:
            required.add("run_controller_loop")
        if any(not hasattr(entrypoint, key) for key in required) or not callable(entrypoint.main):
            raise ValueError("Upstream entrypoint layout changed; review its configuration first")

        # Override only imported factories and configuration globals. The source
        # files and upstream main/evaluator implementations remain unchanged.
        def client_factory(*args, **kwargs):
            # Explicit OE options take precedence if an upstream constructor
            # starts supplying similarly named settings in a later revision.
            selected = kwargs | options | {"output_dir": str(root / "oe-output")}
            return AlphaEvolveClient(*args, **selected)

        entrypoint.AlphaEvolveClient = client_factory
        entrypoint.AlphaEvolveExperiment = experiment_factory
        entrypoint.MAX_PROGRAMS_GENERATED = max(1, iterations)
        entrypoint.MAX_PROGRAMS_EVALUATED = max(1, iterations)
        entrypoint.PARALLEL_EVALUATION = False
        entrypoint.CONCURRENCY = 1
        if example_dir.name == "circle_packing":
            entrypoint.MODEL_1, entrypoint.MODEL_2 = model, ""
            entrypoint.MODEL_1_WEIGHT, entrypoint.MODEL_2_WEIGHT = 1.0, 0.0
            entrypoint.WORKER_CONCURRENCY = 1
        elif example_dir.name == "tsp":
            entrypoint.MODEL = model

        # The signal evaluator optionally writes archives to an environment path.
        # Keep those artifacts in this run, rather than an inherited destination.
        environment = (
            {"SAVE_PROGRAM_DIR": str(root / "programs")}
            if example_dir.name == "signal_processing"
            else {}
        )
        with patch.dict(os.environ, environment):
            entrypoint.main()
        states = [experiment.state for experiment in experiments]
        # The reviewed signal main first creates/queries a diagnostic experiment,
        # then creates a separate one to run. Do not accept an early caught error
        # or silently discard any other incomplete/failed experiment.
        complete = (
            states == ["CREATED", "COMPLETED"]
            if example_dir.name == "signal_processing"
            else bool(states) and all(state == "COMPLETED" for state in states)
        )
        if not complete:
            raise RuntimeError("Upstream entrypoint returned without completing a local experiment")
        result = experiments[-1].result
        return {
            "example": example_dir.name,
            "generation_attempt_limit": iterations,
            "metrics": result.metrics,
            "best_score": result.best_score,
            "output_dir": str(destination) if destination is not None else None,
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example_dir", type=Path)
    parser.add_argument("--api-base", required=True, help="Explicit OpenEvolve model endpoint")
    parser.add_argument("--model", required=True, help="Model served by that endpoint")
    parser.add_argument("--api-key-env", help="Read a key from this named environment variable")
    parser.add_argument(
        "--output-dir", type=Path, help="New directory for sources, OE output and reports"
    )
    parser.add_argument(
        "--evolve", action="store_true", help="Opt in to model calls, which may incur charges"
    )
    parser.add_argument(
        "--iterations", type=int, help="Positive generation attempts (default 1 with --evolve)"
    )
    args = parser.parse_args(argv)
    if args.iterations is not None and (not args.evolve or args.iterations < 1):
        parser.error("positive --iterations requires --evolve")
    api_key = None
    if args.api_key_env is not None:
        api_key = os.environ.get(args.api_key_env)
        if not api_key or not api_key.strip():
            parser.error("--api-key-env must name a non-empty environment variable")
    # Original reports/import-time prints go to stderr; stdout contains the summary.
    with contextlib.redirect_stdout(sys.stderr):
        result = run_entrypoint(
            args.example_dir,
            api_base=args.api_base,
            model=args.model,
            api_key=api_key,
            iterations=(args.iterations or 1) if args.evolve else 0,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
