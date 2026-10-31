"""Preflight an exported Python bundle; explicitly opt in to a paid cloud run."""

import argparse
import asyncio
import contextlib
import importlib
import importlib.util
import json
import math
import re
import sys
import uuid
from pathlib import Path

from openevolve.compat.alphaevolve_bundle import _read_python
from openevolve.compat.alphaevolve_export import _gemini_name, _positive_integer
from openevolve.compat.alphaevolve_reverse import adapt_openevolve_evaluator


def _object(value, required, optional=(), *, label):
    if (
        not isinstance(value, dict)
        or not set(required).issubset(value)
        or set(value) - set(required) - set(optional)
    ):
        raise ValueError(f"{label} must contain only the supported bundle fields")
    return value


def _read_json(path):
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    json.dumps(value, allow_nan=False)  # Reject non-standard NaN/Infinity input.
    return value


def _prepare_bundle(root):
    config = _object(
        _read_json(root / "experiment.json"),
        ("title", "problem_description", "program_language", "run_settings", "generation_settings"),
        label="experiment.json",
    )
    if config["program_language"] != "python":
        raise ValueError("Only single-file Python bundles are supported")
    for field in ("title", "problem_description"):
        if not isinstance(config[field], str) or not config[field].strip():
            raise ValueError(f"{field} must be non-empty text")
    run = _object(config["run_settings"], ("max_programs",), ("concurrency",), label="run_settings")
    _positive_integer(run["max_programs"], "run_settings.max_programs")
    if "concurrency" in run:
        _positive_integer(run["concurrency"], "run_settings.concurrency")
    generation = _object(config["generation_settings"], ("models",), label="generation_settings")
    models = generation["models"]
    if not isinstance(models, list) or not 1 <= len(models) <= 2:
        raise ValueError("Bundle requires one or two explicitly named Gemini models")
    weights = []
    for model in models:
        _object(model, ("name", "weight"), label="generation model")
        _gemini_name(model["name"], "generation model name")
        weight = model["weight"]
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight <= 0
        ):
            raise ValueError("Generation model weights must be finite and positive")
        weights.append(weight)
    try:
        if not math.isfinite(math.fsum(weights)):
            raise ValueError("Generation model weights must have a finite sum")
    except OverflowError:
        raise ValueError("Generation model weights must have a finite sum") from None

    settings = _object(
        _read_json(root / "evaluation.json"),
        ("objective_metric", "failure_score"),
        label="evaluation.json",
    )
    adapt_openevolve_evaluator(lambda path: {}, **settings)
    source = _read_python(root / "src/program.py").decode("utf-8")
    _read_python(root / "src/openevolve_evaluator.py", evaluator=True)
    for file in ("src/__init__.py", "src/evaluate.py"):
        try:
            compile((root / file).read_bytes(), file, "exec")
        except SyntaxError:
            raise ValueError(f"Bundle file must contain valid Python: {file}") from None
    candidate = {"content": {"files": [{"path": "program.py", "content": source}]}}
    return config, candidate


@contextlib.contextmanager
def _bundle_evaluator(root):
    # A private package supports relative imports and dataclasses without
    # replacing an application's existing `src` package or retaining old bundles.
    name = "_openevolve_alphaevolve_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(
        name, root / "src/__init__.py", submodule_search_locations=[str(root / "src")]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        evaluate = importlib.import_module(name + ".evaluate").evaluate
        if not callable(evaluate):
            raise TypeError("Bundle entry point must expose a callable evaluate")
        yield evaluate
    finally:
        for key in list(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]


def _run(root, config, candidate, args):
    # Optional SDK imports occur only after --run and all local input checks.
    from alpha_evolve.client import AlphaEvolveClient
    from alpha_evolve.controller import run_controller_loop
    from alpha_evolve.experiment import AlphaEvolveExperiment
    from alpha_evolve.models import AlphaEvolveExperimentConfig, AlphaEvolveProgramEvaluation

    AlphaEvolveExperimentConfig.model_validate(config)
    stage = "local seed evaluation"
    try:
        with _bundle_evaluator(root) as evaluate:
            seed_evaluation = evaluate(candidate)
            AlphaEvolveProgramEvaluation.model_validate(seed_evaluation)
            if any(
                insight.get("label") == "OpenEvolve Evaluation Error"
                for insight in seed_evaluation.get("insights", {}).get("insights", [])
            ):
                raise ValueError("Seed evaluation failed")
            initial = {**candidate, "evaluation": seed_evaluation}
            stage = "session/experiment creation"
            client = AlphaEvolveClient(
                project_id=args.project_id, engine=args.engine, location=args.location
            )
            experiment = AlphaEvolveExperiment(
                client,
                evaluate,
                max_programs_evaluated=args.max_programs_evaluated,
                parallel_evaluation=False,
            )
            experiment.create_experiment(config)
            print(f"Cloud experiment: {experiment.experiment_name}", file=sys.stderr)
            stage = "seed upload"
            experiment.create_initial_program(initial)
            if not experiment.initial_program_name:
                raise RuntimeError("SDK did not confirm seed creation")
            stage = "experiment start"
            # The experiment helper discards this response; use the public
            # client method so an HTTP failure cannot silently start polling.
            if client.start_experiment(experiment.experiment_name) is None:
                raise RuntimeError("SDK did not confirm experiment start")
            stage = "controller execution"
            asyncio.run(
                run_controller_loop(
                    experiment,
                    num_samplers=1,
                    num_evaluators=1,
                    idle_timeout_s=args.idle_timeout_seconds,
                )
            )
            reached = experiment.stats["num_programs_evaluated"] >= args.max_programs_evaluated
            return {
                "mode": "run",
                "experiment_name": experiment.experiment_name,
                "stats": experiment.stats,
                "evaluation_target_reached": reached,
            }
    except Exception as exc:
        raise RuntimeError(
            f"Cloud launch failed during {stage} ({type(exc).__name__}). "
            "Inspect SDK logs; any created cloud resources are not automatically deleted."
        ) from None


def _positive_option(value):
    try:
        return _positive_integer(int(value), "option")
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None


def _resource_id(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise argparse.ArgumentTypeError(
            "must be a non-empty resource ID, without slashes or spaces"
        )
    return value


def main(argv=None, *, bundle_dir=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if bundle_dir is None:
        parser.add_argument("bundle_dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Offline preflight only (default)")
    mode.add_argument(
        "--run", action="store_true", help="Create/start a cloud experiment; may incur charges"
    )
    parser.add_argument("--project-id", type=_resource_id)
    parser.add_argument("--engine", type=_resource_id)
    parser.add_argument("--location", choices=("global", "us", "eu"), default="global")
    parser.add_argument("--max-programs-evaluated", type=_positive_option)
    parser.add_argument("--idle-timeout-seconds", type=_positive_option, default=120)
    args = parser.parse_args(argv)
    if args.run and not (args.project_id and args.engine and args.max_programs_evaluated):
        parser.error("--run requires --project-id, --engine and --max-programs-evaluated")
    root = Path(bundle_dir if bundle_dir is not None else args.bundle_dir).resolve()
    try:
        config, candidate = _prepare_bundle(root)
        if (
            args.max_programs_evaluated is not None
            and args.max_programs_evaluated > config["run_settings"]["max_programs"]
        ):
            raise ValueError(
                "Evaluation target cannot exceed the exported generation-attempt budget"
            )
    except (OSError, TypeError, ValueError, OverflowError) as exc:
        parser.error(str(exc))
    if not args.run:
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "bundle_dir": str(root),
                    "title": config["title"],
                    "models": config["generation_settings"]["models"],
                    "max_generation_attempts": config["run_settings"]["max_programs"],
                    "max_programs_evaluated": args.max_programs_evaluated,
                    "project_id": args.project_id,
                    "engine": args.engine,
                    "location": args.location,
                    "sdk_and_evaluator_imported": False,
                },
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
        )
        return 0
    try:
        # Redirect Python-level evaluator and SDK chatter away from JSON stdout.
        with contextlib.redirect_stdout(sys.stderr):
            result = _run(root, config, candidate, args)
    except ImportError:
        parser.exit(
            1,
            "Install the official AlphaEvolve SDK and evaluator dependencies; see the bundle README.\n",
        )
    except Exception as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    return 0 if result["evaluation_target_reached"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
