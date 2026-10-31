"""Execute an OE example before and after bundle export/import, guarded offline."""

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import warnings
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    guard = sys.modules.get("sitecustomize")
    if not getattr(guard, "ACTIVE", False):
        raise RuntimeError("Run via test_roundtrip.py with offline tripwires enabled")

    import numpy as np
    from offline_model import fixed_model

    from openevolve.api import run_evolution
    from openevolve.compat.alpha_evolve import load_evaluator
    from openevolve.compat.alphaevolve import adapt_evaluator, config_from_alphaevolve
    from openevolve.compat.alphaevolve_bundle import export_to_alphaevolve
    from openevolve.compat.alphaevolve_launch import _bundle_evaluator
    from openevolve.config import Config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", choices=("function_minimization", "circle_packing"))
    parser.add_argument("workspace", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    example = repo / "examples" / args.example
    config_name = (
        "config.yaml" if args.example == "function_minimization" else "config_phase_1.yaml"
    )
    inputs = args.workspace / "inputs"
    inputs.mkdir()
    for name in ("initial_program.py", "evaluator.py", config_name):
        shutil.copyfile(example / name, inputs / name)
    original_hashes = {
        name: digest(inputs / name) for name in ("initial_program.py", "evaluator.py")
    }
    source = Config.from_yaml(inputs / config_name)
    source.language = "python"
    source.max_iterations = 1  # Bound this test; do not run the example's full budget.
    before = copy.deepcopy(source)
    mapping = (
        {
            "google/gemini-2.0-flash-001": "gemini-2.5-flash",
            "anthropic/claude-3.7-sonnet": "gemini-2.5-pro",
        }
        if args.example == "circle_packing"
        else None
    )
    output = args.workspace / "bundle"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        export_to_alphaevolve(
            inputs / "initial_program.py",
            inputs / "evaluator.py",
            source,
            output,
            title=args.example,
            problem_description=source.prompt.system_message,
            objective_metric="combined_score",
            model_mapping=mapping,
        )
    assert source == before, "Export must not mutate the source configuration"
    export_warnings = [str(item.message) for item in caught]
    relocated = args.workspace / "relocated"
    output.rename(relocated)
    experiment = json.loads((relocated / "experiment.json").read_text())
    settings = json.loads((relocated / "evaluation.json").read_text())
    imported = config_from_alphaevolve(
        experiment, api_base="http://127.0.0.1:1/v1", api_key="roundtrip-placeholder"
    )
    assert imported.max_iterations == source.max_iterations == 1
    assert imported.language == source.language == "python"
    expected_models = [
        ((mapping or {}).get(model.name, model.name), model.weight) for model in source.llm.models
    ]
    assert [(model.name, model.weight) for model in imported.llm.models] == expected_models
    assert imported.prompt.system_message == "Problem description:\n" + source.prompt.system_message
    assert not imported.evaluator.cascade_evaluation
    assert imported.evaluator.parallel_evaluations == 1
    for model in imported.llm.models + imported.llm.evaluator_models:
        assert model.api_base == "http://127.0.0.1:1/v1"
        assert model.api_key == "roundtrip-placeholder"

    # Loading the generated entrypoint exercises its copied evaluator and settings.
    # Remove temporary module registrations BEFORE serialization to real workers.
    direct_evaluate = load_evaluator(inputs / "evaluator.py", "evaluate")
    with _bundle_evaluator(relocated) as candidate_evaluate:
        returned_evaluate = adapt_evaluator(
            candidate_evaluate,
            **{
                "objective_metric": settings["objective_metric"],
                "failure_sentinel": settings["failure_score"],
                "failure_score": settings["failure_score"],
            },
        )
    assert not any(name.startswith("_openevolve_alphaevolve_") for name in sys.modules)
    assert not any(name.startswith("_openevolve_local_ae_evaluator_") for name in sys.modules)
    # Neither run may depend on the original input directory or working directory.
    seed = (relocated / "src/program.py").read_text()
    shutil.rmtree(inputs)
    baseline = copy.deepcopy(source)
    for model, (name, _) in zip(baseline.llm.models, expected_models):
        model.name = name  # The circle example explicitly remaps two provider models.

    results = {}
    for label, config, evaluator in (
        ("direct", baseline, direct_evaluate),
        ("roundtrip", imported, returned_evaluate),
    ):
        # Deliberately align documented losses for a one-proposal comparison.
        # This is NOT equivalence of the original multi-island/cascade search.
        config.database.num_islands = 1
        config.evaluator.cascade_evaluation = False
        config.evaluator.parallel_evaluations = 1
        config.evaluator.use_llm_feedback = False
        config.diff_based_evolution = True
        config.log_level = "WARNING"
        config.evolution_trace.enabled = True
        config.evolution_trace.include_code = True
        config.evolution_trace.include_prompts = False
        config.evolution_trace.buffer_size = 1
        config.evolution_trace.output_path = str(args.workspace / (label + ".jsonl"))
        for model in config.llm.models + config.llm.evaluator_models:
            model.init_client = fixed_model

        def evaluate(path):
            # The function-minimization seed uses global NumPy RNG. Fix its input
            # stream per evaluation on BOTH paths, without editing either source.
            np.random.seed(123)
            worker_guard = sys.modules.get("sitecustomize")
            if not getattr(worker_guard, "ACTIVE", False):
                raise AssertionError("Offline tripwires must be active during evaluation")
            result = evaluator(path)
            with (args.workspace / (label + "-evaluations.jsonl")).open("a") as stream:
                stream.write(json.dumps({"pid": os.getpid(), "hash": digest(Path(path))}) + "\n")
            return result

        result = run_evolution(
            seed if label == "direct" else relocated / "src/program.py",
            evaluate,
            config=config,
            output_dir=str(args.workspace / (label + "-output")),
        )
        results[label] = {
            "best_score": result.best_score,
            "metrics": result.metrics,
            "code": result.best_code,
        }
    copied_hashes = {
        "initial_program.py": digest(relocated / "src/program.py"),
        "evaluator.py": digest(relocated / "src/openevolve_evaluator.py"),
    }
    assert copied_hashes == original_hashes
    assert not inputs.exists() and not output.exists()
    (args.workspace / "roundtrip-result.json").write_text(
        json.dumps(
            {
                "example": args.example,
                "pid": os.getpid(),
                "results": results,
                "models": expected_models,
                "export_warnings": export_warnings,
                "sources_unchanged": True,
                "inputs_removed": True,
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
