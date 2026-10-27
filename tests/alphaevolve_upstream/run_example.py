"""Execute one original upstream main through the public CLI, guarded offline."""

import argparse
import contextlib
import io
import json
import os
import runpy
import sys
from pathlib import Path
from unittest.mock import patch


def main():
    guard = sys.modules.get("sitecustomize")
    if not getattr(guard, "ACTIVE", False):
        raise RuntimeError("Run via test_examples.py with offline tripwires enabled")

    from offline_model import fixed_model

    from openevolve.compat.alphaevolve import config_from_alphaevolve

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", type=Path)
    parser.add_argument("iterations", type=int, choices=(0, 1))
    parser.add_argument("workspace", type=Path)
    args = parser.parse_args()
    runner = Path(__file__).resolve().parents[2] / "examples/alphaevolve_interop/run_entrypoint.py"
    argv = [
        str(runner),
        str(args.example),
        "--api-base",
        "http://127.0.0.1:1/v1",
        "--model",
        "fixed-offline-model",
        "--output-dir",
        str(args.workspace / "run"),
    ]
    if args.iterations:
        argv += ["--evolve", "--iterations", "1"]

    def configure(*values, **options):
        config = config_from_alphaevolve(*values, **options)
        if config.llm.api_key != "local-placeholder":
            raise AssertionError("Runner must not use the dummy ambient key")
        config.database.num_islands = 1
        config.log_level = "WARNING"
        config.evolution_trace.enabled = True
        config.evolution_trace.include_code = True
        config.evolution_trace.include_prompts = False
        config.evolution_trace.buffer_size = 1
        config.evolution_trace.output_path = str(args.workspace / "evolution.jsonl")
        # Seed-only runs construct the real provider client but make no requests.
        # Evolution uses real OE workers with only proposal generation replaced.
        if args.iterations:
            for model in config.llm.models + config.llm.evaluator_models:
                model.init_client = fixed_model
        return config

    stdout = io.StringIO()
    with (
        patch.object(sys, "argv", argv),
        patch(
            "openevolve.compat.alpha_evolve.experiment.config_from_alphaevolve",
            side_effect=configure,
        ),
        contextlib.redirect_stdout(stdout),
    ):
        runpy.run_path(str(runner), run_name="__main__")
    result = json.loads(stdout.getvalue())
    if any(name == "alpha_evolve" or name.startswith("alpha_evolve.") for name in sys.modules):
        raise AssertionError("SDK aliases leaked after main")
    if any(name.startswith("_openevolve_ae_entrypoint_") for name in sys.modules):
        raise AssertionError("Temporary upstream package leaked after main")
    (args.workspace / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (args.workspace / "process.json").write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")


if __name__ == "__main__":
    main()
