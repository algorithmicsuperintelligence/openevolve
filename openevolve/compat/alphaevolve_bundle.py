"""Export a minimal, offline-checkable single-file Python AlphaEvolve bundle."""

import argparse
import ast
import io
import json
import shutil
import tokenize
from pathlib import Path
from typing import Optional, Union

from yaml import YAMLError

from openevolve.compat.alphaevolve_export import config_to_alphaevolve
from openevolve.compat.alphaevolve_reverse import adapt_openevolve_evaluator
from openevolve.config import Config

_EVALUATOR = '''"""AlphaEvolve candidate adapter for the explicitly copied evaluator."""

import json
from pathlib import Path

from openevolve.compat.alphaevolve_reverse import adapt_openevolve_evaluator

from .openevolve_evaluator import evaluate as evaluate_program

settings = json.loads(
    (Path(__file__).resolve().parents[1] / "evaluation.json").read_text(encoding="utf-8")
)
evaluate = adapt_openevolve_evaluator(evaluate_program, **settings)
'''

_CHECK = '''"""Explicitly execute the trusted evaluator against the copied seed."""

import contextlib
import json
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    candidate = {
        "content": {"files": [{
            "path": "program.py",
            "content": (root / "src" / "program.py").read_bytes().decode("utf-8"),
        }]}
    }
    # Redirect Python stdout; evaluators must capture their subprocess output.
    with contextlib.redirect_stdout(sys.stderr):
        from src.evaluate import evaluate

        result = evaluate(candidate)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    failed = any(
        entry.get("label") == "OpenEvolve Evaluation Error"
        for entry in result.get("insights", {}).get("insights", [])
    )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
'''

_LAUNCH = '''"""Offline preflight by default; --run explicitly opts in to cloud use."""

from pathlib import Path

from openevolve.compat.alphaevolve_launch import main

if __name__ == "__main__":
    raise SystemExit(main(bundle_dir=Path(__file__).resolve().parent))
'''

_README = """# Minimal AlphaEvolve evaluator bundle

This directory packages a single Python seed and a synchronous
`evaluate(program_path)` evaluator. It contains:

- `src/program.py`: the seed, copied byte for byte.
- `src/openevolve_evaluator.py`: the evaluator, copied byte for byte.
- `src/evaluate.py`: an AlphaEvolve candidate adapter.
- `evaluation.json`: the objective metric and finite failure score.
- `experiment.json`: the limited AlphaEvolve experiment configuration.
- `check.py`: an explicit local seed-evaluation check.
- `run_alphaevolve.py`: offline preflight or an explicitly requested cloud launch.
- `requirements.txt`, only if explicitly supplied during export.

## Install dependencies

Use Python 3.10 or newer and an OpenEvolve checkout containing
`openevolve.compat.alphaevolve_launch`. Install that checkout in your chosen
environment, for example `python -m pip install /path/to/openevolve`.
The exporter does not infer, install or vendor dependencies. Review and install
your evaluator's packages; if `requirements.txt` is included, review it before
running `python -m pip install -r requirements.txt`. Included requirement files,
local paths and data files are not copied automatically.

Source files are copied verbatim, without secret removal. Review them and any
explicit dependency file before sharing. The source OpenEvolve YAML, endpoints
and API keys are not serialized into the exported experiment configuration.

## Check the seed locally

From this directory, run `python check.py`. You can also run it by absolute path
from another working directory. The result is a JSON evaluation on stdout;
Python-level evaluator/import stdout is redirected to stderr. Evaluators must
capture their own subprocess or direct file-descriptor output to keep stdout
clean. Adapter-reported failures return a nonzero exit status; ordinary import
or contract errors also fail the command.

This command executes trusted evaluator and candidate code with the current
process's privileges. It is not a sandbox or a timeout wrapper. Any file/network
access performed by that code remains its responsibility. Evaluators that turn
failures into ordinary numeric scores cannot be distinguished from success.

## Use the real AlphaEvolve SDK

With the official SDK installed and this directory on Python's module path:

```python
import json
from pathlib import Path

from alpha_evolve.models import AlphaEvolveExperimentConfig
from src.evaluate import evaluate

config = json.loads(Path("experiment.json").read_text(encoding="utf-8"))
AlphaEvolveExperimentConfig.model_validate(config)  # Offline schema check.
# Pass `evaluate` as evaluator_function to your configured AlphaEvolveExperiment.
```

## Preflight or explicitly launch an experiment

Run `python run_alphaevolve.py` (or add `--dry-run`) for an offline JSON summary.
The script resolves the bundle relative to itself, including from another working
directory. It checks the exported configuration subset, evaluation options and
source files without importing the SDK or evaluator, executing the seed,
discovering credentials or making requests. This basic check does not validate
installed dependencies, SDK schemas, permissions or live model availability.
Use `check.py` separately to execute and test your trusted evaluator locally.

For cloud use, install the official SDK from a reviewed checkout of
https://github.com/Google-Cloud-AI/alphaevolve-on-googlecloud, for example
`python -m pip install /path/to/alphaevolve-on-googlecloud`. Follow that project's
README for Google application-default credentials, the GCP project, Gemini
Enterprise engine and access requirements. Choose models currently supported
by your service; the exporter's Gemini-name syntax check is not a model catalog.

After reviewing the exported generation budget, explicitly opt in:

```bash
python run_alphaevolve.py --run \\
  --project-id YOUR_PROJECT_ID --engine YOUR_ENGINE_ID \\
  --max-programs-evaluated 1
```

Only `--run` imports the optional SDK and trusted evaluator. It validates with
the real SDK and evaluates the seed before constructing a client, then creates
a new experiment, uploads the seed with its actual scores, starts generation
and invokes the SDK controller with one sampler and one synchronous evaluator.
`--location` accepts `global` (default), `us` or `eu`. Credentials use the SDK's
normal Google ADC flow; no key, endpoint or .env discovery is added by the runner.

The evaluation target is required for `--run` and cannot exceed the exported
generation-attempt budget. It is a controller stopping target, not a strict
evaluation cap: SDK polling/queued work can overshoot it, and seed evaluation
is additional. Generation limits are not cost limits. The positive
`--idle-timeout-seconds` (default 120) is an idle threshold, not a wall-clock
timeout, sandbox, cloud cancellation or money limit.

The final run JSON includes the experiment name, SDK statistics and whether the
evaluation target was reached. An unmet target or launch failure exits nonzero.
SDK diagnostics go to stderr. Cloud operations may incur charges. Each `--run`
creates a new experiment; reruns do not resume an earlier experiment. Failures
or interruptions do not delete or stop cloud resources automatically; inspect
the reported experiment in your project. No managed-service run is claimed by
the export or offline checks.

## Scope

The exporter checks UTF-8 Python syntax, requires a module-level synchronous
`evaluate` definition, and rejects relative imports and discoverable sibling
modules. It cannot detect all dynamic imports, external data or working-directory
assumptions. Resolve those dependencies before use and run the local check.
The evaluator must accept one path argument without additional required arguments.

The reverse adapter retains numeric metrics, handles explicit error/timeout and
non-finite results, and removes temporary candidate files. It omits artifacts and
does not reproduce OpenEvolve's evaluator pipeline, retries, LLM feedback,
cascade stages, resource limits or search behavior. This is a limited export
slice of OpenEvolve #468, not a complete SDK shim or search-behavior translation.
"""


def _read_python(path: Path, *, evaluator: bool = False) -> bytes:
    if path.suffix.lower() != ".py":
        raise ValueError("Bundle sources must be .py files")
    source = path.read_bytes()
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(source).readline)
        if encoding not in {"utf-8", "utf-8-sig"}:
            raise ValueError("Bundle sources must use UTF-8")
        source.decode("utf-8-sig")
        tree = ast.parse(source)
        compile(tree, "<bundle source>", "exec")  # Validate scopes without executing code.
    except (SyntaxError, UnicodeError):
        raise ValueError("Bundle sources must be valid UTF-8 Python") from None

    if evaluator:
        definitions = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "evaluate"
        ]
        if not definitions or not isinstance(definitions[-1], ast.FunctionDef):
            raise ValueError("Evaluator must define a module-level synchronous evaluate function")
        args = definitions[-1].args
        positional = len(args.posonlyargs) + len(args.args)
        if (
            positional - len(args.defaults) > 1
            or (not positional and args.vararg is None)
            or any(default is None for default in args.kw_defaults)
        ):
            raise ValueError("Evaluator must accept one path without other required arguments")
    for node in ast.walk(tree):
        modules = []
        if isinstance(node, ast.ImportFrom):
            if node.level:
                raise ValueError("Relative imports are not supported in single-file bundles")
            modules = [node.module]
        elif isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        for module in modules:
            name = module.split(".")[0]
            if (path.parent / (name + ".py")).is_file() or (path.parent / name).is_dir():
                raise ValueError("Sibling module imports are not supported in single-file bundles")
    return source


def export_to_alphaevolve(
    initial_program: Union[str, Path],
    evaluator: Union[str, Path],
    config: Config,
    out_dir: Union[str, Path],
    *,
    title: str,
    problem_description: str,
    objective_metric: str,
    failure_score: float = -1e12,
    model_mapping=None,
    generation_concurrency: Optional[int] = None,
    requirements_file: Optional[Union[str, Path]] = None,
) -> Path:
    """Write a new minimal Python bundle without importing/executing source code.

    Reuse the explicit config export and reverse evaluator contracts. Copy only
    the two named UTF-8 .py files and an optional explicit dependency file. Reject
    existing destinations (including symlinks); their parent must already exist.
    Validate inputs before creating output and clean up new output on write errors.
    No dependencies, data, local modules or credentials are inferred. The generated
    launcher defaults to offline preflight; cloud use requires explicit --run.
    Read the generated README and explicitly run check.py after installing deps.
    """
    destination = Path(out_dir).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Bundle destination must not already exist")
    sources = {
        "src/program.py": _read_python(Path(initial_program)),
        "src/openevolve_evaluator.py": _read_python(Path(evaluator), evaluator=True),
    }
    # Validate adapter options without loading the supplied evaluator.
    adapt_openevolve_evaluator(
        lambda path: {}, objective_metric=objective_metric, failure_score=failure_score
    )
    experiment = config_to_alphaevolve(
        config,
        title=title,
        problem_description=problem_description,
        model_mapping=model_mapping,
        generation_concurrency=generation_concurrency,
    )
    settings = {"objective_metric": objective_metric, "failure_score": float(failure_score)}
    for name, value in (("experiment.json", experiment), ("evaluation.json", settings)):
        sources[name] = (
            json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        ).encode("utf-8")
    sources.update(
        {
            "src/__init__.py": b"",
            "src/evaluate.py": _EVALUATOR.encode("utf-8"),
            "check.py": _CHECK.encode("utf-8"),
            "run_alphaevolve.py": _LAUNCH.encode("utf-8"),
            "README.md": _README.encode("utf-8"),
        }
    )
    if requirements_file is not None:
        dependencies = Path(requirements_file).read_bytes()
        dependencies.decode("utf-8-sig")
        sources["requirements.txt"] = dependencies

    # mkdir is exclusive, so a pre-existing directory is never owned or removed.
    destination.mkdir()
    try:
        (destination / "src").mkdir()
        for name, data in sources.items():
            (destination / name).write_bytes(data)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("initial_program", type=Path)
    parser.add_argument("evaluator", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--objective-metric", required=True)
    parser.add_argument("--failure-score", type=float, default=-1e12)
    parser.add_argument("--model-map", action="append", default=[], metavar="SOURCE=GEMINI")
    parser.add_argument("--generation-concurrency", type=int)
    parser.add_argument("--requirements-file", type=Path)
    args = parser.parse_args(argv)
    try:
        mapping = {}
        for entry in args.model_map:
            source, separator, target = entry.partition("=")
            if not separator or not source or not target or source in mapping:
                raise ValueError("--model-map requires unique SOURCE=GEMINI entries")
            mapping[source] = target
        config = Config.from_yaml(args.config)
        if config.language is None:
            config.language = "python"  # Both source inputs must be validated .py files.
        output = export_to_alphaevolve(
            args.initial_program,
            args.evaluator,
            config,
            args.output_dir,
            title=args.title,
            problem_description=args.instructions.read_text(encoding="utf-8"),
            objective_metric=args.objective_metric,
            failure_score=args.failure_score,
            model_mapping=mapping,
            generation_concurrency=args.generation_concurrency,
            requirements_file=args.requirements_file,
        )
    except (OSError, TypeError, ValueError, YAMLError) as exc:
        parser.error(str(exc))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
