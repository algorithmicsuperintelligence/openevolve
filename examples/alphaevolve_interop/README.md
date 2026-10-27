# AlphaEvolve interoperability

This walkthrough covers opt-in evaluator, configuration and Python bundle slices of
[#468](https://github.com/algorithmicsuperintelligence/openevolve/issues/468).
It runs the **unchanged** `program.py` and `evaluate.py` from Google's
[AlphaEvolve examples](https://github.com/Google-Cloud-AI/alphaevolve-on-googlecloud)
through an explicit adapter. A reverse adapter also lets a synchronous OpenEvolve
evaluator score AlphaEvolve candidates. Separate opt-in helpers import a limited
subset of experiment configuration for local Python runs and export a limited
configuration subset in the other direction. A minimal bundle exporter combines
these reverse helpers with explicitly selected Python sources and a local seed
check. These helpers do not replace the cloud SDK or implement all of #468.

## Use a local experiment facade

The first SDK-entrypoint slice is explicitly namespaced under
`openevolve.compat.alpha_evolve`. It provides a small, synchronous lifecycle
backed by OpenEvolve's public `run_evolution` API, without installing Google's
SDK or reading Google credentials. Importing the facade does **not** register a
top-level `alpha_evolve` alias or change imports in other code. Scoped aliases
require the explicit opt-in described below.

This self-contained example runs only a seed by default:

```python
import asyncio

from openevolve.compat.alpha_evolve import (
    AlphaEvolveClient,
    AlphaEvolveExperiment,
    run_controller_loop,
)


def evaluate(candidate):
    namespace = {}
    exec(candidate["content"]["files"][0]["content"], namespace)
    return {"scores": {"scores": [{"metric": "quality", "score": namespace["x"]}]}}


client = AlphaEvolveClient(
    api_base="http://localhost:8000/v1",  # Explicit OE endpoint, not Google base_url.
    api_key="local-placeholder",  # The provider client requires a non-empty key, even for a seed.
    objective_metric="quality",
    iterations=0,  # Seed only. Use 1 to opt in to one generation attempt.
)
experiment = AlphaEvolveExperiment(client, evaluate, max_programs_evaluated=2)
experiment.create_experiment({
    "program_language": "python",
    "problem_description": "Maximize x while preserving the interface.",
    "run_settings": {"max_programs": 2},
    "generation_settings": {"models": [{"name": "my-local-model"}]},
})
experiment.create_initial_program({
    "content": {"files": [{
        "path": "program.py",
        "content": "# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END\n",
    }]},
})
experiment.start_experiment()  # Blocks until the local OE run completes.
asyncio.run(run_controller_loop(experiment))  # Completion check; no second run.
print(experiment.list_programs(params={"order_by": "quality desc"}))
```

Use model names actually served by the explicit endpoint. Set `api_key` on the
local client if required; omitting it retains OE's usual provider behavior,
including `OPENAI_API_KEY`. With `iterations=0`, seed evaluation executes but
no generation is requested. With positive iterations (or `None`, which uses
the imported budget), `start_experiment` invokes the configured provider and
may incur charges. Creation alone does not execute the evaluator, create files
or construct provider clients. Candidate/evaluator code is trusted, with no
additional sandbox. Call `start_experiment` on the main thread outside an
already-running asyncio loop; OpenEvolve installs signal handlers.

The client accepts the six Google connection arguments (`project_id`,
`location`, `collection`, `engine`, `assistant`, `base_url`) only to ease import
migration. Supplied values produce one warning per construction and are ignored;
they are never used as the local endpoint or printed in that warning. The
existing configuration importer supplies model names/weights, problem text and
generation budget. Unsupported fields, including cloud generation concurrency,
still warn; they are not equated with OE evaluator concurrency. Only
`parallel_evaluation=False` is supported by this initial facade.

As with the explicit `.env` importer, `max_programs_evaluated` must equal
`run_settings.max_programs` in this slice. **Neither is a strict evaluation cap:**
the resulting OE limit counts generation attempts, with seed evaluation and
retries additional. Distinct budgets are rejected. Explicit `iterations` may
only reduce the generation budget, including zero. No evaluation counts are
invented or presented as SDK statistics.

The seed must be a mapping containing exactly one non-empty UTF-8 Python source
file; pass SDK objects as `model_dump()` dictionaries. Its path is metadata,
never a write target. Supplied seed scores are ignored with a warning: OE
re-evaluates the exact source. The state progresses from `NEW` to `CREATED`,
`READY`, then `RUNNING` and `COMPLETED` (or `FAILED`). A run cannot be restarted,
including after interruption; construct a new instance instead. Configuration
and candidate validation errors before execution can be corrected in place.
The facade's own copied seed is cleaned up on success and failure. Optional
`output_dir` preserves OE's normal run artifacts; without it, OE uses temporary
output. The completed `EvolutionResult` is available as `experiment.result`.

`list_programs` returns **only the best result**, using the SDK's
`alphaEvolvePrograms` content/evaluation envelope. It is not an archive listing;
there is no pagination. `order_by="<returned metric> asc|desc"` is accepted for
entrypoint migration but has no effect on a one-item result. The generic `score`
ordering names the client's explicit objective, not an average of metrics.
Unknown metrics or other query parameters raise errors. OE's internal `combined_score` is omitted
unless it was explicitly selected as the objective; numeric objective/feature
scores, including negatives, are retained. An error metric adds a generic
failure insight. `run_controller_loop` only checks successful completion.
Explicit positive `num_samplers` and `num_evaluators` integers are accepted with
a warning that they are ignored: the synchronous OE run has already finished,
and no workers are created or concurrency changed. Other cloud controller
options, including idle timeouts, remain unsupported.

`client.get_alpha_evolve_experiment(experiment_name=...)` is an in-memory lookup
of experiments created by that client. It returns a fresh `{"name": ..., "state": ...}`
dictionary, including for a created-but-not-started diagnostic experiment.
State reflects the local lifecycle; it is not a cloud resource/status schema.
Unknown names and names from another client raise `ValueError`. No configuration,
credentials, program data or requests are involved; the client retains its local
experiments for the client's lifetime.

The scoped runner below now executes the three reviewed original entrypoints
with explicit configuration/factory swaps. Complete archive access, other
remote client/controller methods and the full SDK model surface remain
unsupported. No managed Google Cloud run is implied by a local facade run.

```bash
python -m unittest discover -s tests -p 'test_alphaevolve_local_experiment.py' -v
```

## Load evaluator imports without Google's SDK

Five dependency-free evaluation envelopes now cover the local examples'
`alpha_evolve.models` imports: `AlphaEvolveEvaluationScore`,
`AlphaEvolveEvaluationScores`, `AlphaEvolveEvaluationInsight`,
`AlphaEvolveEvaluationInsights` and `AlphaEvolveProgramEvaluation`. They live in
`openevolve.compat.alpha_evolve.models`. These are dataclasses with attribute
access and a no-argument `model_dump()` returning independent nested dictionaries,
including optional `None` fields. They do **not** implement Pydantic validation,
coercion, schemas, `model_validate` or serialization options. The existing
evaluator adapter continues to validate metrics and map failures.

Load an unchanged, trusted upstream evaluator with the local types explicitly:

```python
from pathlib import Path
from openevolve.compat.alpha_evolve import load_evaluator
from openevolve.compat.alphaevolve import adapt_evaluator

example = Path("../alphaevolve-reference/examples/circle_packing")
evaluate = load_evaluator(example / "src/evaluate.py", "circle_packing_evaluation")
adapted = adapt_evaluator(evaluate, objective_metric="sum_of_radii")
print(adapted(example / "src/program.py").metrics)
# Or pass evaluate to the local AlphaEvolveExperiment from the previous section.
```

The helper reads and executes the selected `.py` file, including its import-time
code, under temporary local aliases. It returns the raw candidate-based callable
and removes both the aliases and temporary file-module registration. Registration
during import supports decorators such as `@dataclass`; removal afterwards lets
cloudpickle transport the evaluator by value to fresh OE workers, whose local
envelope classes remain importable under the `openevolve` namespace. No SDK alias
needs to remain active during evolution. The helper uses the current source
bytes rather than timestamp/size-based cached bytecode.

Evaluator dependencies such as NumPy, SciPy and Matplotlib must still be
installed. Relative/sibling imports are not managed. Import-time side effects
outside the temporary module entries are not undone, and candidate/evaluator
code is trusted with the current process's privileges. This adds no sandbox.

For explicit import migration, `local_imports` exposes temporary `client`,
`experiment`, `controller`, `models` and `visualization` modules:

```python
from openevolve.compat.alpha_evolve import local_imports

with local_imports():
    from alpha_evolve.client import AlphaEvolveClient
    from alpha_evolve.visualization import get_score
# Existing references still work; sys.modules aliases have been removed.
```

This context is process-global, main-thread-only and intended for a
single-threaded entrypoint, not a concurrent service. It refuses to enter if
**any** `alpha_evolve` namespace is already loaded, even a nested local context;
use a fresh process rather than shadowing a real SDK. An installed but unimported
SDK is temporarily bypassed only inside the explicit context. Unknown SDK
submodules do not fall back to the cloud package. Exit, including on exception
or interruption, removes only module entries still owned by the context; it
never overwrites replacements made by other code. It does not install packages,
change import paths or credentials, or construct clients. The aliased local
client still requires explicit OE endpoint/objective settings. For evaluator
files, prefer `load_evaluator` so worker serialization does not depend on aliases.

`openevolve.compat.alpha_evolve.visualization.get_score(program, metric)` reads
mapping or attribute-based result envelopes without plotting or executing code.
It preserves finite scores (including zero and negatives), returning `-inf` for
absent, malformed, boolean or non-finite scores.

The existing CLI also accepts `--local-model-types` as a scoped opt-in, including
with seed-only evaluation or imported/OE evolution configs. Without that flag,
its existing Google model imports remain unchanged:

```bash
# Requires the example's scientific dependencies, but no Google's SDK source path.
MPLBACKEND=Agg python examples/alphaevolve_interop/run.py \
  ../alphaevolve-reference/examples/circle_packing --local-model-types

python -m unittest discover -s tests -p 'test_alphaevolve_local_imports.py' -v
```

These types close the local evaluator-type/import gap, not the complete SDK shim
by themselves. They neither enable cloud execution nor change the facade's
budget, concurrency or best-only query limits.

## Run the original local-evaluation entrypoints

`run_entrypoint.py` executes the upstream `src/run_evolution.py`'s original
`main()` for **circle_packing**, **tsp** and **signal_processing** with explicit
local connection/model settings. It stages byte-identical copies of `src/` and optional
`instructions.md`, imports their relative helpers under temporary local aliases,
and swaps imported client/experiment factories and configuration globals. It
does not rewrite the source, replace the evaluator, or contact Google's SDK.

From an installed OE checkout, with a reviewed upstream checkout as above:

```bash
python -m pip install numpy scipy 'matplotlib>=3.9.1,<3.11' nest-asyncio python-dotenv

# Run the original TSP main, including its seed scoring and report generation.
# No generation requests. The output directory must be new and outside the example.
MPLBACKEND=Agg python examples/alphaevolve_interop/run_entrypoint.py \
  ../alphaevolve-reference/examples/tsp \
  --api-base http://localhost:8000/v1 --model my-local-model \
  --output-dir ./tsp-local-seed

# Explicitly opt in to one model-driven generation attempt; may incur charges.
MPLBACKEND=Agg python examples/alphaevolve_interop/run_entrypoint.py \
  ../alphaevolve-reference/examples/circle_packing \
  --api-base http://localhost:8000/v1 --model my-local-model \
  --evolve --iterations 1 --output-dir ./circle-local-run

# Run the original signal-processing main without model or Google Cloud requests.
python examples/alphaevolve_interop/run_entrypoint.py \
  ../alphaevolve-reference/examples/signal_processing \
  --api-base http://localhost:8000/v1 --model my-local-model \
  --output-dir ./signal-local-seed
```

The endpoint/model are required even for the default seed-only run; their live
availability is not checked until generation is requested. `--evolve` defaults
to one generation attempt. Explicit positive `--iterations` requires `--evolve`.
These are OE generation attempts, not hard evaluation/request/time/cost caps;
seed evaluation, retries and the original reporting code's own evaluations are
additional. An upstream entrypoint can score its seed before OE scores it again.
Remote/cloud-evaluation examples such as adaptive sorting and LLM fine tuning
remain unsupported.

Only a named `--api-key-env VARIABLE_NAME` selects a key, which must be non-empty.
Without it, an explicit local placeholder is used, including for provider-client
construction in seed-only mode. The runner does **not** fall back to
`OPENAI_API_KEY` or an example `.env` for credentials. Authenticated endpoints
need an explicitly selected key before using `--evolve`. Google connection
arguments in the upstream main are accepted by the existing facade and warn
that they are ignored.

The runner disables `dotenv.load_dotenv` during entrypoint import and temporarily
supplies the selected model, equal positive source budgets, synchronous
evaluation and concurrency-one configuration. Import-time environment changes
are restored afterwards. It then sets the corresponding entrypoint globals
explicitly, including zero local generation iterations for a seed-only run. This
does not equate cloud generation concurrency with OE worker concurrency: the
facade keeps its existing single-worker behavior and unsupported-field warnings.
Missing expected configuration names cause a layout-drift error rather than a
guessed adaptation.

Signal processing's reviewed entrypoint has no model configuration. For that
entrypoint only, a local experiment subclass supplies a single
`generation_settings.models` entry from the required `--model`, overriding any
upstream model list without mutating the original dictionary. The generic facade
still requires explicit models; it does not guess a default. The original main
first creates/queries a diagnostic experiment, then creates a separate experiment
to run. The runner requires precisely this `CREATED`, `COMPLETED` sequence;
an early return, caught diagnostic error or incomplete run is not success.

While signal processing's main runs, `SAVE_PROGRAM_DIR` is set to the staged
`programs/` directory and restored afterwards, including on failure. This keeps
the evaluator's optional archive writes away from an inherited destination.
The original saver can reuse filenames; these files are not a complete archive
of evaluations. No other arbitrary environment or filesystem side effects are
isolated.

`.env*` files and bytecode caches are excluded from the staged `src/` copy;
symlinks in the selected source tree or instructions are rejected. Original
source files remain untouched, and reports operate on the staged copy. A new
`--output-dir` retains copied sources, `oe-output/`, and upstream report artifacts,
including partial output after failure. Existing directories/files/symlinks are
never reused; the parent directory must already exist. Without `--output-dir`,
the temporary run and reports are removed on success or failure.

The final stdout is an OE metrics/best-score JSON summary; Python-level upstream
prints are redirected to stderr. Upstream report/plot routines still execute;
`MPLBACKEND=Agg` avoids interactive plot windows. Reports receive **only the
facade's best result**, not the complete archive, so their history charts and
reported program-list counts are not full-run evaluation statistics.
The reviewed circle-packing plot helper uses `matplotlib.cm.get_cmap`, which
[Matplotlib removed in 3.11](https://matplotlib.org/stable/api/prev_api_changes/api_changes_3.11.0.html#matplotlib-cm-get-cmap).
The optional dependency range above retains that upstream plotting API; it is
not a restriction on OE's base dependencies. This runner does not patch plotting
libraries or rewrite upstream visualization code.

Use a fresh, single-threaded main process outside an active asyncio loop.
Temporary package and SDK aliases are removed before `main()` so evaluator
callables serialize by value to workers. Already-loaded cloud SDK namespaces
are refused. Trusted source code, including import-time code and plotting
helpers, executes with normal process privileges; copying and alias cleanup
provide no sandbox or general rollback of external side effects. Dynamic
relative imports after initial loading are outside this reviewed subset.

This walkthrough covers three original-entrypoint paths toward #468. It is not a
full SDK replacement, managed-service validation or completion of the issue.

```bash
python -m unittest discover -s tests -p 'test_alphaevolve_entrypoint.py' -v
```

### Pinned upstream-example CI

The separate **AlphaEvolve upstream examples** workflow runs on relevant pull
requests and can be dispatched manually after it is on the default branch. It
uses a standard Ubuntu runner with Python 3.12, read-only repository permissions,
no Secrets, no Google SDK and no paid provider. It has a 15-minute job limit,
cancels superseded runs on the same PR/ref, and uploads no artifacts or caches.
Checkout and dependency installation need Internet access; the example subprocesses
then run with network/SDK-import tripwires and a small allowlist of environment
variables, containing only deliberate dummy credential traps.

[`tests/alphaevolve_upstream/revision.txt`](../../tests/alphaevolve_upstream/revision.txt)
is the single upstream commit pin. Changing Google's `main` does not silently
change these tests. The suite requires clean example trees at that revision,
with the checkout inside OE's parent workspace (a sibling or nested directory).
Paths are canonicalized and escapes from that workspace are rejected. The
scientific dependencies are test-only and pinned separately; the Matplotlib pin
retains the reviewed upstream plotting API. OE's base dependencies are unchanged.

There are six original-entrypoint cases: circle packing, TSP and signal
processing, each seed-only and with one fixed comment-only proposal. They use
the existing public runner, unmodified upstream evaluators and real OE workers.
The evolution cases check OE's actual trace for an evaluated child containing
the proposal marker, instead of treating a provider call or successful process
exit alone as proof. The cases also check expected finite seed metrics, exact
original/staged source hashes, alias cleanup and report artifacts. TSP must
produce all five PNG reports and its saved program; signal processing must
retain its evaluator archive under the run directory.

Python audit hooks record blocked socket/DNS attempts immediately, including in
workers that exit without running `atexit`. Unscoped `alpha_evolve` and Google
imports are blocked and recorded too; the explicit local aliases still work.
Two negative tests check that even caught network/import exceptions remain
observable and fail offline validation. These are tripwires for the trusted
examples, not a security sandbox. Fixed proposals establish compatibility, not
real-model optimization quality or a managed AlphaEvolve service run.

To reproduce the eight opt-in checks locally from an OE checkout:

```bash
git clone https://github.com/Google-Cloud-AI/alphaevolve-on-googlecloud.git ../alphaevolve-reference
git -C ../alphaevolve-reference checkout "$(cat tests/alphaevolve_upstream/revision.txt)"
python -m pip install -e . -r tests/alphaevolve_upstream/requirements.txt
ALPHAEVOLVE_UPSTREAM_DIR=../alphaevolve-reference \
  python tests/alphaevolve_upstream/test_examples.py -v
```

Run this command in a Python 3.12 environment matching the workflow. No model
server or Google credentials are needed. A missing path, wrong revision, dirty
upstream checkout or offline violation makes explicit execution fail. Temporary
run directories are cleaned up; failures include captured subprocess diagnostics.
This suite is separate from default unit discovery and the existing
`tests/integration` real-model job; neither implicitly downloads the upstream
repository or requires these optional scientific packages.

## Evaluate an upstream seed locally

From an installed OpenEvolve checkout:

```bash
git clone https://github.com/Google-Cloud-AI/alphaevolve-on-googlecloud.git ../alphaevolve-reference

# Evaluator-only dependencies; no Google client or authentication is used.
python -m pip install numpy scipy matplotlib pydantic
export PYTHONPATH="$(cd ../alphaevolve-reference/src && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg

python examples/alphaevolve_interop/run.py ../alphaevolve-reference/examples/circle_packing
python examples/alphaevolve_interop/run.py ../alphaevolve-reference/examples/tsp
python examples/alphaevolve_interop/run.py ../alphaevolve-reference/examples/signal_processing
```

Without `--config`, the runner only evaluates the seed and prints metrics and
artifacts. It makes **no LLM or Google Cloud requests**. The `alpha_evolve.models`
import used by circle packing resolves to the upstream source, not a fake SDK.
Upstream example dependencies can change; inspect their requirements if an import
fails. The signal-processing evaluator writes its normal `programs/` artifacts
under the current directory, so run it in a disposable working directory if needed.

To generate and evaluate new candidates, provide **your own OpenEvolve config**:

```bash
python examples/alphaevolve_interop/run.py ../alphaevolve-reference/examples/circle_packing \
  --config /path/to/your-openevolve-config.yaml --iterations 1 --output-dir ./interop-output
```

This second command invokes the model/provider you configured and may incur API
charges. It does not use or translate the Google example's `.env`, cloud client,
`run_evolution.py`, or Google credentials. Local model endpoints work as with any
other OpenEvolve run. Remote Cloud Run/GKE evaluators are not supported here.

## Use the adapter directly

```python
from openevolve.compat.alphaevolve import adapt_evaluator

# imported_evaluator is the original synchronous AlphaEvolve evaluator callable.
evaluate = adapt_evaluator(imported_evaluator, objective_metric="sum_of_radii")
```

Use `evaluate` with `run_evolution(..., evaluator=evaluate)`, or expose it in an
evaluator module for the CLI. The helper:

- Reads the candidate's source file as UTF-8 and constructs
  `{"content": {"files": [{"path": filename, "content": source}]}}`.
- Accepts the nested `scores.scores` / `insights.insights` envelope as dictionaries
  or SDK-style objects, without importing Google's package in OpenEvolve.
- Keeps all metrics for features and maps the explicitly selected objective to
  `combined_score`, avoiding accidental averaging of unrelated metrics.
- Stores insights as JSON in the `alphaevolve_insights` artifact.
- Marks exact `-1e12` sentinel, null, and non-finite scores as failures, with
  `error: 0.0`, an error artifact, and a `combined_score` of `-1e12`. This avoids
  giving a failed candidate zero fitness when valid objectives are negative.

The sentinel is an example convention, not a universal SDK rule. Set
`failure_sentinel=None` to disable it or choose a different sentinel. Choose a
finite `failure_score` **below every valid score** for your problem. A failed
secondary metric also marks the whole evaluation failed. Duplicate metrics,
missing objectives and malformed scores raise actionable errors. The names
`error` and `combined_score` are reserved by this mapping; select
`objective_metric="combined_score"` if that is already your original metric.

Exceptions propagate to OpenEvolve's existing retry/error handling. That generic
handling may return zero fitness on exceptions or timeouts; this adapter does not
change that behavior. For negative objectives, inspect error/timeout results
before interpreting the best program.

The adapter provides **no sandbox**. Like the original local evaluators, it runs
trusted Python code with the current process's privileges. Use a separate
execution environment for untrusted candidates.

## Use an OpenEvolve evaluator with AlphaEvolve candidates

The reverse adapter accepts AlphaEvolve's `content.files` candidate format and
calls an existing synchronous `evaluate(program_path)` function:

```python
from my_evaluator import evaluate as evaluate_openevolve
from openevolve.compat.alphaevolve_reverse import adapt_openevolve_evaluator

evaluate_candidate = adapt_openevolve_evaluator(
    evaluate_openevolve,
    objective_metric="combined_score",
    failure_score=-1e12,
)
result = evaluate_candidate(
    {"content": {"files": [{"path": "src/main.py", "content": "# candidate source\n"}]}}
)
# {"scores": {"scores": [{"metric": "combined_score", "score": ...}, ...]}}
```

Pass `evaluate_candidate` as `evaluator_function` to the upstream SDK's
`AlphaEvolveExperiment`. The nested dictionary result is accepted by its evaluator
wrapper and by `AlphaEvolveProgramEvaluation.model_validate(result)`. The adapter
itself does not import the Google SDK, create a client, authenticate, or make
requests. Managed-service setup and execution remain the caller's responsibility.

Only one Python source file is supported, supplied as a dictionary or SDK-style
object. Its source is written unchanged as UTF-8 to a private temporary
`program.py`; candidate path metadata is ignored, and files are removed after
evaluation, including on exceptions. Evaluators requiring sibling files, a
particular original filename, asynchronous execution or files that outlive the
call are not supported. The adapter does not change the working directory,
provide a sandbox, or enforce a timeout; the trusted evaluator retains its
existing execution privileges and resource controls.

Both a metrics dictionary and `EvaluationResult` are supported. Numeric metrics
keep their names and values. Boolean diagnostic flags and reserved `error` /
`timeout` fields are not exported as optimization metrics. `objective_metric`
must be present on success; it is not a request to rename metrics or configure
the service's optimization objective.

Evaluator exceptions, explicit `error` in metrics/artifacts, truthy `timeout`,
and non-finite scores map **every available numeric metric and the objective**
to the finite `failure_score`. Generic error feedback is included without
copying exception text or artifact contents. Choose a failure value below all
valid scores for the problem, especially for negative objectives. Malformed
candidates, multi-file input, missing successful objectives and malformed return
contracts raise errors. Interrupts propagate after cleanup.
Zero or negative scores without an explicit failure marker remain valid scores;
the adapter cannot infer failures that an evaluator silently converts to numbers.

Artifacts are omitted with a warning, apart from recognizing the `error` marker;
text, images, and `alphaevolve_insights` artifacts are not round-tripped. This is
only the reverse evaluator-contract slice of #468, not a cloud export bundle,
SDK shim, or an assertion that a managed-service experiment has been run.

## Export an OpenEvolve experiment configuration

`config_to_alphaevolve` returns a plain experiment dictionary for the upstream
SDK, without creating files, loading templates, importing Google dependencies,
looking up credentials, or making requests:

```python
from openevolve.compat.alphaevolve_export import config_to_alphaevolve
from openevolve.config import Config

config = Config.from_dict({
    "language": "python",
    "max_iterations": 20,
    "llm": {"models": [{"name": "my-local-model", "weight": 1.0}]},
    "prompt": {"system_message": "Pack circles into a unit square."},
})
experiment = config_to_alphaevolve(
    config,
    title="Circle Packing",
    # Pass literal problem text, not the name of a prompt template.
    problem_description=config.prompt.system_message,
    model_mapping={"my-local-model": "gemini-2.5-flash"},
    generation_concurrency=2,
)
# UserWarning: OpenEvolve settings not exported (including defaults): ...
# UserWarning: Model names explicitly remapped: llm.models[0].name
```

The result can be checked offline with the upstream SDK's
`AlphaEvolveExperimentConfig.model_validate(experiment)`. Supplying it to a real
SDK experiment's `create_experiment` is a separate, authenticated cloud operation
that may incur charges; this helper does not perform that operation.

| OpenEvolve / explicit input | Exported setting |
| --- | --- |
| `config.max_iterations` | `run_settings.max_programs` |
| `config.language="python"` | `program_language="python"` |
| Resolved `llm.models` names and relative weights | `generation_settings.models` |
| Required literal `title` and `problem_description` arguments | Top-level text fields |
| Optional `generation_concurrency` argument | `run_settings.concurrency` |

Only single-file Python interoperability is covered. Non-Gemini models require
an explicit name mapping; no fallback model is selected. Duplicate target names
are merged by summing weights, with at most two distinct targets. Weights must be
finite and positive. Names are checked for Gemini form, not against a live model
catalog. Service availability, project permissions and model-tier mixture limits
(such as caps on some Pro models) remain the caller/service's responsibility.

The generation budget counts attempts, not successful evaluations, API requests,
tokens, elapsed time, or money; seed evaluation is separate. Evaluation worker
counts are **not** copied into generation concurrency. Configure the upstream
experiment's local evaluation workers and `max_programs_evaluated` separately.
Without `generation_concurrency`, the service's concurrency setting is omitted.

Export always warns by field path about omitted settings, including defaults:
MAP-Elites/islands/migration, local evaluator settings and timeouts, LLM feedback,
prompt construction, per-island prompts, local endpoints/keys, sampling/retry
controls, hooks, checkpointing, tracing and early stopping. Warning text contains
field names, not values or arbitrary dictionary keys. Prompt templates are not
resolved: explicitly supply the literal problem text you intend to use.

The input config and mapping are not modified. Export followed by
`config_from_alphaevolve` preserves the supported budget/language/model subset;
the importer prefixes literal problem text and requires a new local endpoint.
This does not preserve search behavior or provide a runnable cloud export bundle.

## Export a minimal Python bundle

From an installed OpenEvolve checkout, export an existing single-file Python
seed and synchronous `evaluate(program_path)` evaluator:

```bash
printf '%s\n' 'Improve the search for the global minimum of the supplied function.' > problem.md

python -m openevolve.compat.alphaevolve_bundle \
  examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py \
  --config examples/function_minimization/config.yaml \
  --output-dir ./alphaevolve-bundle \
  --title "Function minimization" \
  --instructions problem.md \
  --objective-metric combined_score \
  --requirements-file examples/function_minimization/requirements.txt

# Review/install the evaluator's dependencies in your environment first.
python ./alphaevolve-bundle/check.py

# Offline preflight; does not import the SDK/evaluator or execute either source.
python ./alphaevolve-bundle/run_alphaevolve.py
```

The destination must not exist, and its parent must already exist. Exporting
does not execute either source file, install packages, import Google's SDK,
look up credentials, or contact a service. The CLI infers Python when the YAML
omits its language. `--instructions` supplies literal problem text; it does not
resolve prompt templates. Non-Gemini model names require explicit repeated
`--model-map SOURCE=GEMINI` options. The configuration mapping and warnings are
the same as `config_to_alphaevolve` above.

The output contains:

```text
alphaevolve-bundle/
  experiment.json            # Limited experiment configuration
  evaluation.json            # Explicit objective metric and failure score
  src/
    __init__.py
    program.py               # Original seed bytes
    openevolve_evaluator.py   # Original evaluator bytes
    evaluate.py              # Reverse adapter entry point
  check.py
  run_alphaevolve.py          # Offline preflight by default; explicit --run for cloud use
  README.md
  requirements.txt           # Only when explicitly supplied
```

Use `from src.evaluate import evaluate` with the bundle root on Python's module
path and pass that callable as `evaluator_function` to the upstream SDK. Validate
`experiment.json` with `AlphaEvolveExperimentConfig.model_validate` as described
above. Read the generated README for installation and integration instructions.
The same OpenEvolve checkout containing the reverse adapter must be installed in
the target environment; no dependency or SDK is vendored into the bundle.

Sources and the optional dependency file are copied byte for byte, without
secret removal or automatic discovery of local modules, included requirements
or data files. The exporter requires UTF-8 Python, a module-level synchronous
`evaluate` accepting one path, and rejects relative imports and discoverable
sibling module imports. Dynamic imports and working-directory dependencies
cannot all be detected. Use self-contained sources and review them before
sharing or running the bundle.

`check.py` runs the copied seed through the exported evaluator, prints a JSON
evaluation to stdout and redirects Python-level evaluator/import stdout to
stderr. Evaluators must capture their own subprocess output to keep stdout
clean. The check also works by absolute path from another directory.
Import/contract errors and adapter-reported failures produce a nonzero exit
status. The check executes trusted code with the current process's privileges;
it adds no sandbox or timeout and cannot recognize failures silently converted
into numeric scores.

### Preflight and opt-in cloud launch

`run_alphaevolve.py` defaults to a basic offline JSON summary of the bundle,
models and generation budget. It checks the exported configuration subset,
evaluation options and Python sources without importing the SDK or evaluator,
executing a seed, looking up credentials or making requests. `--dry-run` is an
explicit alias for this default. Dependency installation, full SDK schemas,
permissions and live model availability are not checked in this mode.

For cloud use, install the official SDK from a reviewed checkout of
[Google's repository](https://github.com/Google-Cloud-AI/alphaevolve-on-googlecloud)
and follow its README for ADC, project, engine and service access setup. Review
the generation budget and select currently available models explicitly during
export; the Gemini-name check is not a live model catalog. Then opt in:

```bash
python ./alphaevolve-bundle/run_alphaevolve.py --run \
  --project-id YOUR_PROJECT_ID --engine YOUR_ENGINE_ID \
  --max-programs-evaluated 1
```

The project, engine and positive evaluation target are required for `--run`.
`--location` accepts `global` (default), `us` or `eu`. No project/engine values
are inferred from `.env` or environment variables; Google credentials follow the
SDK's ADC behavior only after cloud use is requested. There is no endpoint override.

The launcher validates the configuration with the real SDK and evaluates the
seed locally before constructing a client. It creates a new experiment, uploads
the seed with its actual scores, checks seed creation and start responses, then
invokes the SDK controller with one sampler and one synchronous evaluator. This
avoids requiring thread-safe or signal-free evaluators. The generated entry point
resolves the bundle relative to itself and works from another directory. Existing
bundles can also use `python -m openevolve.compat.alphaevolve_launch /path/to/bundle`.

The final run JSON includes the experiment name, SDK statistics and whether the
evaluation target was reached; failures or unmet targets exit nonzero. The target
cannot exceed the exported generation-attempt budget, but it is not a strict
evaluation cap: SDK polling/queued work can overshoot it, and seed evaluation is
additional. The positive `--idle-timeout-seconds` (default 120) is an idle threshold,
not a wall-clock timeout or a cost limit. Running trusted sources adds no sandbox.

Cloud operations may incur charges. Every `--run` creates a new experiment;
reruns do not resume existing work. Failure, interruption or local controller exit
does not automatically stop or delete remote resources. Inspect the reported
experiment in your project. Export and preflight never start cloud work, and
offline validation is not evidence of a managed-service run. This remains a
limited #468 export slice and does not reproduce OpenEvolve's evaluator pipeline,
search behavior or the broader SDK shim.

Run the offline bundle tests with:

```bash
python -m unittest discover -s tests -p 'test_alphaevolve_bundle.py' -v
python -m unittest discover -s tests -p 'test_alphaevolve_launch.py' -v
```

### Verify a complete local round trip

The dedicated interop workflow also exercises two existing OpenEvolve examples
through this complete path, without Google's SDK or a model service:

```text
OE Python seed + evaluator + YAML
  -> export_to_alphaevolve
  -> relocate bundle and remove the original input directory
  -> config_from_alphaevolve(experiment.json)
  -> generated candidate evaluator -> adapt_evaluator
  -> real OpenEvolve seed evaluation + one evaluated proposal
```

From an installed source checkout, reproduce just these checks with:

```bash
python tests/alphaevolve_upstream/test_roundtrip.py -v
```

Unlike the original-entrypoint tests, this command needs no AlphaEvolve source
checkout. It uses the unchanged `function_minimization` and `circle_packing`
sources from this repository. The helper runs from an unrelated temporary
working directory, checks source hashes and configuration mappings, and removes
temporary import registrations before transporting evaluators to OE workers.
Both the seed and a fixed comment-only proposal are evaluated on each side.
The tests compare all numeric seed/child metrics except elapsed evaluation time,
check positive finite fitness, and require worker evaluation records and actual
evolution traces. A provider call or successful process exit alone is insufficient.
The existing test-only startup tripwires block and record network and cloud SDK
attempts in the helper, workers and evaluator subprocesses. These are safeguards
for trusted examples, not a security sandbox.

This is execution equivalence of the **supported subset under controlled inputs**,
not a lossless search/configuration round trip. The test bounds the budget to one,
fixes NumPy's random stream per evaluation, and explicitly aligns one island,
single-stage single-worker evaluation and diff mode. The circle example needs
an explicit model-name mapping; model weights and the mapped names are checked
before replacing generation with the fixed offline provider. The imported prompt
has the documented `Problem description:` prefix. Export warnings for omitted
settings are checked; original artifacts and timing are not expected to survive.
There is no claim of real-model improvement or of preservation of the examples'
original multi-island/cascade behavior.

For #468, the three original local entrypoints and this OE-to-bundle-to-OE path
now have automated offline coverage. **An authenticated managed AlphaEvolve run
remains a separate acceptance check.** Neither these tests nor GitHub Actions
exercise Google's service, verify project access/live model availability, or
establish cloud cost limits. A maintainer with service access can review a bundle,
run its local `check.py` and preflight, then explicitly test `--run` as documented
above before deciding whether to close #468 or track remaining scope separately.

## Import a local Python experiment configuration

`config_from_alphaevolve` imports an explicit experiment dictionary without
reading `.env`, contacting Google, or creating model clients:

```python
from openevolve.compat.alphaevolve import config_from_alphaevolve

experiment = {
    "title": "Circle Packing",
    "program_language": "python",
    "problem_description": "Pack circles into a unit square.",
    "run_settings": {"max_programs": 10},
    "generation_settings": {
        "models": [{"name": "my-local-model", "weight": 1}],
    },
}
config = config_from_alphaevolve(
    experiment,
    api_base="http://localhost:8000/v1",
    api_key="local-placeholder",  # Replace if the chosen endpoint requires authentication.
)
# Pass config to run_evolution(program_path, adapted_evaluator, config=config).
# Running evolution invokes your chosen endpoint; conversion alone does not.
```

The imported names must be served by the explicitly supplied OpenAI-compatible
endpoint. Gemini model names are not remapped to local models. Endpoint/key
values embedded in the cloud dictionary are never copied. A missing `api_key`
retains OpenEvolve's usual provider behavior, which may use `OPENAI_API_KEY` when
the client is subsequently constructed. Conversion adds no Google dependency.
For a Google SDK config object, pass its `model_dump()` dictionary.

| AlphaEvolve field | OpenEvolve mapping |
| --- | --- |
| `generation_settings.models[].name/weight` | `llm.models`, with relative weights unchanged; null/missing weights become 1 |
| `run_settings.max_programs` | `max_iterations`, requiring an explicit positive integer |
| `problem_description` + optional `generation_settings.context` | Literal `prompt.system_message` text |
| `program_language="python"` | `language="python"`, `.py` suffix |

The converter disables cascade evaluation for the single-stage local evaluator;
other OE settings retain their defaults. It does not mutate the input or start
a run. Missing models, missing budgets and unsupported languages raise instead
of guessing defaults. `title` is metadata and is not imported.

`max_iterations` counts generation attempts, **not** successful evaluations,
requests, tokens or money. OE also evaluates the initial seed, and retries can
make additional calls. `run_settings.max_duration` is rejected because this
helper cannot enforce that wall-clock limit. Non-null unsupported settings emit
warnings naming the fields, including cloud `concurrency`, prompt-inclusion
flags and evolution settings. Their semantics are not silently presented as
equivalent OE controls; OE's worker count stays at its default of one unless
you configure it separately. Model-specific extra fields also warn and are not
forwarded. Warning messages include field paths, not their values.

The dictionary helper provides a subset of inbound configuration support (A3
in #468). It does not read `.env` files, emulate the experiment/controller SDK,
or export cloud-compatible bundles. Use the returned `Config` directly with
the Python API; explicit `.env` import is described below.

## Import an explicit example .env file

Install the optional parser (the base OpenEvolve install is unchanged):

```bash
python -m pip install "openevolve[alphaevolve-env]"
# From a source checkout: python -m pip install -e ".[alphaevolve-env]"
```

```python
from pathlib import Path
from openevolve.compat.alphaevolve_env import config_from_alphaevolve_env

config = config_from_alphaevolve_env(
    "../alphaevolve-reference/examples/tsp/example.env",
    problem_description=Path(
        "../alphaevolve-reference/examples/tsp/instructions.md"
    ).read_text(encoding="utf-8"),
    api_base="http://localhost:8000/v1",
    api_key="local-placeholder",
)
# Pass config to run_evolution with the original seed and adapted evaluator.
# Importing alone does not construct clients or call the endpoint.
```

Only the named file is read. The helper does not find a default `.env`, read
other environment variables to fill missing settings, interpolate variable
references, change `os.environ`, or execute shell commands or the original
`run_evolution.py`. The problem text and OE endpoint/key remain explicit caller
arguments. A missing `api_key` still allows OE's usual provider behavior when
a model client is later constructed, as described above.

| .env setting | Mapping / behavior |
| --- | --- |
| `MODEL`, optional `MODEL_WEIGHT` | Single model; missing weight becomes 1 |
| `MODEL_1..N`, optional `MODEL_i_WEIGHT` | Weighted model list in numeric index order; gaps allowed |
| `MAX_PROGRAMS_GENERATED` | Required positive integer mapped to `max_iterations` |
| `MAX_PROGRAMS_EVALUATED` | If present, must equal the generation budget; always warns that it is not a hard evaluation cap |
| Other non-null variables, including cloud credentials/endpoints and concurrency | Not imported; warnings contain names, not values |

Either model form may be used, but not both. Empty names, orphan weights,
non-positive/non-finite weights, duplicate assignments, malformed syntax and
variable/command references in model names raise errors. Quoting, comments,
`export` prefixes and multiline quoted values are parsed with `python-dotenv`;
expansion is intentionally disabled. No model names are guessed or remapped.
Some upstream examples, including signal processing, do not declare models in
their `example.env`: add an explicit model or use the dictionary helper instead.

An evaluation-only budget, or different generation and evaluation budgets,
is rejected rather than translated into an allegedly equivalent limit. Even
equal budgets are not equivalent: OE counts generation attempts, additionally
evaluates the seed, and may retry. This helper does not impose a strict
evaluation, request, token, time or cost limit. The existing dictionary helper
performs the final configuration validation; existing config loading is
unchanged. The example runner can use either helper as described below.

## Run with imported configuration from the command line

The runner accepts one of `--config` (OE YAML), `--alphaevolve-config`
(experiment JSON) or `--alphaevolve-env` (an explicitly named `.env`). The two
AlphaEvolve options **only validate the configuration and evaluate the seed by
default**. Add `--evolve` to generate candidates using the selected endpoint.
The legacy `--config` option still starts evolution, with its existing default
of one iteration.

For `.env`, install the optional parser above and prepare a copy such as
`./local.env`. Set `MODEL` or the numbered model names to models actually served
by your OE endpoint; importing a Gemini name does not make it available locally.
Provide a problem-text file explicitly rather than running the upstream
entrypoint to extract it:

```bash
# Import/validate and evaluate the original TSP seed; no model requests.
python examples/alphaevolve_interop/run.py ../alphaevolve-reference/examples/tsp \
  --alphaevolve-env ./local.env \
  --instructions ../alphaevolve-reference/examples/tsp/instructions.md \
  --api-base http://localhost:8000/v1

# Explicit model-driven evolution; may incur charges at the selected endpoint.
python examples/alphaevolve_interop/run.py ../alphaevolve-reference/examples/tsp \
  --alphaevolve-env ./local.env \
  --instructions ../alphaevolve-reference/examples/tsp/instructions.md \
  --api-base http://localhost:8000/v1 \
  --evolve --iterations 1 --output-dir ./interop-output
```

Alternatively, save an experiment dictionary (including `program_language`,
`problem_description`, `run_settings` and `generation_settings`) as JSON:

```bash
python examples/alphaevolve_interop/run.py ../alphaevolve-reference/examples/circle_packing \
  --alphaevolve-config ./experiment.json --api-base http://localhost:8000/v1 \
  --evolve --iterations 1 --output-dir ./interop-output
```

Both imports require an explicit `--api-base`. To select a key without putting
its value in command arguments, add `--api-key-env OE_API_KEY` after setting
that environment variable yourself. An explicitly selected missing/empty key
variable is an error. If this option is omitted, the existing OE provider's
key behavior (including `OPENAI_API_KEY` fallback) applies when a client is
constructed. Nothing is copied from Google credential fields in the source.

With `--evolve`, omitting `--iterations` uses the imported generation budget.
An explicit positive `--iterations` may only reduce that budget. As above,
generation attempts are not a strict request/evaluation/cost cap. `--iterations`
without an evolution mode is rejected rather than ignored. `--instructions`
is required only for `.env`; JSON already contains the problem description.
Invalid option combinations and import errors are reported before loading
the example evaluator. The original cloud `run_evolution.py` is not imported.

## Validation

Default unit tests use synthetic envelopes and OpenEvolve's real evaluator and
callable-serialization path, without Google's SDK, keys, or network access:

```bash
python -m unittest discover -s tests -p test_alphaevolve_compat.py
python -m unittest discover -s tests -p test_alphaevolve_config.py
python -m unittest discover -s tests -p test_alphaevolve_env.py
python -m unittest discover -s tests -p test_alphaevolve_runner.py
python -m unittest discover -s tests -p test_alphaevolve_reverse.py
python -m unittest discover -s tests -p test_alphaevolve_export.py
```

The walkthrough additionally permits direct validation against an upstream
checkout. An evaluator-only run demonstrates contract compatibility, not learned
improvement or a real-model evolution experiment.
