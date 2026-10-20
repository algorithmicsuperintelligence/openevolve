# AlphaEvolve local evaluator interoperability

This is the evaluator-contract slice of [#468](https://github.com/algorithmicsuperintelligence/openevolve/issues/468).
It runs the **unchanged** `program.py` and `evaluate.py` from Google's
[AlphaEvolve examples](https://github.com/Google-Cloud-AI/alphaevolve-on-googlecloud)
through an explicit adapter. It does not replace the cloud SDK, translate cloud
configuration, export bundles, or implement the rest of #468.

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

## Validation

Default unit tests use synthetic envelopes and OpenEvolve's real evaluator and
callable-serialization path, without Google's SDK, keys, or network access:

```bash
python -m unittest discover -s tests -p test_alphaevolve_compat.py
```

The walkthrough additionally permits direct validation against an upstream
checkout. An evaluator-only run demonstrates contract compatibility, not learned
improvement or a real-model evolution experiment.
