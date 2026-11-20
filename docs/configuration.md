# Configuration

Every option, with its default, is in [`configs/default_config.yaml`](../configs/default_config.yaml). This page covers the ones people most often need.

```yaml
# Advanced Configuration Example
max_iterations: 1000
random_seed: 42  # Full reproducibility

# Optional prefix for all logger names, e.g. "run1" logs as "run1.openevolve.*";
# useful to tell apart log output from multiple concurrent runs
logger_prefix: "run1"

llm:
  # Ensemble configuration
  models:
    - name: "gemini-2.5-pro"
      weight: 0.6
    - name: "gemini-2.5-flash"
      weight: 0.4
  temperature: 0.7

database:
  # MAP-Elites quality-diversity
  population_size: 500
  num_islands: 5  # Parallel evolution
  # island_initial_programs: [null, "seeds/alt.py"]  # Optional per-island starting programs
  migration_interval: 20
  feature_dimensions: ["complexity", "diversity", "performance"]

  # Optional novelty filtering with Gemini embeddings
  embedding_model: "gemini-embedding-001"
  similarity_threshold: 0.99
  # Or any OpenAI-compatible embedding endpoint (OpenRouter, local servers):
  # embedding_api_base: "https://openrouter.ai/api/v1"  # or OPENAI_EMBEDDING_BASE_URL

evaluator:
  enable_artifacts: true      # Error feedback to LLM
  cascade_evaluation: true    # Multi-stage testing
  use_llm_feedback: true      # AI code quality assessment
  evaluator_args:             # Keyword arguments forwarded to evaluate()
    weight: 2.0

prompt:
  # Sophisticated inspiration system
  num_top_programs: 3         # Best performers
  num_diverse_programs: 2     # Creative exploration
  include_artifacts: true     # Execution feedback
  
  # Custom templates
  template_dir: "custom_prompts/"
  use_template_stochasticity: true  # Randomized prompts
```

For Gemini embeddings, set `GEMINI_API_KEY`. `GOOGLE_API_KEY` is also supported
as a fallback. OpenEvolve uses Google's OpenAI-compatible endpoint automatically.

## Command-Line Options

```bash
openevolve-run initial_program.py evaluator.py \
  --config config.yaml \
  --iterations 100 \
  --target-score 0.95 \
  --checkpoint path/to/checkpoint_directory \
  --logger-prefix run1
```

`--target-score` stops the run once `combined_score` reaches the value. `--checkpoint` resumes from a saved run. `--logger-prefix` keeps log output distinguishable across parallel runs (it logs as `run1.openevolve.*`).

## Checkpoint storage

A checkpoint publishes `metadata.json` atomically after writing the program and
artifact objects it references. Loading restores that exact population; removed
programs are not recovered from leftover files. If a referenced object is missing
or corrupt, loading fails without replacing an already running database.

New checkpoints use `checkpoint_version: 1` and a `program_files` mapping into
`programs/.objects/`. Large artifacts are copied into `.checkpoint_artifacts/`,
so moving the whole checkpoint directory preserves them. Prompt logs and island
feature ranges are restored as well. The visualizer and static export read the
same manifest. Active working artifacts are protected from retention cleanup.

Legacy `programs/<id>.json` checkpoints remain readable. Older OpenEvolve releases
cannot read the new format; use this version to resume new checkpoints. External
checkpoint readers should use `openevolve.checkpoint.load_checkpoint()` rather
than globbing program files. Immutable objects are retained so a reader can finish
using an older manifest while another save publishes a replacement. Remove an
entire checkpoint directory when it is no longer needed; do not prune its objects
while readers may still be using it. These guarantees cover process interruption,
not new hardware power-loss durability guarantees.

## Evolution Modes

```yaml
diff_based_evolution: true      # SEARCH/REPLACE diffs; false asks for a full rewrite every iteration
full_rewrite_probability: 0.2    # With diffs on, chance an iteration asks for a full rewrite instead
enforce_evolve_blocks: true      # Revert edits outside the EVOLVE-BLOCK-START/END markers

evolution_trace:
  enabled: true
  include_failures: true         # Also log failed iterations to <trace name>_failures.jsonl
```

## Feature Engineering

**Control how programs are organized in the quality-diversity grid:**

```yaml
database:
  feature_dimensions: 
    - "complexity"      # Built-in: code length
    - "diversity"       # Built-in: structural diversity
    - "performance"     # Custom: from your evaluator
    - "memory_usage"    # Custom: from your evaluator
    
  feature_bins:
    complexity: 10      # 10 complexity levels
    performance: 20     # 20 performance buckets
    memory_usage: 15    # 15 memory usage categories
```

**Important**: Return raw values from evaluator, OpenEvolve handles binning automatically.

## Passing Arguments to Your Evaluator

If your `evaluate()` function takes extra keyword arguments beyond the program
path, pass them with the `--evaluator-args` CLI flag (a JSON object):

```bash
openevolve-run initial_program.py evaluator.py \
  --evaluator-args '{"weight": 2.0, "label": "experiment-1"}'
```

or configure them in YAML:

```yaml
evaluator:
  evaluator_args:
    weight: 2.0
    label: "experiment-1"
```

OpenEvolve inspects the signature of your `evaluate()` function: arguments are
forwarded only when the function accepts them (named parameters or `**kwargs`),
and a startup error lists the offending keys and the actual signature if it
does not — so typos fail fast instead of silently doing nothing. Evaluators
with a single `program_path` argument are unaffected.

This replaces the earlier workaround of smuggling per-run settings to the
evaluator through environment variables. (Note: custom top-level keys in
`config.yaml` are dropped silently by the config parser — use
`evaluator.evaluator_args` or the CLI flag instead.)

## Evaluation and Prompt Feedback

- `evaluator.cascade_evaluation` selects the evaluation path. With a cascade,
  `evaluate_stage1(path)` can screen out candidates before more expensive stages.
  Each threshold gates the next stage; the check uses a numeric
  `combined_score` when available and otherwise averages the numeric metrics,
  excluding the `error` metric and boolean flags. If the cascade is off
  or there is no stage 1, OpenEvolve calls `evaluate(path)` directly.
- `evaluator.use_llm_feedback` adds a separate LLM judging pass after evaluation
  when judge models are configured. Numeric judge results add `llm_*` metrics.
  If there is already a `combined_score` and the judge returns numeric scores,
  the final score is 70% evaluator score and 30% mean judge score.
  `llm_feedback_weight` scales the reported judge metrics, not that blend.
- `prompt.use_template_stochasticity` samples wording from
  `prompt.template_variations` for matching placeholders in the user template.
  It changes the prompt text, not how a candidate is scored. Use fixed wording
  in place of those custom placeholders when turning variations off.

For diagnostics from your own evaluator, return an [artifact](#artifacts)
instead of enabling the judge. For example, save the following as `evaluator.py`:

```python
import ast
from pathlib import Path

from openevolve.evaluation_result import EvaluationResult


def evaluate_stage1(program_path):
    try:
        ast.parse(Path(program_path).read_text())
    except SyntaxError:
        return EvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={"stderr": "Fix the Python syntax before running stage 2."},
        )
    return EvaluationResult(metrics={"combined_score": 1.0})


def evaluate_stage2(program_path):
    # Replace this marker with the expensive evaluation in a real run.
    return {"stage2_reached": 1.0}


def evaluate(program_path):
    result = evaluate_stage1(program_path)
    if result.metrics["combined_score"] >= 0.5:
        result.metrics.update(evaluate_stage2(program_path))
    return result
```

With OpenEvolve installed, run this from the same directory to check the
cascade and render the diagnostic locally. It makes no model calls:

```python
import asyncio

from openevolve.config import EvaluatorConfig, PromptConfig
from openevolve.evaluator import Evaluator
from openevolve.prompt.sampler import PromptSampler

evaluator = Evaluator(
    EvaluatorConfig(
        cascade_evaluation=True,
        cascade_thresholds=[0.5],
        use_llm_feedback=False,
    ),
    "evaluator.py",
)
metrics = asyncio.run(evaluator.evaluate_program("def broken(", program_id="example"))
assert metrics == {"combined_score": 0.0}

sampler = PromptSampler(PromptConfig(include_artifacts=True, use_template_stochasticity=False))
prompt = sampler.build_prompt(
    current_program="def broken(",
    program_metrics=metrics,
    program_artifacts=evaluator.get_pending_artifacts("example"),
)
assert "Fix the Python syntax before running stage 2." in prompt["user"]
assert asyncio.run(evaluator.evaluate_program("pass"))["stage2_reached"] == 1.0
```

## Custom Prompt Templates

**Advanced prompt engineering** with custom templates:

```yaml
prompt:
  template_dir: "custom_templates/"
  use_template_stochasticity: true
  template_variations:
    greeting:
      - "Let's enhance this code:"
      - "Time to optimize:"
      - "Improving the algorithm:"
    improvement_suggestion:
      - "Here's how we could improve this code:"
      - "I suggest the following improvements:"
      - "We can enhance this code by:"
```

**How it works:** Place `{greeting}` or `{improvement_suggestion}` placeholders in your templates, and OpenEvolve will randomly choose from the variations for each generation, adding diversity to prompts.

See [prompt examples](../examples/llm_prompt_optimization/templates/) for complete template customization.

## Island Scheduling

Pass an optional selector to `run_evolution`:

```python
def select_island(context):
    # Schedule the next iteration on the island with the fewest pending tasks.
    return min(range(len(context.islands)), key=lambda i: context.pending_counts[i])

result = run_evolution(
    initial_program="program.py",
    evaluator="evaluator.py",
    island_selector=select_island,
)
```

The selector receives a read-only `IslandSelectionContext` with the iteration number,
pending task counts, and each island's population size, best and average score, diversity, and
generation. It must return an island ID from `0` to `num_islands - 1`. It runs in the
main process before each iteration is submitted. Without a selector, OpenEvolve uses
its existing balanced island scheduling.

## Population Strategies

Population management can use decision hooks. For example, this policy lets a
candidate replace an island's MAP-Elites cell occupant when their scores tie:

```python
from openevolve import run_evolution
from openevolve.population import PopulationStrategy

strategy = PopulationStrategy(
    replace_cell=lambda state, candidate, incumbent, island: (
        candidate.metrics["combined_score"] >= incumbent.metrics["combined_score"]
    ),
)
result = run_evolution("program.py", "evaluator.py", population_strategy=strategy)
```

Other optional hooks are `admit(state, candidate, island) -> bool`,
`archive(state, candidate) -> ArchiveDecision`,
`evict(state, count, protected_ids) -> sequence of program IDs`,
`migration_due(state) -> bool`, and `migrate(state) -> sequence of MigrationMove`.
`state` is a detached `PopulationSnapshot` of programs, island memberships, cell owners,
the archive, population limits, and migration generations. Unspecified hooks retain
the existing rules. `ProgramDatabase` validates decisions and applies all changes;
the strategy never receives the mutable database. The initial seed cannot be rejected.

## Artifacts

**Artifacts side-channel** provides rich feedback to accelerate evolution:

```python
# Evaluator can return execution context
from openevolve.evaluation_result import EvaluationResult

return EvaluationResult(
    metrics={"performance": 0.85, "correctness": 1.0},
    artifacts={
        "stderr": "Warning: suboptimal memory access pattern",
        "profiling_data": {...},
        "llm_feedback": "Code is correct but could use better variable names",
        "build_warnings": ["unused variable x"]
    }
)
```

When this program is used as the parent, the default prompt templates include
its saved artifacts if `prompt.include_artifacts` is enabled:

```markdown
## Previous Execution Feedback
Warning: suboptimal memory access pattern
LLM Feedback: Code is correct but could use better variable names
Build Warnings: unused variable x
```

This creates a **feedback loop** where each generation learns from previous mistakes!

## Visualization

**Real-time evolution tracking** with interactive web interface:

```bash
# Install visualization dependencies
pip install -r scripts/requirements.txt

# Launch interactive visualizer
python scripts/visualizer.py

# Or visualize specific checkpoint
python scripts/visualizer.py --path examples/function_minimization/openevolve_output/checkpoints/checkpoint_100/
```

**Features:**

- **Evolution tree** with parent-child relationships
- **Performance tracking** across generations
- **Code diff viewer** showing mutations
- **MAP-Elites grid** visualization
- **Multi-metric analysis** with custom dimensions

![OpenEvolve Visualizer](../openevolve-visualizer.png)
