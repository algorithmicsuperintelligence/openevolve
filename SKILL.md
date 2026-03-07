---
name: openevolve
description: >
  Use OpenEvolve to evolve and optimize code using LLMs. Triggers: user wants to
  automatically improve an algorithm, optimize a function, discover better solutions
  through evolutionary search, or run an AlphaEvolve-style coding agent. Covers the
  full workflow: scaffolding experiments, writing evaluators, configuring evolution,
  running jobs, inspecting checkpoints, and retrieving the best evolved programs.
---

# OpenEvolve Agent Skill

OpenEvolve is an evolutionary coding agent that uses LLMs to iteratively improve code.
Given an initial program and an evaluation function, it maintains diverse populations
of program variants (MAP-Elites) and evolves them over generations to discover better
algorithms, optimizations, or solutions.

## When to use this skill

- User asks to "evolve", "optimize", or "improve" code automatically
- User wants to discover novel algorithms for a problem
- User mentions AlphaEvolve, evolutionary coding, or MAP-Elites
- User wants to benchmark multiple approaches to a problem and find the best one
- User asks to optimize GPU kernels, sorting algorithms, mathematical functions, etc.

## Prerequisites

```bash
pip install openevolve
export OPENAI_API_KEY="your-api-key"   # works for OpenAI, Gemini, or any compatible provider
```

Verify: `openevolve-run --help`

---

## Core Concepts

Every OpenEvolve experiment requires exactly **3 files**:

| File | Purpose |
|------|---------|
| `initial_program.py` | Starting code with `EVOLVE-BLOCK` markers around the section to evolve |
| `evaluator.py` | Defines `evaluate(program_path) -> dict` that scores each variant |
| `config.yaml` | LLM provider, iterations, population size, system message |

---

## Workflow

### Step 1: Scaffold the experiment

Create a project directory with the three required files.

```bash
mkdir -p my_experiment
```

#### initial_program.py

Wrap the code to evolve in `EVOLVE-BLOCK` markers. Code outside the markers stays fixed.

```python
import math

# Helper functions (not evolved)
def distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# EVOLVE-BLOCK-START
def solve(data):
    """This function will be evolved by OpenEvolve."""
    # Start with a simple/naive implementation
    result = sorted(data)
    return result
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    print(solve([3, 1, 4, 1, 5]))
```

**Rules for EVOLVE-BLOCK markers:**
- Both `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` are required as a pair
- Multiple blocks are supported for evolving different sections
- If omitted, OpenEvolve wraps the entire file (less control)
- Keep the code between markers self-contained — imports and helpers go outside

#### evaluator.py

Must define `evaluate(program_path)` returning a dict of numeric metrics.
The key `combined_score` (or the average of all numeric values) determines fitness.

```python
import importlib.util
import time

def evaluate(program_path: str) -> dict:
    """Score an evolved program. Return dict with numeric metrics."""
    # Load the evolved module
    spec = importlib.util.spec_from_file_location("evolved", program_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return {"combined_score": 0.0, "error": str(e)}

    # Define test cases
    test_cases = [
        ([3, 1, 2], [1, 2, 3]),
        ([5, 2, 8, 1], [1, 2, 5, 8]),
        ([1], [1]),
    ]

    correct = 0
    start = time.time()
    for inp, expected in test_cases:
        try:
            result = module.solve(inp.copy())
            if result == expected:
                correct += 1
        except Exception:
            pass
    elapsed = time.time() - start

    accuracy = correct / len(test_cases)
    return {
        "combined_score": accuracy,
        "accuracy": accuracy,
        "runtime": elapsed,
    }
```

**Evaluator best practices:**
- Always wrap `exec_module` in try/except — evolved code may crash
- Copy mutable inputs (lists, dicts) before passing to the evolved function
- Return `combined_score: 0.0` on any failure so evolution can continue
- Add multiple metrics (accuracy, runtime, memory) for richer MAP-Elites grids
- For custom MAP-Elites dimensions, return raw continuous values — OpenEvolve handles binning

**Advanced: using EvaluationResult for artifact feedback**

```python
from openevolve.evaluation_result import EvaluationResult

def evaluate(program_path):
    # ... run tests ...
    return EvaluationResult(
        metrics={"combined_score": 0.85, "accuracy": 0.9, "runtime": 0.3},
        artifacts={
            "stderr": captured_stderr,
            "test_failures": failure_details,
        }
    )
```

Artifacts are fed back to the LLM in the next generation's prompt, creating a feedback loop.

#### config.yaml

```yaml
max_iterations: 200
random_seed: 42

llm:
  api_base: "https://generativelanguage.googleapis.com/v1beta/openai/"  # Gemini
  primary_model: "gemini-2.5-flash"
  temperature: 0.7

prompt:
  system_message: |
    You are an expert programmer. Your task is to improve the sorting algorithm
    for better performance and correctness. Focus on algorithmic improvements.
  num_top_programs: 3
  num_diverse_programs: 2
  include_artifacts: true

database:
  population_size: 500
  num_islands: 5
  feature_dimensions: ["complexity", "diversity"]

evaluator:
  enable_artifacts: true
  cascade_evaluation: true
```

**Key config parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iterations` | 10000 | Total evolution iterations |
| `random_seed` | 42 | For reproducibility |
| `llm.api_base` | OpenAI URL | LLM API endpoint |
| `llm.primary_model` | — | Model name (required) |
| `llm.temperature` | 0.7 | Higher = more exploration |
| `prompt.system_message` | generic | Domain-specific guidance for the LLM |
| `prompt.num_top_programs` | 3 | Best programs shown as inspiration |
| `prompt.num_diverse_programs` | 2 | Diverse programs for exploration |
| `database.population_size` | 1000 | MAP-Elites grid capacity |
| `database.num_islands` | 5 | Parallel evolving populations |
| `database.migration_interval` | 50 | Generations between island migration |
| `evaluator.cascade_evaluation` | true | Multi-stage filtering of bad programs |
| `evaluator.enable_artifacts` | true | Feed errors/warnings back to LLM |

**LLM provider examples:**

```yaml
# OpenAI
llm:
  api_base: "https://api.openai.com/v1"
  primary_model: "gpt-4o"

# Google Gemini
llm:
  api_base: "https://generativelanguage.googleapis.com/v1beta/openai/"
  primary_model: "gemini-2.5-pro"

# Local (Ollama)
llm:
  api_base: "http://localhost:11434/v1"
  primary_model: "codellama:7b"
```

### Step 2: Validate the setup

Before running, verify files are correct:

```bash
# Check initial program has EVOLVE-BLOCK markers
grep -n "EVOLVE-BLOCK" my_experiment/initial_program.py

# Check evaluator has evaluate() function
grep -n "def evaluate" my_experiment/evaluator.py

# Dry-run the evaluator on the initial program
python -c "
import sys; sys.path.insert(0, 'my_experiment')
from evaluator import evaluate
print(evaluate('my_experiment/initial_program.py'))
"
```

The evaluator should return a dict with numeric scores. If it crashes or returns 0, fix it before running evolution.

### Step 3: Run evolution

**CLI (recommended):**

```bash
openevolve-run my_experiment/initial_program.py \
  my_experiment/evaluator.py \
  --config my_experiment/config.yaml \
  --iterations 200 \
  --output my_experiment/output
```

**All CLI flags:**

```
openevolve-run <initial_program> <evaluator> [options]

Required:
  initial_program          Path to initial program file
  evaluation_file          Path to evaluator file

Options:
  --config, -c CONFIG      Config YAML file
  --output, -o DIR         Output directory (default: openevolve_output)
  --iterations, -i N       Max iterations (overrides config)
  --target-score, -t SCORE Stop early at this score
  --log-level, -l LEVEL    DEBUG|INFO|WARNING|ERROR
  --checkpoint PATH        Resume from checkpoint directory
  --api-base URL           Override LLM API base URL
  --primary-model NAME     Override primary LLM model
  --secondary-model NAME   Override secondary LLM model
```

**Python API (for programmatic use):**

```python
from openevolve import run_evolution, evolve_function

# File-based
result = run_evolution(
    initial_program="my_experiment/initial_program.py",
    evaluator="my_experiment/evaluator.py",
    config="my_experiment/config.yaml",
    iterations=200,
)
print(f"Best score: {result.best_score}")
print(result.best_code)

# Inline — evolve a function directly
def bubble_sort(arr):
    for i in range(len(arr)):
        for j in range(len(arr)-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

result = evolve_function(
    bubble_sort,
    test_cases=[([3,1,2], [1,2,3]), ([5,2,8], [2,5,8])],
    iterations=50,
)
```

### Step 4: Inspect results

**Output directory structure:**

```
output/
├── checkpoints/
│   ├── checkpoint_100/
│   │   ├── metadata.json       # iteration info, best program ID
│   │   └── programs/
│   │       ├── <program_id>.py # evolved program variants
│   │       └── ...
│   └── checkpoint_200/
└── evolution_trace.jsonl        # per-iteration log (if enabled)
```

**Get the best program from the latest checkpoint:**

```bash
# Find the latest checkpoint
ls -d output/checkpoints/checkpoint_* | sort -t_ -k2 -n | tail -1

# Read metadata to find best program ID
cat output/checkpoints/checkpoint_200/metadata.json | python -m json.tool

# View the best evolved code
cat output/checkpoints/checkpoint_200/programs/<best_program_id>.py
```

**Check score progression (if evolution_trace is enabled):**

```bash
# Extract scores from JSONL trace
python -c "
import json
with open('output/evolution_trace.jsonl') as f:
    for line in f:
        entry = json.loads(line)
        if 'metrics' in entry:
            print(f\"Iter {entry.get('iteration')}: {entry['metrics'].get('combined_score', 'N/A')}\")
"
```

### Step 5: Resume from checkpoint

```bash
openevolve-run my_experiment/initial_program.py \
  my_experiment/evaluator.py \
  --config my_experiment/config.yaml \
  --checkpoint output/checkpoints/checkpoint_200 \
  --iterations 100
```

This loads the MAP-Elites population from the checkpoint and runs 100 more iterations.

### Step 6: Visualize (optional)

```bash
pip install flask plotly
python scripts/visualizer.py --path output/checkpoints/checkpoint_200/
```

Opens a web UI with evolution tree, score progression, code diffs, and MAP-Elites grid.

---

## Writing Effective System Messages

The system message in `config.yaml` is the single most important lever for evolution quality.

**Structure:** Role → Task → Domain knowledge → Allowed changes → Constraints

**Example (GPU kernel optimization):**

```yaml
prompt:
  system_message: |
    You are an expert Metal GPU programmer for Apple Silicon.

    GOAL: Optimize the attention kernel for 5-15% speedup.

    OPTIMIZATION OPPORTUNITIES:
    1. Memory coalescing and vectorized SIMD loading
    2. Fuse max-finding with score computation
    3. Pre-compute frequently used indices

    MUST NOT CHANGE:
    - Kernel function signature
    - Algorithm correctness
    - Template parameter types

    ALLOWED:
    - Memory access patterns
    - Computation order
    - Apple Silicon-specific intrinsics
```

**Iterative refinement process:**
1. Start with a basic message, run 20-50 iterations
2. Check where evolution gets stuck (read artifacts/errors)
3. Add specific guidance for observed failure modes
4. Repeat — the system message should evolve alongside the code

---

## Common Patterns

**Multi-language evolution (Rust, R, etc.):** Set `language` and `file_suffix` in config:

```yaml
language: rust
file_suffix: ".rs"
```

The evaluator can compile and run any language — it just receives a file path.

**Prompt evolution (evolve prompts, not code):** Use the initial program as a prompt template
and the evaluator to test LLM output quality. See `examples/llm_prompt_optimization/`.

**Multi-objective optimization:** Return multiple metrics from the evaluator and add them
as `feature_dimensions` in the database config. OpenEvolve maintains Pareto-optimal solutions.

**Cost management:**
- Start with `gemini-2.5-flash` (~$0.01-0.05/iter) for exploration
- Switch to `gemini-2.5-pro` or `gpt-4o` for refinement
- Use `cascade_evaluation: true` to filter bad programs cheaply
- Start with 100-200 iterations, increase only if scores are still improving

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `OPENAI_API_KEY not set` | `export OPENAI_API_KEY="your-key"` (used for all providers including Gemini) |
| Evaluator always returns 0 | Test evaluator manually: `python -c "from evaluator import evaluate; print(evaluate('initial_program.py'))"` |
| Evolution stuck at same score | Increase `temperature`, add more `num_diverse_programs`, improve system message |
| Out of memory | Reduce `population_size`, enable `cascade_evaluation` |
| LLM rate limits | Add `retry_delay: 10` in llm config, or use OptiLLM proxy |
| Bad evolved code | Enable `enable_artifacts: true` so errors feed back to the LLM |

---

## Built-in Examples

Clone the repo for ready-to-run examples:

```bash
git clone https://github.com/algorithmicsuperintelligence/openevolve.git
cd openevolve
```

| Example | Directory | Description |
|---------|-----------|-------------|
| Function Minimization | `examples/function_minimization/` | Random search → simulated annealing |
| Circle Packing | `examples/circle_packing/` | State-of-the-art n=26 packing |
| GPU Kernels | `examples/mlx_metal_kernel_opt/` | 2.8x Metal kernel speedup |
| Sorting | `examples/rust_adaptive_sort/` | Adaptive sort in Rust |
| Prompt Optimization | `examples/llm_prompt_optimization/` | +23% accuracy on HotpotQA |
| Symbolic Regression | `examples/symbolic_regression/` | Automated equation discovery |
| Signal Processing | `examples/signal_processing/` | Filter design |

Run any example:

```bash
openevolve-run examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py \
  --config examples/function_minimization/config.yaml \
  --iterations 50
```
