# CLAUDE.md

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

OpenEvolve is an open-source implementation of Google DeepMind's AlphaEvolve system - an evolutionary coding agent that uses LLMs to optimize code through iterative evolution. The framework can evolve code in multiple languages (Python, R, Rust, Metal shaders, etc.) for tasks like scientific computing, optimization, algorithm discovery, and GPU kernel optimization.

**Key Achievement**: Production-ready framework with proven results including 2-3x GPU speedups, state-of-the-art circle packing solutions, and automated algorithm discovery across multiple domains.

## Essential Commands

### Development Setup
```bash
# Install in development mode with all dependencies
pip install -e ".[dev]"

# Or use Makefile
make install

# Install with integration test support (includes optillm)
make install-dev
```

### Running Tests
```bash
# Run unit tests only (fast, no LLM required)
make test
# or: python -m unittest discover -s tests -p "test_*.py"

# Run integration tests (requires optillm, tests with actual LLM)
make test-integration

# Run all tests (unit + integration)
make test-all

# Run integration tests with existing optillm server (for development)
make test-integration-dev
```

### Code Formatting
```bash
# Format with Black (line length: 100)
python -m black openevolve examples tests scripts

# Or use Makefile
make lint
```

### Running OpenEvolve

**Command Line Usage:**
```bash
# Basic evolution run
python openevolve-run.py path/to/initial_program.py path/to/evaluator.py --config path/to/config.yaml --iterations 1000

# Using the installed CLI
openevolve-run path/to/initial_program.py path/to/evaluator.py --config path/to/config.yaml --iterations 1000

# Resume from checkpoint
python openevolve-run.py path/to/initial_program.py path/to/evaluator.py \
  --config path/to/config.yaml \
  --checkpoint path/to/checkpoint_directory \
  --iterations 50

# Override model in config
openevolve-run examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py \
  --config examples/function_minimization/config.yaml \
  --model gemini-2.0-flash-lite \
  --iterations 50
```

**Library Usage:**
```python
from openevolve import run_evolution, evolve_function

# Evolution with inline code (no files needed!)
result = run_evolution(
    initial_program='''
    def fibonacci(n):
        if n <= 1: return n
        return fibonacci(n-1) + fibonacci(n-2)
    ''',
    evaluator=lambda path: {"score": benchmark_fib(path)},
    iterations=100
)

# Evolve Python functions directly
def bubble_sort(arr):
    for i in range(len(arr)):
        for j in range(len(arr)-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

result = evolve_function(
    bubble_sort,
    test_cases=[([3,1,2], [1,2,3]), ([5,2,8], [2,5,8])],
    iterations=50
)
print(f"Evolved code: {result.best_code}")
```

### Visualization
```bash
# Install visualization dependencies
pip install -r scripts/requirements.txt

# Launch interactive visualizer
python scripts/visualizer.py

# Or visualize specific checkpoint
python scripts/visualizer.py --path examples/function_minimization/openevolve_output/checkpoints/checkpoint_100/

# Or use Makefile
make visualizer
```

## Directory Structure

```
openevolve/
├── openevolve/              # Main package
│   ├── __init__.py         # Package exports (run_evolution, evolve_function, etc.)
│   ├── _version.py         # Version management
│   ├── api.py              # High-level API (run_evolution, evolve_function)
│   ├── cli.py              # Command-line interface
│   ├── config.py           # Configuration management and validation
│   ├── controller.py       # Main orchestrator for evolution process
│   ├── database.py         # MAP-Elites database with island-based evolution
│   ├── embedding.py        # Text embedding for similarity detection
│   ├── evaluator.py        # Program evaluation with cascade pattern
│   ├── evaluation_result.py # Dataclass for evaluation results
│   ├── evolution_trace.py  # Evolution trace logging for RL training
│   ├── iteration.py        # Single iteration worker process
│   ├── novelty_judge.py    # Novelty detection for programs
│   ├── process_parallel.py # Parallel process execution utilities
│   ├── llm/                # LLM integration
│   │   ├── base.py        # Base LLM interface
│   │   ├── openai.py      # OpenAI-compatible API client
│   │   └── ensemble.py    # Ensemble of multiple models
│   ├── prompt/             # [Legacy directory - unused]
│   ├── prompts/            # Prompt templates
│   │   └── defaults/      # Default prompt templates
│   │       ├── system_message.txt
│   │       ├── evaluator_system_message.txt
│   │       ├── diff_user.txt
│   │       ├── full_rewrite_user.txt
│   │       ├── evaluation.txt
│   │       ├── evolution_history.txt
│   │       ├── inspiration_program.txt
│   │       ├── inspirations_section.txt
│   │       ├── previous_attempt.txt
│   │       ├── top_program.txt
│   │       └── fragments.json  # Template variation fragments
│   └── utils/              # Utility modules
│       ├── async_utils.py  # Async/await helper functions
│       ├── code_utils.py   # Code parsing and manipulation
│       ├── format_utils.py # String formatting utilities
│       ├── metrics_utils.py # Metrics calculation and aggregation
│       └── trace_export_utils.py # Evolution trace export utilities
├── configs/                # Configuration examples
│   ├── default_config.yaml
│   ├── island_config_example.yaml
│   ├── island_examples.yaml
│   └── early_stopping_example.yaml
├── examples/               # Example projects (17 examples)
│   ├── function_minimization/
│   ├── circle_packing/
│   ├── mlx_metal_kernel_opt/
│   ├── rust_adaptive_sort/
│   ├── llm_prompt_optimization/
│   ├── symbolic_regression/
│   ├── web_scraper_optillm/
│   ├── alphaevolve_math_problems/
│   ├── attention_optimization/
│   ├── algotune/
│   └── ... (and more)
├── tests/                  # Test suite
│   ├── test_*.py          # Unit tests (39 test files)
│   └── integration/       # Integration tests
├── scripts/
│   └── visualizer.py      # Web-based evolution visualizer
├── openevolve-run.py      # Main entry point script
├── setup.py               # Package setup
├── pyproject.toml         # Modern Python packaging config
├── Makefile               # Development automation
├── Dockerfile             # Docker containerization
└── README.md              # User-facing documentation
```

## High-Level Architecture

### Core Components

1. **Controller (`openevolve/controller.py`)**:
   - Main orchestrator managing the evolution process
   - Uses ProcessPoolExecutor for parallel iteration execution
   - Handles checkpointing, resumption, and early stopping
   - Coordinates between database, evaluator, and LLM
   - Key functions: `run()`, `_save_checkpoint()`, `_load_checkpoint()`

2. **Database (`openevolve/database.py`)**:
   - Implements MAP-Elites algorithm with island-based evolution
   - Programs mapped to multi-dimensional feature grid (quality-diversity)
   - Multiple isolated populations (islands) evolve independently
   - Periodic migration between islands prevents premature convergence
   - Tracks absolute best program across all islands
   - Key classes: `Database`, `Island`, `Program`
   - Key methods: `add_program()`, `sample_programs()`, `migrate()`, `get_best_program()`

3. **Evaluator (`openevolve/evaluator.py`)**:
   - Cascade evaluation pattern for efficiency:
     - Stage 1: Quick validation (syntax, basic correctness)
     - Stage 2: Basic performance testing
     - Stage 3: Comprehensive evaluation
   - Programs must pass thresholds at each stage
   - Parallel evaluation using TaskPool
   - Optional LLM-based feedback integration
   - Key functions: `evaluate_program()`, `_cascade_evaluate()`

4. **LLM Integration (`openevolve/llm/`)**:
   - Ensemble approach with multiple models
   - Configurable model weights and fallback strategies
   - Async generation with retry logic and timeout handling
   - Supports any OpenAI-compatible API
   - Key classes: `OpenAIModel`, `LLMEnsemble`
   - Handles both evolution models and evaluator models separately

5. **Iteration (`openevolve/iteration.py`)**:
   - Worker process that executes a single evolution iteration
   - Samples programs from islands
   - Generates mutations via LLM
   - Evaluates mutated programs
   - Stores artifacts (execution logs, errors, etc.)
   - Runs in isolated process with database snapshot
   - Key function: `run_iteration()`

6. **API (`openevolve/api.py`)**:
   - High-level interface for library usage
   - `run_evolution()`: Full evolution with custom evaluator
   - `evolve_function()`: Simplified interface for Python functions
   - Handles configuration setup and result packaging

7. **Configuration (`openevolve/config.py`)**:
   - YAML-based hierarchical configuration
   - Validation and default value handling
   - Supports environment variable interpolation
   - Key class: `Config` with nested structures for llm, database, evaluator, prompt

8. **Evolution Trace (`openevolve/evolution_trace.py`)**:
   - Logs detailed traces for RL training and analysis
   - Supports JSONL, JSON, and HDF5 formats
   - Configurable inclusion of code, prompts, and responses
   - Buffered writing for performance

### Key Architectural Patterns

- **Island-Based Evolution**: Multiple populations (islands) evolve separately with periodic migration. Each island maintains its own MAP-Elites grid. Migration happens on a per-island basis when islands reach generation thresholds.

- **MAP-Elites**: Quality-diversity algorithm that maintains diversity by mapping programs to feature grid cells. Each cell stores the best program for that feature combination. Built-in features (complexity, diversity) + custom features from evaluator.

- **Artifact System**: Side-channel for programs to return debugging data:
  - Small artifacts (<10KB): Stored in database as JSON
  - Large artifacts: Saved to disk as files
  - Includes: stderr, stdout, profiling_data, llm_feedback, build_warnings
  - Automatically included in next generation's prompt for learning

- **Process Worker Pattern**: Each iteration runs in a fresh process with database snapshot. Provides:
  - Isolation from main process
  - True parallelism
  - Clean environment for each evaluation
  - Resource cleanup after completion

- **Double-Selection**: Programs used for inspiration (shown to LLM) differ from programs used for performance baseline. Allows exploring diverse solutions while maintaining quality.

- **Lazy Migration**: Islands migrate based on their own generation counts, not global iterations. Prevents forced synchronization and allows natural evolution pacing.

- **Cascade Evaluation**: Multi-stage evaluation with increasing computational cost:
  - Early exit for invalid programs
  - Reduces wasted computation
  - Configurable thresholds per stage

- **Template Stochasticity**: Random variations in prompt templates using fragment substitution. Adds diversity to LLM prompts across generations.

### Code Evolution Markers

Mark code sections to evolve using special comments:
```python
# EVOLVE-BLOCK-START
# Code to evolve goes here
def optimize_me():
    pass
# EVOLVE-BLOCK-END
```

**Behavior:**
- `diff_based_evolution: true`: LLM sees only the marked block and suggests diffs
- `diff_based_evolution: false`: LLM sees full file and rewrites entire block
- No markers: Entire file is evolved

### Configuration System

OpenEvolve uses YAML configuration with hierarchical structure. See `configs/default_config.yaml` for all available options.

**Key Configuration Sections:**

1. **General Settings:**
   - `max_iterations`: Maximum evolution iterations
   - `checkpoint_interval`: Save frequency
   - `random_seed`: For reproducibility (default: 42)
   - `early_stopping_patience`: Convergence detection
   - `diff_based_evolution`: Diff-based vs full rewrites

2. **LLM Configuration (`llm`):**
   - **Model Ensemble:**
     - `models`: List of models with weights for evolution
     - `evaluator_models`: Separate models for LLM feedback
     - Example: `[{name: "gemini-2.0-flash-lite", weight: 0.8}, {name: "gemini-2.0-flash", weight: 0.2}]`
   - **API Settings:**
     - `api_base`: Base URL (supports OpenAI, Gemini, local models, OptiLLM)
     - `api_key`: Defaults to OPENAI_API_KEY env var
   - **Generation Parameters:**
     - `temperature`: 0.0-2.0 (default: 0.7)
     - `top_p`: 0.0-1.0 (default: 0.95)
     - `max_tokens`: Token limit (default: 4096)
   - **Retry Logic:**
     - `timeout`: Per-request timeout (default: 60s)
     - `retries`: Max retry attempts (default: 3)
     - `retry_delay`: Delay between retries (default: 5s)

3. **Prompt Configuration (`prompt`):**
   - `system_message`: Core instruction for LLM (most important!)
   - `evaluator_system_message`: For LLM-based feedback
   - `template_dir`: Custom template directory
   - `num_top_programs`: Elite programs to show (default: 3)
   - `num_diverse_programs`: Diverse programs for exploration (default: 2)
   - `use_template_stochasticity`: Random prompt variations (default: true)
   - `template_variations`: Custom fragment substitutions
   - `include_artifacts`: Show execution feedback (default: true)
   - `max_artifact_bytes`: Artifact size limit (default: 20KB)

4. **Database Configuration (`database`):**
   - **Population Management:**
     - `population_size`: Max programs in memory (default: 1000)
     - `archive_size`: Elite archive size (default: 100)
     - `num_islands`: Number of separate populations (default: 5)
   - **Island Migration:**
     - `migration_interval`: Generations between migrations (default: 50)
     - `migration_rate`: Fraction of programs to migrate (default: 0.1)
   - **Selection Parameters:**
     - `elite_selection_ratio`: Elite program selection (default: 0.1)
     - `exploitation_ratio`: Exploit vs explore (default: 0.7)
     - `exploration_ratio`: Random exploration (default: 0.2)
   - **MAP-Elites Features:**
     - `feature_dimensions`: List of features (built-in or custom)
       - Built-in: "complexity" (code length), "diversity" (structure)
       - Custom: Any metrics returned by evaluator
     - `feature_bins`: Bins per dimension (int or dict)
     - `diversity_reference_size`: Reference set size (default: 20)

5. **Evaluator Configuration (`evaluator`):**
   - **Basic Settings:**
     - `timeout`: Max evaluation time (default: 300s)
     - `max_retries`: Retry attempts (default: 3)
     - `parallel_evaluations`: Concurrent evaluations (default: 4)
   - **Cascade Evaluation:**
     - `cascade_evaluation`: Enable multi-stage (default: true)
     - `cascade_thresholds`: List of thresholds per stage
   - **LLM Feedback:**
     - `use_llm_feedback`: Enable AI code review (default: false)
     - `llm_feedback_weight`: Weight in final score (default: 0.1)

6. **Evolution Trace (`evolution_trace`):**
   - `enabled`: Enable trace logging (default: false)
   - `format`: 'jsonl', 'json', or 'hdf5'
   - `include_code`: Include full program code (default: false)
   - `include_prompts`: Include LLM interactions (default: true)
   - `output_path`: Custom path (defaults to output_dir)
   - `buffer_size`: Buffering for performance (default: 10)
   - `compress`: GZIP compression (default: false)

**Configuration Examples:**

```yaml
# Simple configuration
max_iterations: 100
llm:
  models:
    - name: "gemini-2.0-flash-lite"
      weight: 1.0
  api_base: "https://generativelanguage.googleapis.com/v1beta/openai/"
prompt:
  system_message: "You are an expert optimizer..."
```

```yaml
# Advanced configuration with custom features
database:
  feature_dimensions:
    - "performance"      # Custom from evaluator
    - "memory_efficiency" # Custom from evaluator
    - "complexity"       # Built-in
  feature_bins:
    performance: 20
    memory_efficiency: 15
    complexity: 10
```

### Prompt System

OpenEvolve uses a sophisticated prompt building system:

**Prompt Templates** (`openevolve/prompts/defaults/`):
- `system_message.txt`: Base system message
- `diff_user.txt`: Template for diff-based evolution
- `full_rewrite_user.txt`: Template for full rewrites
- `evaluation.txt`: Evaluation results formatting
- `evolution_history.txt`: Historical performance data
- `inspiration_program.txt`: Single program example
- `inspirations_section.txt`: Multiple program examples
- `previous_attempt.txt`: Failed attempt feedback
- `top_program.txt`: Best program showcase
- `fragments.json`: Template variation fragments

**Prompt Building Process:**
1. Start with system message (from config or template)
2. Add task description and constraints
3. Include top-performing programs (inspiration)
4. Include diverse programs (exploration)
5. Add previous attempt feedback (if retrying)
6. Include artifacts (execution logs, errors)
7. Add evolution history (recent improvements)
8. Apply template stochasticity (random variations)
9. Format for diff-based or full rewrite mode

**Template Stochasticity:**
- Randomly substitutes fragments in templates
- Defined in `fragments.json` or config `template_variations`
- Example: `{improvement_suggestion}` → random choice from variations
- Adds diversity to prompts without changing core meaning

### Important Patterns

1. **Checkpoint/Resume**:
   - Automatic saving of entire system state
   - Includes: database, config, iteration count, random state
   - Seamless resume from any checkpoint
   - Location: `{output_dir}/checkpoints/checkpoint_{iteration}/`

2. **Parallel Evaluation**:
   - Multiple programs evaluated concurrently via ProcessPoolExecutor
   - Configurable with `evaluator.parallel_evaluations`
   - Each evaluation runs in isolated process
   - Timeouts and resource limits enforced

3. **Error Resilience**:
   - Individual failures don't crash system
   - Extensive retry logic at all levels (LLM, evaluation, etc.)
   - Timeout protection for all operations
   - Graceful degradation when components fail

4. **Prompt Engineering**:
   - Template-based system with context-aware building
   - Evolution history tracking
   - Artifact feedback loop
   - Template stochasticity for diversity
   - Customizable via config or custom template directory

5. **Reproducibility**:
   - `random_seed: 42` by default
   - Seeds: Python random, numpy, LLM temperature (if deterministic)
   - Component isolation via hashing
   - Deterministic evolution across runs

6. **Fitness Calculation**:
   - Uses `combined_score` if returned by evaluator
   - Otherwise: averages all metrics EXCEPT feature_dimensions
   - Features are for diversity, not fitness
   - Higher fitness = better program

7. **Artifact Threshold**:
   - Small (<10KB): Stored in database as JSON
   - Large (≥10KB): Saved to disk with reference
   - Automatic cleanup of old artifacts
   - Security filtering for sensitive data

8. **Process Workers**:
   - Load database snapshots for true parallelism
   - Fresh Python interpreter per iteration
   - Resource isolation and cleanup
   - Serialize/deserialize via pickle

## Development Workflow

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/algorithmicsuperintelligence/openevolve.git
cd openevolve

# Create virtual environment
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Verify installation
python -m unittest discover tests
```

### Making Changes

1. **Create a feature branch:**
   ```bash
   git checkout -b feat-my-feature
   ```

2. **Make changes and format code:**
   ```bash
   # Edit files...
   make lint  # Format with Black
   ```

3. **Run tests:**
   ```bash
   make test           # Unit tests
   make test-all       # Unit + integration tests
   ```

4. **Commit and push:**
   ```bash
   git add .
   git commit -m "feat: add my feature"
   git push origin feat-my-feature
   ```

### Testing Strategy

**Unit Tests** (`tests/test_*.py`):
- Fast, no external dependencies
- Mock LLM responses
- Test individual components
- Run with: `make test`

**Integration Tests** (`tests/integration/`):
- Test with real LLM (optillm + local model)
- End-to-end workflows
- Slower but more comprehensive
- Run with: `make test-integration`

**Key Test Files:**
- `test_database.py`: MAP-Elites and island logic
- `test_evaluator_timeout.py`: Timeout handling
- `test_island_migration.py`: Migration mechanics
- `test_checkpoint_resume.py`: Checkpoint functionality
- `test_artifacts.py`: Artifact system
- `test_evolution_trace.py`: Trace logging
- `test_llm_ensemble.py`: LLM ensemble behavior

### Common Development Tasks

**Adding a New Feature:**
1. Add unit tests first (TDD)
2. Implement feature
3. Update config schema if needed
4. Add integration test if applicable
5. Update documentation (README.md, this file)
6. Format code with Black

**Debugging Evolution Issues:**
1. Enable DEBUG logging in config: `log_level: "DEBUG"`
2. Check artifacts for execution errors
3. Use visualizer to inspect evolution tree
4. Examine prompts in database: `database.log_prompts: true`
5. Run with single island for simpler debugging: `database.num_islands: 1`

**Performance Optimization:**
1. Profile with cProfile: `python -m cProfile -o profile.stats openevolve-run.py ...`
2. Increase `evaluator.parallel_evaluations`
3. Reduce `database.population_size` for less memory
4. Use cascade evaluation to filter early
5. Enable compilation caching in evaluator

### Code Style Guidelines

- **Formatting**: Black with 100 character line length
- **Type Hints**: Use type hints for all functions (mypy-compatible)
- **Docstrings**: Google-style docstrings for public APIs
- **Imports**: isort with black profile
- **Naming**:
  - Functions/variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`
  - Private: prefix with `_`

### Release Process

1. Update version in `openevolve/_version.py`
2. Update CHANGELOG (if exists)
3. Run all tests: `make test-all`
4. Build package: `python -m build`
5. Upload to PyPI: `python -m twine upload dist/*`
6. Tag release: `git tag v{version} && git push --tags`

## Utilities (`openevolve/utils/`)

### `async_utils.py`
- Async/await helper functions
- Concurrent execution with timeout
- Retry decorators for async functions
- Example: `async_retry()`, `run_with_timeout()`

### `code_utils.py`
- Code parsing and manipulation
- Extract evolve blocks from code
- Apply diffs to code
- Calculate code complexity metrics
- Example: `extract_evolve_block()`, `apply_diff()`, `calculate_complexity()`

### `format_utils.py`
- String formatting utilities
- Pretty printing of data structures
- Truncation helpers
- Example: `truncate_string()`, `format_dict()`

### `metrics_utils.py`
- Metrics calculation and aggregation
- Statistical analysis functions
- Performance tracking
- Example: `aggregate_metrics()`, `calculate_statistics()`

### `trace_export_utils.py`
- Evolution trace export to various formats
- JSONL, JSON, HDF5 writers
- Compression support
- Example: `export_trace_jsonl()`, `export_trace_hdf5()`

## Examples (`examples/`)

OpenEvolve includes 17+ comprehensive examples across different domains:

### Quick Start Examples
- **function_minimization/**: Optimization algorithm evolution (best starting point)
- **circle_packing/**: Geometric optimization with visualization

### Hardware Optimization
- **mlx_metal_kernel_opt/**: Apple Silicon GPU kernel optimization
- **attention_optimization/**: Attention mechanism optimization

### Algorithm Discovery
- **rust_adaptive_sort/**: Data-aware sorting algorithm evolution
- **symbolic_regression/**: Automated equation discovery

### AI Integration
- **llm_prompt_optimization/**: Meta-evolution of prompts
- **web_scraper_optillm/**: Web scraping with test-time compute

### Scientific Computing
- **signal_processing/**: Filter design automation
- **r_robust_regression/**: R statistical method optimization

### Domain-Specific
- **online_judge_programming/**: Competitive programming solutions
- **alphaevolve_math_problems/**: Mathematical problem solving
- **algotune/**: Algorithm parameter tuning
- **lm_eval/**: Language model evaluation harness

Each example includes:
- `initial_program.{py,rs,r}`: Starting code
- `evaluator.py`: Evaluation logic
- `config.yaml`: Configuration
- `README.md`: Description and results

## Troubleshooting

### Common Issues

**Issue: Evolution gets stuck / no improvement**
- Solution 1: Increase `prompt.num_diverse_programs` for more exploration
- Solution 2: Adjust `database.migration_interval` for more cross-pollination
- Solution 3: Enable `prompt.use_template_stochasticity` for prompt diversity
- Solution 4: Review and improve `prompt.system_message` with domain knowledge

**Issue: Out of memory**
- Solution 1: Reduce `database.population_size`
- Solution 2: Reduce `database.num_islands`
- Solution 3: Reduce `evaluator.parallel_evaluations`
- Solution 4: Set `evolution_trace.include_code: false`

**Issue: API rate limits**
- Solution 1: Reduce `evaluator.parallel_evaluations`
- Solution 2: Increase `llm.retry_delay`
- Solution 3: Use OptiLLM for intelligent rate limiting
- Solution 4: Switch to local models (Ollama, vLLM)

**Issue: Programs fail evaluation**
- Solution 1: Check artifacts for error messages
- Solution 2: Increase `evaluator.timeout`
- Solution 3: Review `prompt.system_message` for clarity
- Solution 4: Add constraints in system message

**Issue: Checkpoint resume fails**
- Solution 1: Ensure checkpoint directory is complete
- Solution 2: Check for version mismatch (OpenEvolve version)
- Solution 3: Verify checkpoint wasn't created during error
- Solution 4: Start fresh if checkpoint is corrupted

### Getting Help

1. **Documentation**: Read README.md for detailed usage
2. **Examples**: Check `examples/` for similar use cases
3. **Tests**: Look at `tests/` for API usage patterns
4. **Issues**: Search/create GitHub issues
5. **Discussions**: GitHub Discussions for questions

## Key Conventions for AI Assistants

When working with this codebase:

1. **Always format code with Black** before committing:
   ```bash
   python -m black openevolve examples tests scripts
   ```

2. **Run tests before pushing**:
   ```bash
   make test  # At minimum, run unit tests
   ```

3. **Respect the architecture**:
   - Don't bypass the Controller for evolution logic
   - Use Database API for all program storage
   - Let Evaluator handle all program execution
   - Use LLMEnsemble for all LLM calls

4. **Configuration changes**:
   - Update `configs/default_config.yaml` with new options
   - Update `config.py` validation logic
   - Document in README.md if user-facing

5. **Adding features**:
   - Write unit tests first
   - Update type hints
   - Add to `__init__.py` if public API
   - Update this CLAUDE.md file

6. **Debugging approach**:
   - Enable DEBUG logging first
   - Check artifacts for execution context
   - Use visualizer for evolution analysis
   - Examine prompts if LLM behavior is unexpected

7. **Performance considerations**:
   - Evaluation is the bottleneck (optimize evaluators)
   - LLM calls are second bottleneck (use ensemble fallback)
   - Database operations are fast (in-memory)
   - Prefer parallel operations where possible

8. **Error handling**:
   - All external calls should have timeouts
   - All operations should have retries
   - Log errors with context
   - Gracefully degrade when possible

9. **Type hints**:
   - Use type hints for all function signatures
   - Use dataclasses for structured data
   - Enable mypy checks for new code

10. **Documentation**:
    - Update README.md for user-facing changes
    - Update CLAUDE.md for development changes
    - Add docstrings for new public APIs
    - Include examples in docstrings

## Additional Resources

- **Main README**: `/home/user/openevolve/README.md` - Comprehensive user guide
- **Configuration Examples**: `/home/user/openevolve/configs/` - All config options
- **Examples**: `/home/user/openevolve/examples/` - 17+ complete examples
- **Tests**: `/home/user/openevolve/tests/` - Usage patterns and edge cases
- **Visualizer**: `/home/user/openevolve/scripts/visualizer.py` - Evolution analysis tool

## Version Information

- **Python**: >=3.10 required
- **Key Dependencies**: openai>=1.0.0, pyyaml>=6.0, numpy>=1.22.0, tqdm>=4.64.0, flask
- **Optional**: pytest, black, isort, mypy (dev dependencies)
- **Testing**: pytest, pytest-asyncio, requests (integration tests)
- **LLM Support**: Any OpenAI-compatible API (OpenAI, Google Gemini, Ollama, vLLM, OptiLLM, etc.)
