# MLX Metal Kernel Optimization Example

This example uses OpenEvolve to automatically discover optimized Metal GPU kernels for Grouped Query Attention (GQA) in Qwen3-0.6B on Apple Silicon.

## Target Configuration

- **Model**: Qwen3-0.6B-bf16
- **Architecture**: 16 query heads : 8 KV heads (2:1 ratio), 2048 hidden size, 128 head dimension
- **Hardware**: Apple M-series GPUs with unified memory
- **Baseline**: `mx.fast.scaled_dot_product_attention` via `mlx_lm.generate`
- **Goal**: Evolve custom Metal kernel source code to outperform baseline

## Quick Start

### Prerequisites

```bash
pip install mlx mlx-lm openevolve

# Set API key (Gemini via OpenAI-compatible endpoint)
export OPENAI_API_KEY="your-gemini-key"
```

### Run Evolution

```bash
cd openevolve/examples/mlx_metal_kernel_opt

# Using the experiment runner script
./run_evolve_experiment.sh --run-name test_run --iterations 25

# Or directly
python -m openevolve.cli \
    --initial-program initial_program.py \
    --evaluator evaluator.py \
    --config config.yaml \
    --iterations 25 \
    --output ./openevolve_output
```

### Verify Evaluation Validity

```bash
# Run a single benchmark with verbose output
python -c "
from evaluator import Qwen3GQAEvaluator
e = Qwen3GQAEvaluator()
result = e.evaluate('initial_program.py')
print(result['summary'])
"
```

## Files

| File | Purpose |
| ---- | ------- |
| `initial_program.py` | Starting Metal kernel (to be evolved) |
| `evaluator.py` | Correctness + performance evaluation |
| `config.yaml` | Evolution configuration |
| `qwen3_benchmark_suite.py` | Benchmark definitions |
| `mlx_lm_generate_with_hook.py` | Subprocess hook wrapper |
| `run_evolve_experiment.sh` | Experiment runner script |

## Validity Fixes (This PR)

This PR corrects critical issues that invalidated prior evaluation results:

1. **Subprocess Kernel Hook**: Evolved kernels are now properly applied in benchmark subprocesses via `mlx_lm_generate_with_hook.py`

2. **bfloat16 Correctness Gate**: Correctness tests now use `mx.bfloat16` inputs to match actual inference dtype

3. **Architecture Alignment**: Fixed head ratio from 40:8 to correct 16:8 (2:1 GQA pattern)

4. **Evaluation Flow Optimizations**: Early exit on compilation errors, correctness-before-baseline ordering, GPU state cleanup between runs

## Current Status

After fixing validity issues, we ran 25 evolution iterations.

**Result: The best evolved kernel is 3.2% SLOWER than MLX's baseline implementation.**

The evolution improved from an initial -11.5% regression to -3.2%, but never exceeded baseline. This indicates fundamental limitations in the current evolution mechanism that require further investigation.

For detailed experiment results and analysis, see [EVOLUTION_ANALYSIS.md](./EVOLUTION_ANALYSIS.md).

### Demo Results (Committed)

For review and reproducibility, this example repo includes a committed snapshot of one post-fix evolution run:

- `best_program.py`: best evolved program (iteration 23)
- `best_program_info.json`: metrics + baseline comparisons (includes the -3.2% result)

The full run output directory is intentionally git-ignored (see `.gitignore`) to avoid committing large run artifacts.

### Known Limitations

1. MAP-Elites selection uses abstract `combined_score` instead of direct speedup ratios
2. LLM context underutilized (only 1 parent + 5 samples per iteration)
3. No GPU profiling data to guide optimization
4. 32% bf16 compilation failure rate

## References

- [OpenEvolve](https://github.com/codelion/openevolve)
- [MLX](https://github.com/ml-explore/mlx)
- [MLX-LM](https://github.com/ml-explore/mlx-examples)
- [KernelBench](https://github.com/ScalingIntelligence/KernelBench)
