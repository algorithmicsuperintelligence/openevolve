# Evolution Analysis: Why Optimization Failed

This document analyzes the evolution experiment results after applying validity fixes, and proposes improvements for future work.

## Experiment Results

After applying validity fixes, we ran 25 evolution iterations to verify that the evaluation now works correctly.

**Note**: The `maximum_context_stress_test` benchmark was disabled to reduce memory requirements on test hardware.

### Evolution Summary

| Metric | Value |
| ------ | ----- |
| Total Iterations | 25 |
| Programs Evaluated | 25 |
| Compilation Failures (bf16) | 8 (32%) |
| Best Program Found | Iteration 23 |
| Best combined_score | 2.96 |
| Benchmarks Used | 4 (stress test disabled) |

### Performance of Best Evolved Kernel

| Benchmark | Baseline (tok/s) | Custom (tok/s) | Change |
| --------- | ---------------- | -------------- | ------ |
| short_context_quick | 59.1 | 63.1 | **+6.9%** ✓ |
| code_generation | 58.3 | 58.1 | -0.4% |
| long_context_detailed | 54.7 | 46.0 | **-15.9%** |
| long_generation | 48.0 | 46.4 | -3.4% |
| **Average** | **55.0** | **53.4** | **-3.2%** |

### Key Finding

> **The best evolved kernel is still 3.2% SLOWER than MLX's baseline implementation.**

The evolution only improved from an initial -11.5% regression to -3.2% regression. It never exceeded baseline performance.

### Evolution Trajectory

```text
Iteration 0 (Initial):  -11.5% regression
Iterations 1-4:         Failed (bf16 compilation errors)
Iteration 5:            -23.6% regression
...
Iteration 19:           -3.6% regression (first "positive" score)
Iteration 23:           -3.2% regression (best found)
Iteration 25:           Evolution complete, no improvement
```

---

## Why Evolution Failed

The failure reveals fundamental limitations in the current evolution mechanism. Framing through a **Reinforcement Learning lens**:

| RL Concept | Current State | Problem |
| ---------- | ------------- | ------- |
| **Reward Signal** | Detailed metrics but abstract ranking score | LLM sees metrics but selection uses opaque `combined_score` |
| **State Representation** | Code text + char-level features | Doesn't capture performance-relevant program properties |
| **Observability** | No GPU profiling data | Partially Observable MDP; agent blind to actual bottlenecks |
| **Credit Assignment** | Per-program metrics, no diff-level attribution | Cannot identify which code mutation caused improvement |
| **Exploration** | 1 parent + 5 samples per iteration | Severely underutilizes available information (128K context) |

### 1. Meaningless Feature Dimensions

Current MAP-Elites dimensions are inadequate for kernel optimization:

| Dimension | Current Implementation | Problem for Kernels |
| --------- | -------------------- |-------------------- |
| `complexity` | Code character count | Two kernels with different algorithms can have similar length |
| `diversity` | Character-level diff | Renaming variables looks "diverse"; algorithmic changes don't |

**What would be meaningful**: tiling strategy, vectorization width, memory access pattern, thread block size.

### 2. Fitness Feedback Interpretability

The LLM receives detailed metrics (decode speed, prefill speed, per-benchmark results), but:

- **Relative performance unclear**: Raw `53.4 tok/s` means little without knowing baseline is `55.0 tok/s`
- **No performance diagnosis**: Cannot tell if kernel is memory-bound vs compute-bound
- **Selection uses abstract score**: MAP-Elites ranking uses `combined_score`, not individual metrics
- **Missing actionable guidance**: "Score: 2.96" doesn't tell LLM what to fix

### 3. Lack of Profiling Data

Without GPU profiling feedback, the LLM is essentially optimizing blind. Metal performance depends heavily on:

- Memory coalescing patterns
- Register pressure
- Warp divergence
- Cache utilization

None of this information is available to guide evolution.

### 4. Conservative Parent Selection

Default configuration uses 70% exploitation (selecting from elites). For kernel optimization where the search space has many local optima, this may cause premature convergence to suboptimal solutions.

### 5. Underutilized LLM Context Window

Each iteration only feeds the LLM:

- 1 parent program
- 3 top programs (inspirations)
- 2 diverse programs

This is extremely conservative given modern LLM context capabilities (128K+ tokens).

**The real cost**: Each evolution iteration is expensive (~10 minutes for model loading + benchmarking), yet the LLM receives minimal information to guide its optimization. This is a **massive waste of resources**.

**Better approach**: Feed the LLM as much context as possible—all programs from the current population, complete benchmark results, historical evolution trajectory. Only apply context pruning when approaching actual model limits.

### 6. High Failure Rate

32% of generated kernels failed to compile with bfloat16. The LLM generates syntactically valid Metal code but often uses float-only operations incompatible with bf16.

### 7. Benchmarking Feedback Quality

While the evaluator returns detailed metrics, the **ranking and selection** uses a single `combined_score`:

```python
# Detailed metrics ARE available to LLM:
performance_metrics = {'avg_decode_speed': 53.4, 'baseline_comparison': {'avg_decode_improvement_pct': -3.2}}

# But MAP-Elites selection uses:
combined_score = 2.96  # What does this mean? Is 3.0 good? Is 10.0 possible?
```

---

## KernelBench Comparison

[KernelBench](https://github.com/ScalingIntelligence/KernelBench) provides a complete, evolution-ready metric system that could address many of these issues:

### KernelBench Evaluation Structure

**1. Binary Correctness Gates**:

```python
class KernelExecResult:
    compiled: bool      # Did the kernel compile?
    correctness: bool   # Did it pass numerical correctness? (multiple trials)
    metadata: dict      # max_difference, avg_difference, error details
```

**2. Primary Optimization Objective** (direct speedup ratio):

```python
speedup = baseline_time / custom_time  # 1.2 = 20% faster, directly interpretable
```

**3. Statistical Rigor**:

```python
runtime_stats = {
    "mean": 3.68,      # Average runtime (ms)
    "std": 0.011,      # Standard deviation
    "min": 3.65,       # Best case
    "max": 3.74,       # Worst case
    "num_trials": 100  # With warmup runs
}
```

**4. Multi-threshold Performance Metrics**:

```python
# fast_p: fraction of kernels that are BOTH correct AND achieve speedup > p
fast_0.0 = 0.85  # 85% correct
fast_1.0 = 0.42  # 42% faster than baseline
fast_1.5 = 0.18  # 18% achieve 1.5x speedup
fast_2.0 = 0.05  # 5% achieve 2x speedup
```

**5. Population-level Metrics**:

```python
geometric_mean_speedup = 1.23  # Average 23% improvement across population
pass_at_1 = 0.42
pass_at_5 = 0.78
```

### How KernelBench Metrics Could Integrate with Evolution

| OpenEvolve Component | Current | KernelBench-style Improvement |
| ------------------- | ------- | ---------------------------- |
| **Fitness Score** | Abstract `combined_score` | Direct `speedup` ratio |
| **Correctness Gate** | Binary pass/fail | Binary + `max_difference`, `avg_difference` for gradient |
| **Performance Feedback** | Single number | `mean ± std` with confidence intervals |
| **MAP-Elites Features** | Code length, char diff | Speedup tier (0.5x, 1x, 1.5x, 2x), runtime variance |
| **Early Stopping** | Fixed threshold | `fast_p` targets: stop when `fast_1.5 > 0.1` |
| **Prompt Feedback** | "Score: 2.96" | "Speedup: 0.85x (15% slower), need to beat 1.0x" |

The key insight: **KernelBench's metrics are designed to be directly actionable**. The LLM can understand "this kernel is 15% slower than baseline" but cannot learn from "combined_score = 2.96".

Additionally, KernelBench enables **temporal credit assignment**:

- Compare child speedup vs parent speedup (not just vs baseline)
- Track which mutations led to improvement
- Provide mutation-specific feedback: "Adding SIMD vectorization improved prefill by 23%"

---

## Proposed Improvements

### Priority 1: Adopt KernelBench-style Evaluation

- Replace `combined_score` with direct speedup ratio: `baseline_time / custom_time`
- Return statistical timing data: mean, std, min, max, num_trials
- Use `fast_p` as milestone targets for early stopping
- Report correctness metrics: `max_difference`, `avg_difference`, tolerance margin
- Provide actionable prompt feedback: "Speedup: 0.85x, need to beat 1.0x"

### Priority 2: Performance-based MAP-Elites Features

- `speedup_tier`: (0-0.5x, 0.5-1x, 1-1.5x, 1.5-2x, >2x) instead of code length
- `runtime_variance`: (low/medium/high std) for consistency tracking
- `correctness_margin`: distance from tolerance threshold

### Priority 3: Integrate Metal GPU Profiling

- Feed occupancy, bandwidth, cache stats back to LLM
- Use profiling data as additional feature dimensions

### Priority 4: Domain-specific Strategy Tracking

- `uses_simd_vectorization: 0-3` (none/2/4/8-wide)
- `memory_access_pattern: coalesced/strided/random`
- `algorithm_type: 2pass/3pass/online`

### Priority 5: Maximize LLM Context Utilization

- Feed entire population (or top N by speedup) instead of just 1 parent + 5 samples
- Include complete benchmark results with statistical breakdowns
- Show evolution history: what worked, what failed, why
- Only prune context when approaching actual model limits (128K+ tokens)

### Priority 6: Curated Metal bf16 Examples

- Add few-shot examples of correct bf16 Metal syntax
- Include common pitfalls in system prompt

---

## References

- [KernelBench](https://github.com/ScalingIntelligence/KernelBench)
- MAP-Elites: Mouret & Clune, 2015

---

*Experiment run: 2026-01-05 18:09 - 21:20 (3h 11m)*
*Note: `maximum_context_stress_test` disabled for this validation run*
