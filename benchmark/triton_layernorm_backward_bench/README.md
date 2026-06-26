# Triton LayerNorm Backward Benchmark

This example is a forward-to-backward Triton benchmark for row-wise LayerNorm.
The task is to implement a correct Triton backward kernel for a known forward
operator, then optionally optimize it with OpenEvolve.

## Operator

Forward:

```python
mean = torch.mean(x, dim=-1, keepdim=True)
var = torch.mean((x - mean) ** 2, dim=-1, keepdim=True)
xhat = (x - mean) / torch.sqrt(var + eps)
y = xhat * weight + bias
```

Backward target:

```python
dx = (one - mean(one) - xhat * mean(xhat * one)) / torch.sqrt(var + eps)
dweight = torch.sum(dy * xhat, dim=0)
dbias = torch.sum(dy, dim=0)
```

where:

```python
one = dy * weight
```

Inputs:

```text
dy:     [rows, hidden]
x:      [rows, hidden]
weight: [hidden]
bias:   [hidden]
eps:    scalar
```

Outputs:

```text
dx:      [rows, hidden]
dweight: [hidden]
dbias:   [hidden]
```

The candidate API is:

```python
def layernorm_backward_triton(dy, x, weight, bias, eps):
    return dx, dweight, dbias
```

## Naive Baseline

`backward_naive_triton.py` is a deliberately simple seven-kernel baseline:

```text
1. compute xhat and rstd
2. compute one = dy * weight
3. compute mean(one)
4. compute mean(xhat * one)
5. compute dx
6. compute dweight
7. compute dbias
```

This baseline favors readability and verifiability over speed. The correctness
oracle is PyTorch autograd over `forward_ref.py`.

## Files

- `forward_ref.py`: PyTorch semantic reference.
- `forward_triton.py`: simple Triton forward kernel used as layout context.
- `backward_ref.py`: PyTorch-autograd gradient oracle.
- `backward_naive_triton.py`: verified naive Triton backward baseline.
- `autograd_wrapper.py`: `torch.autograd.Function` integration.
- `task_spec.py`: correctness cases, oracle, tolerances, and candidate API.
- `evaluator.py`: thin wrapper around the shared evaluator core.
- `benchmark.py`: latency benchmark for the naive baseline.
- `benchmark_strong_baselines.py`: compares an evolved candidate with PyTorch
  native eager/compile training-step baselines.
- `initial_program.py`: OpenEvolve seed program.
- `evolved_best_program.py`: reference optimized candidate from an initial run.
- `meta.yaml`: task metadata.
- `config.yaml`: OpenEvolve configuration.

## Correctness

Run on a CUDA-visible GPU node:

```bash
cd openevolve/benchmark/triton_layernorm_backward_bench
python test_correctness.py
python evaluator.py initial_program.py
```

The evaluator checks `dx`, `dweight`, and `dbias` separately against PyTorch
autograd. Incorrect candidates are not benchmarked. During OpenEvolve, both
correctness and speedup use PyTorch autograd. Speedup is measured on the unified
dynamic + nontile layernorm shape suite in both float32 and float16.

## OpenEvolve

```bash
cd openevolve/benchmark/triton_layernorm_backward_bench

openevolve-run \
  initial_program.py \
  evaluator.py \
  --config config.yaml \
  --iterations 10 \
  --output /tmp/openevolve_triton_layernorm_backward_10 \
  --save-best-to evolved_best_program.py
```

## NCU-Integrated Evaluator (optional)

`evaluator_autograd_pair_ncu.py` wraps `evaluator_autograd_pair.py`: it runs the
normal timing-based evaluation first, and -- only when a candidate becomes the
new best score seen so far in this run -- profiles it with Nsight Compute
(`ncu`) and folds the result back in:

- The raw metrics are classified against `ncu-report-skill/reference/06-diagnosis-playbook.md`'s
  documented thresholds in `ncu_profile.classify_patterns()` -- the four
  patterns checkable from a `--metrics` pass without `--set source`/PM
  sampling: **Pattern A** (small grid / SM idle), **Pattern E**
  (latency-bound: `long_scoreboard_ratio > 3` and `dram_bytes_read_pct < 10`),
  **Pattern J** (achieved occupancy far below theoretical), **Pattern K**
  (register spill: `local_ld`/`local_st` instructions > 0, or
  `registers_per_thread > 128`). Patterns C/D/F/G/H/I/L/M/N are not
  implemented -- they need deeper profiling (`--set source`, PM-sampling
  timelines) or don't apply to this kernel family (no matmul, no atomics,
  fp32-accumulated by design).
- Only Pattern K (register spill) feeds the score directly -- a penalty
  (`AUTOGRAD_PAIR_NCU_SPILL_PENALTY`, default `0.1`) -- because it's the one
  pattern here with an unambiguous "this is bad" reading. The others are
  exposed as `ncu_pattern_*` metrics and in the `ncu_profile` artifact's
  `patterns` field (each with an `evidence` string citing the exact values
  that triggered it) without changing the score, since e.g. a small grid or
  low occupancy isn't necessarily worse if the candidate is still fastest in
  wall-clock terms.
- The full metric set + Nsight's own rule-engine suggestion are attached as
  the `ncu_profile` artifact, which OpenEvolve's prompt sampler renders into
  the *next* LLM call -- so the model reads a pre-classified hardware
  diagnosis (plus the supporting numbers) instead of guessing from latency
  alone or re-deriving the diagnosis from raw metrics itself.

This is a Triton-native re-implementation of the harness/collection/parsing
workflow from a CUDA-oriented `ncu` profiling skill, not a copy of it: the
"harness" is a throwaway Python script (no `nvcc`, no `-lineinfo` -- Triton
embeds its own source debug metadata), warmup happens before `ncu` attaches to
get past JIT compilation, and every kernel launched in one forward+backward
call is captured and aggregated (a single candidate here can dispatch several
Triton kernels, not one).

Requirements (GPU node only): the `ncu` CLI, perf-counter access (see
`ncu-report-skill/reference/09-common-issues.md` if you hit `ERR_NVGPUCTRPERM`),
and the `ncu_report` Python module on `PYTHONPATH` or discoverable via
`NCU_PYTHON_PATH` (auto-probed under `/usr/local/cuda*/nsight-compute-*/extras/python`).

Smoke-test before trusting it in a real run:
```bash
python benchmark/triton_layernorm_backward_bench/ncu_profile.py \
  benchmark/triton_layernorm_backward_bench/initial_program_autograd_pair.py \
  8 4096 float16
```

Run with OpenEvolve:
```bash
openevolve-run \
  initial_program_autograd_pair.py \
  evaluator_autograd_pair_ncu.py \
  --config config_autograd_pair_ncu.yaml \
  --iterations 10 \
  --output /tmp/openevolve_layernorm_ncu \
  --save-best-to evolved_best_ncu.py
```

Env vars:

| Var | Default | Meaning |
|---|---|---|
| `AUTOGRAD_PAIR_NCU_MODE` | `on_improve` | `off` \| `always` \| `on_improve` -- when to actually invoke `ncu` |
| `AUTOGRAD_PAIR_NCU_SHAPE` | largest `BENCHMARK_CASES` entry | `"rows,cols,dtype"` override for which shape to profile |
| `AUTOGRAD_PAIR_NCU_SPILL_PENALTY` | `0.1` | fractional score penalty applied when register spilling is detected |
| `AUTOGRAD_PAIR_NCU_TIMEOUT` | `120` | seconds before the `ncu` subprocess is killed |
| `AUTOGRAD_PAIR_NCU_STATE` | `.autograd_pair_ncu_state.json` next to this file | where the "best score seen" tracker lives -- set a distinct path per concurrent run to avoid two runs sharing one file |

This whole evaluator is best-effort: if `ncu`/`ncu_report` aren't available or
permission is denied, profiling fails into an `ncu_profile_error` artifact and
scoring falls back to the unmodified timing-based result -- it never blocks or
crashes the evolutionary loop.

## Strong Baseline Check

After an OpenEvolve run:

```bash
python benchmark_strong_baselines.py evolved_best_program.py
```

This reports backward-only and forward+backward latencies against:

```text
pytorch_autograd_backward_ms
pytorch_native_eager_train_step_ms
pytorch_native_compile_train_step_ms
liger_backward_ms                 # optional, requires liger-kernel
liger_forward_backward_ms         # optional
naive_triton_backward_ms          # legacy seed reference
candidate_triton_backward_ms
candidate_speedup_vs_pytorch_autograd
candidate_speedup_vs_liger_backward
```

## Result Snapshot

One initial run optimized the seven-kernel seed into a fused candidate. With the
PyTorch-autograd speedup baseline, reported numbers depend on the seed program
used for evolution; use `benchmark_strong_baselines.py` for Liger and training-step
comparisons after optimization.

Legacy snapshot against the naive Triton seed baseline:

```text
best speedup vs naive Triton backward: 1.9537x
best candidate backward latency:       0.2877 ms
naive baseline backward latency:       0.5622 ms
correctness: dx=1.0, dweight=1.0, dbias=1.0
```

On fp16 benchmark cases, the optimized candidate also outperformed PyTorch
native eager LayerNorm training-step latency in this isolated benchmark.
