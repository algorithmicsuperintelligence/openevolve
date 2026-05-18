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
cd openevolve/examples/triton_layernorm_backward_bench
python test_correctness.py
python evaluator.py initial_program.py
```

The evaluator checks `dx`, `dweight`, and `dbias` separately against PyTorch
autograd. Incorrect candidates are not benchmarked.

## OpenEvolve

```bash
cd openevolve/examples/triton_layernorm_backward_bench

openevolve-run \
  initial_program.py \
  evaluator.py \
  --config config.yaml \
  --iterations 10 \
  --output /tmp/openevolve_triton_layernorm_backward_10 \
  --save-best-to evolved_best_program.py
```

## Strong Baseline Check

After an OpenEvolve run:

```bash
python benchmark_strong_baselines.py evolved_best_program.py
```

This reports:

```text
naive_triton_backward_ms
candidate_triton_backward_ms
candidate_triton_forward_backward_ms
pytorch_native_eager_train_step_ms
pytorch_native_compile_train_step_ms
```

## Result Snapshot

One initial run optimized the seven-kernel baseline into a two-kernel candidate:

```text
best speedup vs naive Triton backward: 1.9537x
best candidate backward latency:       0.2877 ms
naive baseline backward latency:       0.5622 ms
correctness: dx=1.0, dweight=1.0, dbias=1.0
```

On fp16 benchmark cases, the optimized candidate also outperformed PyTorch
native eager LayerNorm training-step latency in this isolated benchmark.
