# Triton Linear (matmul) Backward Benchmark

This example is a forward-to-backward Triton benchmark for a Linear layer
`y = x @ weight.T + bias`. The task is to implement a correct Triton backward
kernel, then optionally optimize it with OpenEvolve.

Unlike the normalization benchmarks (LayerNorm, RMSNorm), which are memory-bound
and trivially beat PyTorch autograd, this task is **compute-bound**: the autograd
baseline is backed by cuBLAS, so a naive tiled Triton matmul does **not**
automatically win. That makes the speedup metric far more discriminating — real
gains require genuine GEMM engineering (tiling, pipelining, swizzling, split-K).

## Operator

Forward:

```python
y = x @ weight.T + bias            # x [M, K], weight [N, K], bias [N], y [M, N]
```

Backward target:

```python
dx      = dy @ weight              # [M, K]   (contract over N)
dweight = dy.T @ x                 # [N, K]   (contract over M)
dbias   = torch.sum(dy, dim=0)     # [N]
```

Inputs:

```text
dy:     [M, N]
x:      [M, K]
weight: [N, K]
```

Outputs:

```text
dx:      [M, K]
dweight: [N, K]
dbias:   [N]
```

The candidate API is:

```python
def linear_backward_triton(dy, x, weight):
    return dx, dweight, dbias
```

The bias value is not needed for the backward (`dbias = sum(dy)`), so it is not
an input to the candidate.

## Naive Baseline

`backward_naive_triton.py` is a deliberately simple three-kernel baseline:

```text
1. dx      = dy @ weight   via a generic tiled matmul kernel
2. dweight = dy.T @ x      via the same matmul kernel (transpose handled by strides)
3. dbias   = sum(dy, 0)    via a column-reduction kernel
```

Accumulation is in float32, tiles are fixed (64x64x32), no autotuning or fusion.
The correctness oracle is PyTorch autograd over `forward_ref.py` (`F.linear`).

## Files

- `forward_ref.py`: PyTorch semantic reference (`F.linear`).
- `forward_triton.py`: simple tiled Triton forward kernel.
- `backward_ref.py`: PyTorch-autograd gradient oracle.
- `backward_naive_triton.py`: verified naive Triton backward baseline.
- `autograd_wrapper.py`: `torch.autograd.Function` integration.
- `task_spec.py`: correctness cases, oracle, tolerances, and candidate API.
- `evaluator.py`: thin wrapper around the shared evaluator core.
- `benchmark.py`: latency benchmark for the naive baseline.
- `initial_program.py`: OpenEvolve seed program.
- `meta.yaml`: task metadata.
- `config.yaml`: OpenEvolve configuration.

## Correctness

Run on a CUDA-visible GPU node:

```bash
cd openevolve/benchmark/triton_linear_backward_bench
python test_correctness.py
python evaluator.py initial_program.py
```

The evaluator checks `dx`, `dweight`, and `dbias` separately against PyTorch
autograd. Incorrect candidates are not benchmarked. Tolerances are looser than
the elementwise tasks because the matmul reductions (and TF32 on the fp32 path)
introduce more numerical spread.

## OpenEvolve

```bash
cd openevolve/benchmark/triton_linear_backward_bench

openevolve-run \
  initial_program.py \
  evaluator.py \
  --config config.yaml \
  --iterations 10 \
  --output /tmp/openevolve_triton_linear_backward_10 \
  --save-best-to evolved_best_program.py
```
