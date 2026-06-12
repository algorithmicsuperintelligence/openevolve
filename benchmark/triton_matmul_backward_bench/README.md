# Triton Matmul (GEMM) Backward Benchmark

This example is a forward-to-backward Triton benchmark for a plain matmul
`c = a @ b`. The task is to implement a correct Triton backward kernel, then
optionally optimize it with OpenEvolve.

It is the canonical **compute-bound** task: the autograd baseline is backed by
cuBLAS, so a naive tiled Triton matmul does **not** automatically win. Real gains
require genuine GEMM engineering (tiling, pipelining, swizzling, split-K).

## Operator

Forward:

```python
c = a @ b                 # a [M, K], b [K, N], c [M, N]
```

Backward target:

```python
da = dc @ b.T             # [M, K]   (contract over N)
db = a.T @ dc             # [K, N]   (contract over M)
```

Inputs:

```text
dc: [M, N]
a:  [M, K]
b:  [K, N]
```

Outputs:

```text
da: [M, K]
db: [K, N]
```

The candidate API is:

```python
def matmul_backward_triton(dc, a, b):
    return da, db
```

## Naive Baseline

`backward_naive_triton.py` computes both gradients with one generic tiled matmul
kernel (transposes handled purely through strides), fp32 accumulation, fixed
64x64x32 tiles, no autotuning. The correctness oracle is PyTorch autograd over
`forward_ref.py`.

## Files

- `forward_ref.py`: PyTorch semantic reference (`a @ b`).
- `forward_triton.py`: simple tiled Triton forward kernel.
- `backward_ref.py`: PyTorch-autograd gradient oracle.
- `backward_naive_triton.py`: verified naive Triton backward baseline.
- `autograd_wrapper.py`: `torch.autograd.Function` integration.
- `task_spec.py`: correctness cases, oracle, tolerances, and candidate API.
- `evaluator.py`: thin wrapper around the shared evaluator core.
- `benchmark.py`: latency benchmark for the naive baseline.
- `initial_program.py`: OpenEvolve seed program.
- `meta.yaml`: task metadata.
- `config.yaml`: OpenEvolve configuration (runs on GPT-5.5).

## Correctness

Run on a CUDA-visible GPU node:

```bash
cd openevolve/benchmark/triton_matmul_backward_bench
python test_correctness.py
python evaluator.py initial_program.py
```

The evaluator checks `da` and `db` separately against PyTorch autograd. Incorrect
candidates are not benchmarked. Tolerances are looser than the elementwise tasks
because the matmul reductions (and TF32 on the fp32 path) introduce more spread.

## OpenEvolve

```bash
cd openevolve/benchmark/triton_matmul_backward_bench

export OPENAI_API_KEY=...    # config.yaml runs on gpt-5.5

openevolve-run \
  initial_program.py \
  evaluator.py \
  --config config.yaml \
  --iterations 10 \
  --output /tmp/openevolve_triton_matmul_backward_10 \
  --save-best-to evolved_best_program.py
```
