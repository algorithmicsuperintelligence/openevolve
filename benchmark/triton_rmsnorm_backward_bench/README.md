# Triton RMSNorm Backward Benchmark

This example is a forward-to-backward Triton benchmark for row-wise RMSNorm.
The task is to implement a correct Triton backward kernel for a known forward
operator, then optionally optimize it with OpenEvolve. RMSNorm is the LayerNorm
cousin without mean-centering and without bias, so it produces two gradients
(`dx`, `dweight`) instead of three.

## Operator

Forward:

```python
ms = torch.mean(x ** 2, dim=-1, keepdim=True)
rrms = torch.rsqrt(ms + eps)
xhat = x * rrms
y = xhat * weight
```

Backward target:

```python
g = dy * weight
dx = (g - xhat * mean(g * xhat)) * rrms
dweight = torch.sum(dy * xhat, dim=0)
```

Inputs:

```text
dy:     [rows, hidden]
x:      [rows, hidden]
weight: [hidden]
eps:    scalar
```

Outputs:

```text
dx:      [rows, hidden]
dweight: [hidden]
```

The candidate API is:

```python
def rmsnorm_backward_triton(dy, x, weight, eps):
    return dx, dweight
```

## Naive Baseline

`backward_naive_triton.py` is a deliberately simple five-kernel baseline:

```text
1. compute xhat and rrms
2. compute g = dy * weight
3. compute mean(g * xhat)
4. compute dx
5. compute dweight
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
- `initial_program.py`: OpenEvolve seed program.
- `meta.yaml`: task metadata.
- `config.yaml`: OpenEvolve configuration.

## Correctness

Run on a CUDA-visible GPU node:

```bash
cd openevolve/benchmark/triton_rmsnorm_backward_bench
python test_correctness.py
python evaluator.py initial_program.py
```

The evaluator checks `dx` and `dweight` separately against PyTorch autograd.
Incorrect candidates are not benchmarked. During OpenEvolve, both correctness
and speedup use PyTorch autograd. Speedup is measured on the dynamic + nontile
shape suite in both float32 and float16.

## OpenEvolve

```bash
cd openevolve/benchmark/triton_rmsnorm_backward_bench

openevolve-run \
  initial_program.py \
  evaluator.py \
  --config config.yaml \
  --iterations 10 \
  --output /tmp/openevolve_triton_rmsnorm_backward_10 \
  --save-best-to evolved_best_program.py
```
