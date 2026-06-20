# Triton Fused LayerNorm -> Linear Backward Benchmark

This example is a forward-to-backward Triton benchmark for a **fused
`LayerNorm -> Linear`** operator, ported from the MegaFold project
([arXiv:2506.20686](https://arxiv.org/abs/2506.20686)). MegaFold replaces a
`LayerNorm(K)` followed by `Linear(K, N, bias=False)` with one fused kernel that
normalizes activation blocks *inside* the matmul, avoiding a round trip of the
normalized activation through HBM. It is the building block of the AF3 Transition
layer (`LayerNorm -> Linear -> SwiGLU -> Linear`).

It is essentially the combination of the existing
[triton_layernorm_backward_bench](../triton_layernorm_backward_bench) and
[triton_linear_backward_bench](../triton_linear_backward_bench) — the **fusion**
is the interesting evolve target.

## Operator

Forward (`linear_weight` is stored as `[K, N]`, matching MegaFold's
`F.linear(x_hat, linear_weight.T)` convention, with no linear bias):

```python
mean  = x.mean(-1, keepdim=True)
var   = ((x - mean) ** 2).mean(-1, keepdim=True)
rstd  = torch.rsqrt(var + eps)
x_hat = (x - mean) * rstd
y_hat = x_hat * weight + bias        # weight=gamma, bias=beta  [K]
out   = y_hat @ linear_weight        # [M, K] @ [K, N] -> [M, N]
```

Backward target:

```python
dy_hat         = dout @ linear_weight.T     # [M, K]
dlinear_weight = y_hat.T @ dout             # [K, N]
dweight        = (dy_hat * x_hat).sum(0)    # [K]   (gamma grad)
dbias          = dy_hat.sum(0)              # [K]   (beta grad)
wdy            = dy_hat * weight
dx             = (wdy - x_hat * (x_hat * wdy).mean(-1) - wdy.mean(-1)) * rstd
```

Inputs / outputs:

```text
dout:          [M, N]
x:             [M, K]
weight (gamma):[K]
bias   (beta): [K]
linear_weight: [K, N]
eps:           scalar

dx:             [M, K]
dlinear_weight: [K, N]
dweight:        [K]
dbias:          [K]
```

The candidate API is:

```python
def layernorm_linear_backward_triton(dout, x, weight, bias, linear_weight, eps):
    return dx, dlinear_weight, dweight, dbias
```

## Naive Baseline (the seed)

`backward_naive_triton.py` is the OpenEvolve seed: a deliberately simple,
hand-written **unfused multi-kernel** Triton backward (per-row stats + two tiled
matmuls + column reductions + a per-row LayerNorm-input pass). The evolve target is
MegaFold's fusion: recompute `x_hat` in the backward instead of storing it, fuse the
`dy_hat = dout @ B^T` matmul with the LayerNorm-input backward (producing the per-row
constants `c1 = mean(x_hat*wdy)`, `c2 = mean(wdy)` in the same pass), and atomic-add
the weight gradients across M-blocks.

`backward_naive.py` is the readable PyTorch reference for the same math. All
seeds/backends are verified against PyTorch autograd on GPU (see
`test_correctness.py`).

Like the Linear benchmark, this task is partly compute-bound — the autograd
baseline is backed by cuBLAS — so a naive backward does not trivially win; real
gains require genuine fusion + GEMM engineering.

## Files

- `forward_ref.py`: PyTorch semantic reference (`layer_norm @ linear_weight`).
- `backward_ref.py`: PyTorch-autograd gradient oracle.
- `backward_naive_triton.py`: verified naive unfused Triton backward (the seed).
- `backward_naive.py`: readable unfused PyTorch reference backward.
- `backward_atenir.py`: AtenIR lowering — runs the autograd-extracted ATen backward
  graph through the AtenIR runtime + primitive-Triton dispatch (per-shape cached
  extraction). Reference graph in `atenir/layernorm_linear_bwd_graph.json`.
- `initial_program_from_atenir.py`: alternate OpenEvolve seed using the AtenIR path.
- `autograd_wrapper.py`: `torch.autograd.Function` integration.
- `task_spec.py`: correctness cases, oracle, tolerances, and candidate API.
- `evaluator.py`: thin wrapper around the shared evaluator core.
- `benchmark.py`: latency benchmark for the naive baseline.
- `benchmark_strong_baselines.py`: compares an evolved candidate against PyTorch
  autograd and MegaFold's fused Triton kernel.
- `initial_program.py`: OpenEvolve seed program.
- `meta.yaml`: task metadata.
- `config.yaml`: OpenEvolve configuration.

## Correctness

Run on a CUDA-visible GPU node:

```bash
cd openevolve/benchmark/triton_layernorm_linear_backward_bench
python test_correctness.py
python evaluator.py initial_program.py
```

The evaluator checks `dx`, `dlinear_weight`, `dweight`, and `dbias` separately
against PyTorch autograd. Incorrect candidates are not benchmarked. Tolerances are
matmul-appropriate (looser than the elementwise norm benches); `dweight`/`dbias`
reduce over M and get extra slack on the fp16 path.

## OpenEvolve

```bash
cd openevolve/benchmark/triton_layernorm_linear_backward_bench

openevolve-run \
  initial_program.py \
  evaluator.py \
  --config config.yaml \
  --iterations 10 \
  --output /tmp/openevolve_triton_layernorm_linear_backward_10 \
  --save-best-to evolved_best_program.py
```

## Strong Baseline Check

After an OpenEvolve run:

```bash
python benchmark_strong_baselines.py evolved_best_program.py
```

This reports backward-only latencies for the candidate, PyTorch autograd, the
naive baseline, and MegaFold's fused Triton `LayernormLinear` (imported from
`MegaFold/megafold/model/FusedLayernormLinear` when its dependencies, e.g.
liger-kernel, are installed), plus candidate speedups vs each.
