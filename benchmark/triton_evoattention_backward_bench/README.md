# Triton EvoformerAttention (3D EvoAttention) Backward Benchmark

This example is a forward-to-backward Triton benchmark for **AlphaFold3-style
EvoformerAttention**, ported from the MegaFold project
([arXiv:2506.20686](https://arxiv.org/abs/2506.20686)). EvoAttention is the most
memory-intensive operator in AF3 training, and MegaFold's headline kernel is a
fused, flash-attention-style Triton implementation of it.

It differs from ordinary (2D) attention in three ways that have **no analog** in
the other benchmarks here:

1. **3D / 5D layout** — attention runs over a 5D tensor with an extra MSA axis
   `N_seq`, so the program grid is `(B * N_seq * Head, residue blocks)`.
2. **Trainable pair bias** — a pairwise bias `[B, 1, Head, N_res, N_res]` is added
   to the scores and broadcast over `N_seq`, so the backward must produce
   **`d_pair_bias`** (summed over `N_seq`).
3. **Additive residue mask** — a per-key mask `[B, N_seq, 1, 1, N_res]`, no grad.

## Operator

Forward (`q, k, v` are `[B, N_seq, N_res, Head, Dim]`, transposed internally to
`[B, N_seq, Head, N_res, Dim]` so attention contracts over the residue axis):

```python
scale = Dim ** -0.5
S = scale * Q @ K.transpose(-1, -2) + pair_bias + res_mask   # [B, N_seq, Head, N_res, N_res]
P = softmax(S.float(), dim=-1).to(dtype)
O = P @ V                                                     # back to [B, N_seq, N_res, Head, Dim]
```

Backward target:

```python
dV          = P.transpose(-1, -2) @ dO
dP          = dO @ V.transpose(-1, -2)
dS          = P * (dP - rowsum(dP * P))      # softmax jacobian over the key residue axis
dQ          = scale * dS @ K
dK          = scale * dS.transpose(-1, -2) @ Q
d_pair_bias = dS.sum(dim=N_seq, keepdim=True)
```

Inputs / outputs:

```text
do:        [B, N_seq, N_res, Head, Dim]
q, k, v:   [B, N_seq, N_res, Head, Dim]
res_mask:  [B, N_seq, 1, 1, N_res]        additive (0 keep, -1e9 drop), no grad
pair_bias: [B, 1, Head, N_res, N_res]     trainable, float32

dq, dk, dv:  [B, N_seq, N_res, Head, Dim]
d_pair_bias: [B, 1, Head, N_res, N_res]   float32
```

The candidate API is:

```python
def evoattention_backward_triton(do, q, k, v, res_mask, pair_bias):
    return dq, dk, dv, d_pair_bias
```

## Naive Baseline (the seed)

`backward_naive_triton.py` is the OpenEvolve seed: a deliberately simple,
hand-written **single-kernel** Triton backward. Per (batch, msa, head, query-block)
it recomputes the row softmax statistics online (flash forward) to get `O` and the
logsumexp, then makes a second streaming pass that accumulates `dQ` in registers and
**atomic-adds** `dK`, `dV`, and `d_pair_bias`. The atomic-adds + full recompute are
the "naive" part; the evolve target is MegaFold's atomic-free two-pass flash design
(preprocess `D = rowsum(dO * O)`, a dK/dV pass, and a dQ pass). MegaFold's kernel is
the reference design to study and beat.

`backward_naive.py` is the readable PyTorch reference for the same math: it
materializes the full `[N_res, N_res]` probability matrix and applies the textbook
attention-backward formulas. All seeds/backends are verified against PyTorch
autograd on GPU (see `test_correctness.py`).

## Files

- `forward_ref.py`: PyTorch semantic reference for the forward.
- `backward_ref.py`: PyTorch-autograd gradient oracle.
- `backward_naive_triton.py`: verified naive single-kernel Triton backward (the seed).
- `backward_naive.py`: readable materialized PyTorch reference backward.
- `backward_atenir.py`: AtenIR lowering — runs the autograd-extracted ATen backward
  graph through the AtenIR runtime + primitive-Triton dispatch (per-shape cached
  extraction). Reference graph in `atenir/evoattention_bwd_graph.json`.
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
cd openevolve/benchmark/triton_evoattention_backward_bench
python test_correctness.py
python evaluator.py initial_program.py
```

The evaluator checks `dq`, `dk`, `dv`, and `d_pair_bias` separately against
PyTorch autograd. Incorrect candidates are not benchmarked. Tolerances are
attention-appropriate (fp16/bf16 accumulation through two matmuls and a softmax
jacobian); `d_pair_bias` gets extra slack because it reduces over the `N_seq` axis.

## OpenEvolve

```bash
cd openevolve/benchmark/triton_evoattention_backward_bench

openevolve-run \
  initial_program.py \
  evaluator.py \
  --config config.yaml \
  --iterations 10 \
  --output /tmp/openevolve_triton_evoattention_backward_10 \
  --save-best-to evolved_best_program.py
```

## Strong Baseline Check

After an OpenEvolve run:

```bash
python benchmark_strong_baselines.py evolved_best_program.py
```

This reports backward-only latencies for the candidate, PyTorch autograd, the
naive baseline, and MegaFold's fused Triton EvoformerAttention (imported from
`MegaFold/megafold/model/FusedEvoAttention/evoattention.py` when available),
plus candidate speedups vs each.
