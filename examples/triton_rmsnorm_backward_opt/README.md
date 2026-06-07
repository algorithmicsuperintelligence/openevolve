# Triton RMSNorm Backward-dx Optimization

This example is a Stage 2 optimization target for OpenEvolve. It starts from a
correct but conservative Triton implementation of RMSNorm backward `dx` and
evolves the kernel for better latency.

## Task

Given upstream gradient `dy`, input `x`, and RMSNorm weight `weight`, compute
only `dx`:

```python
inv_rms = torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + eps)
grad_normed = dy.float() * weight.float()
projection = torch.sum(grad_normed * x.float(), dim=-1, keepdim=True)
dx = grad_normed * inv_rms - x.float() * (inv_rms ** 3) * projection / hidden_size
```

The public candidate API is fixed:

```python
def rmsnorm_backward_dx_triton(dy, x, weight, eps=1e-6):
    ...
```

This MVP intentionally excludes `dweight`; adding parameter-gradient reduction
is a good next step after the `dx` optimizer is stable.

## Files

- `initial_program.py`: baseline Triton RMSNorm backward-dx candidate.
- `evaluator.py`: correctness hard gate plus latency benchmark.
- `config.yaml`: OpenEvolve configuration and RMSNorm-specific prompt.

## Environment

Run this on a GPU node with CUDA-visible PyTorch and Triton installed:

```bash
conda activate openev
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
python -m pip install triton
cd /u/wzhan/openevolve/examples/triton_rmsnorm_backward_opt
```

On Delta-like nodes with a CUDA 12.8 driver, avoid the default PyPI PyTorch
wheel if it resolves to `+cu130`; that build requires a newer driver and will
make `torch.cuda.is_available()` return `False`.

If Triton fails while compiling `cuda_utils.c` with a `LIBCTF_1.1` linker error,
make the gcc-toolset runtime libraries visible before running the evaluator:

```bash
export LD_LIBRARY_PATH=/opt/rh/gcc-toolset-13/root/usr/lib64:${LD_LIBRARY_PATH}
```

## Direct Evaluator Check

Before running evolution, validate the baseline:

```bash
python evaluator.py initial_program.py
```

A correct baseline should report:

```text
"correct": 1.0
"partial_correctness": 1.0
"speedup": around 1.0
```

## OpenEvolve Run

For OpenAI-compatible OpenAI models:

```bash
openevolve-run \
  initial_program.py \
  evaluator.py \
  --config config.yaml \
  --api-base https://api.openai.com/v1 \
  --primary-model gpt-5.4 \
  --secondary-model gpt-5.4 \
  --iterations 10 \
  --output /tmp/openevolve_triton_rmsnorm_backward_10
```

The config disables OpenEvolve cascade evaluation so each candidate receives the
full correctness-gated benchmark score. That makes `combined_score` equal to
aggregate speedup for correct candidates.

The config also sets `max_tasks_per_child: 1`. On Python 3.10 this makes
OpenEvolve use `spawn` workers instead of forked workers, which is required for
CUDA evaluators.

## Scoring

Correctness is a hard gate:

```python
if import_or_compile_error:
    combined_score = -1e9
elif not correct:
    combined_score = -1e6 + partial_correctness
else:
    combined_score = speedup
```

The expected success signal is a best-so-far correct candidate with
`speedup > 1.0` relative to the baseline.
