# Triton Softmax Backward Optimization

This example is a Stage 2 MVP for Triton backward optimization with OpenEvolve.
It starts from a correct but conservative Triton implementation of row-wise
softmax backward and evolves the kernel for better latency.

## Task

Given softmax output `y` and upstream gradient `dy`, compute:

```python
dx = y * (dy - torch.sum(dy * y, dim=-1, keepdim=True))
```

The public candidate API is fixed:

```python
def softmax_backward_triton(dy, y):
    ...
```

OpenEvolve is only expected to edit the `EVOLVE-BLOCK` in `initial_program.py`,
which contains the Triton kernel and launch-parameter selection.

## Files

- `initial_program.py`: baseline Triton softmax backward candidate.
- `evaluator.py`: correctness hard gate plus latency benchmark.
- `config.yaml`: OpenEvolve configuration and Triton-specific prompt.

## Environment

Run this on a GPU node with CUDA-visible PyTorch and Triton installed:

```bash
conda activate openev
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
python -m pip install triton
cd /u/wzhan/openevolve/examples/triton_softmax_backward_opt
```

The evaluator intentionally fails with a large negative score if CUDA is not
available. Login nodes are fine for editing code, but not for running the
Triton correctness or benchmark path.

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

The exact latency depends on GPU model and load.

## OpenEvolve Run

For OpenAI-compatible OpenAI models:

```bash
openevolve-run \
  initial_program.py \
  evaluator.py \
  --config config.yaml \
  --api-base https://api.openai.com/v1 \
  --primary-model gpt-4o-mini \
  --secondary-model gpt-4o-mini \
  --iterations 10 \
  --output /tmp/openevolve_triton_softmax_backward_10
```

The config disables OpenEvolve cascade evaluation so each candidate receives the
full correctness-gated benchmark score. That makes `combined_score` equal to
aggregate speedup for correct candidates.

The config also sets `max_tasks_per_child: 1`. On Python 3.10 this makes
OpenEvolve use `spawn` workers instead of forked workers, which is required for
CUDA evaluators. Without this, candidate evaluations can fail with
`Cannot re-initialize CUDA in forked subprocess`.

For the config default Gemini-compatible endpoint, set `OPENAI_API_KEY` to the
Gemini key and omit the model overrides if desired.

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

This means an incorrect candidate should never be selected over a correct one.
Structured feedback is emitted in artifacts, including shape, dtype, max
absolute error, max relative error, and benchmark latency.

## Interpretation

A successful run does not require every candidate to improve. The expected
signal is that the best-so-far correct candidate has `speedup > 1.0` relative
to the baseline. Results are saved under the selected output directory, with
the final best candidate in `best/best_program.py`.
