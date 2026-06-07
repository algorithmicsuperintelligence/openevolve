# Triton Bias + ReLU Backward Benchmark Sample

This is a minimal forward-to-backward Triton benchmark task for OpenEvolve.
It is intentionally small: the goal is to show the task structure for training
operators, not to collect another forward-kernel benchmark.

## Operator

Forward:

```python
y = torch.relu(x + bias)
```

Backward:

```python
dx = dy * ((x + bias) > 0)
dbias = torch.sum(dx, dim=0)
```

`x` and `dy` are 2D tensors shaped `[rows, cols]`; `bias` is a 1D tensor
shaped `[cols]`. This makes the sample a Level 2 broadcasting-backward task.

## Inputs And Outputs

This example has three related interfaces.

The mathematical forward operator takes `x` and `bias` as input and produces
`y`:

```python
x:    [rows, cols]
bias: [cols]
y = relu(x + bias): [rows, cols]
```

The backward problem takes the upstream gradient `dy` plus the saved forward
inputs `x` and `bias`, then produces gradients for the differentiable inputs:

```python
dy:    [rows, cols]
x:     [rows, cols]  # saved from forward
bias:  [cols]        # saved from forward

dx:    [rows, cols]
dbias: [cols]
```

The OpenEvolve candidate API is therefore:

```python
def bias_relu_backward_triton(dy, x, bias) -> tuple[torch.Tensor, torch.Tensor]:
    return dx, dbias
```

`forward_triton.py` is not the target output of this benchmark. It is included
as task context: in a real forward-to-backward benchmark, the agent may receive
an existing Triton forward kernel and use it to understand indexing, layout,
broadcasting, and saved tensors. The thing being evaluated is whether the agent
can implement the corresponding Triton backward kernel and autograd integration.

## Files

Public task context:

- `forward_ref.py`: PyTorch forward semantics.
- `forward_triton.py`: simple Triton forward kernel used as context, not as
  the candidate output.
- `meta.yaml`: task metadata, workloads, dtypes, tolerances, and required
  gradients.
- `stage1_spec.yaml`: normalized Stage 1 construction spec generated from
  `meta.yaml`.
- `prompts/`: dry-run prompt artifacts for formula derivation, naive Triton
  backward generation, repair feedback, and autograd oracle guidance.
- `initial_program.py`: OpenEvolve seed candidate with an EVOLVE-BLOCK. The
  seed is a naive decomposed backward implementation, useful as a lower-bound
  optimization starting point.
- `config.yaml`: OpenEvolve configuration and task prompt.

Hidden oracle and evaluation files:

- `backward_ref.py`: PyTorch autograd oracle for `dx` and `dbias`.
- `test_correctness.py`: randomized gradient correctness checks.
- `benchmark.py`: backward latency benchmark.
- `evaluator.py`: thin wrapper around the shared evaluator core.
- `task_spec.py`: operator-specific input generation, oracle, output names, and
  tolerances used by the shared evaluator.

Verified baseline:

- `backward_naive_triton.py`: manually verified naive decomposed Triton backward
  baseline. It uses one pointwise kernel for `dx` and one column-wise reduction
  kernel for `dbias`.

## Stage 1 Construction Artifacts

This directory now records the forward-to-backward construction path before
OpenEvolve optimization:

```bash
cd /u/wzhan/openevolve
python -m benchmark.triton_backward_bench_common.builder_cli emit-spec \
  examples/triton_bias_relu_backward_bench
python -m benchmark.triton_backward_bench_common.builder_cli emit-prompts \
  examples/triton_bias_relu_backward_bench
python -m benchmark.triton_backward_bench_common.builder_cli check-oracle \
  examples/triton_bias_relu_backward_bench
python -m benchmark.triton_backward_bench_common.builder_cli verify \
  examples/triton_bias_relu_backward_bench
```

Stage 1 treats `forward_ref.py` as the semantic ground truth and `backward_ref.py`
as the PyTorch-autograd oracle. The files in `prompts/` are dry-run artifacts for
future API-driven formula derivation, naive Triton generation, and repair loops;
they are not ground truth. Stage 2 then uses `initial_program.py` and
`evaluator.py` for OpenEvolve optimization.

To run the API-driven Stage 1 synthesis loop on a GPU node:

```bash
cd /u/wzhan/openevolve
python -m benchmark.triton_backward_bench_common.builder_cli synthesize \
  examples/triton_bias_relu_backward_bench \
  --api-base https://api.openai.com/v1 \
  --model gpt-5.5 \
  --max-attempts 3 \
  --output-dir stage1_candidates
```

The loop writes attempts under `stage1_candidates/attempt_XXX/`, verifies each
candidate against PyTorch autograd, and writes repair prompts when verification
fails. A passing candidate is copied to `stage1_candidates/best/`. This command
does not overwrite the manually verified `backward_naive_triton.py`; promotion to
the official baseline should remain a manual review step.

## Direct Checks

Run on a GPU node with CUDA-visible PyTorch and Triton:

```bash
cd /u/wzhan/openevolve/examples/triton_bias_relu_backward_bench
python test_correctness.py
python benchmark.py
python evaluator.py initial_program.py
```

The correctness test compares the Triton forward, naive Triton backward, and
custom autograd wrapper against PyTorch reference behavior. The evaluator uses
the shared core in `benchmark/triton_backward_bench_common/`, reports
per-gradient correctness for `dx` and `dbias`, and does not benchmark incorrect
candidates.

## OpenEvolve Run

```bash
openevolve-run \
  initial_program.py \
  evaluator.py \
  --config config.yaml \
  --iterations 10 \
  --output /tmp/openevolve_triton_bias_relu_backward_10 \
  --save-best-to evolved_best_program.py
```

This sample is meant to seed a broader benchmark suite where agents receive a
PyTorch forward reference and/or Triton forward kernel, then synthesize the
corresponding autograd-compatible Triton backward implementation.
