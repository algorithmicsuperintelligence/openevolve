# Naive Triton Backward Generation Prompt

You are writing a naive, readable Triton backward baseline for `bias_relu`.

Forward semantic ground truth:

```text
y = relu(x + bias)
```

Inputs:

- `x`: shape `[rows, cols]`, layout `contiguous`, differentiable
- `bias`: shape `[cols]`, layout `contiguous`, differentiable

Required outputs:

```text
dx, dbias
```

Formula hints:

- `dx`: `dy * ((x + bias) > 0)`
- `dbias`: `sum(dx, dim=0)`

Correctness requirements:

- Match `backward_ref.py`, which uses PyTorch autograd over `forward_ref.py`.
- Prioritize clarity and faithful decomposition over speed.
- Use simple pointwise and reduction kernels.
- Deliberately avoid fusion optimizations.
- Materialize named intermediates from the decomposition when reasonable.
- Use fp32 accumulation for reductions.
- Support the correctness workloads in `stage1_spec.yaml`.
- Return gradients in the order listed above.
- Do not add argument-order compatibility hacks. Use exactly the public API
  requested by the synthesis prompt.
- Do not special-case only benchmark shapes.

Do not use Liger or expert fused code as ground truth. Liger may be used later
only as an expert performance baseline.


Public API requirement:

```python
def bias_relu_backward_naive_triton(
    dy: torch.Tensor,
    x: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

The verifier calls this exact function as:

```python
inputs = task_spec.make_inputs(torch, case)
actual = bias_relu_backward_naive_triton(inputs[0], inputs[1], inputs[2])
```

You must implement exactly the signature shown above.
Do not use `*args`, `**kwargs`, argument-order inference, or compatibility
branches for alternate argument orders.

Naive decomposition to implement:

```markdown
{NAIVE_DECOMPOSITION_FROM_FORMULA_STAGE}
```

Implementation policy:

- Implement the decomposition above directly.
- Prefer one small Triton kernel per named intermediate/reduction group.
- Do not fuse all row-wise work into a single optimized kernel.
- Materialize intermediate tensors where the decomposition names them.
- Keep comments short and focused on the decomposition.


Stage 1 normalized spec:

```yaml
stage: stage1_forward_to_backward_construction
task_name: triton_bias_relu_backward_bench
operator: bias_relu
difficulty:
  level: 2
  label: broadcasting_backward
description: Minimal forward-to-backward Triton benchmark task for y = relu(x + bias).
  The backward implementation must return both dx and dbias.
semantic_source:
  forward_ref: forward_ref.py
  forward_formula: y = relu(x + bias)
inputs:
- name: x
  shape: '[rows, cols]'
  layout: contiguous
  differentiable: true
- name: bias
  shape: '[cols]'
  layout: contiguous
  differentiable: true
output:
  name: y
  shape: '[rows, cols]'
backward_target:
  upstream_gradient: dy
  required_gradients:
  - dx
  - dbias
  saved_tensors:
  - x
  - bias
  formula_hints:
    dx: dy * ((x + bias) > 0)
    dbias: sum(dx, dim=0)
oracle_policy:
  ground_truth: PyTorch autograd over forward_ref.py
  llm_generated_code_is_not_ground_truth: true
workloads:
  correctness:
    shapes:
    - - 17
      - 31
    - - 128
      - 256
    - - 129
      - 257
    - - 512
      - 1024
    dtypes:
    - float32
    - float16
  benchmark:
    shapes:
    - - 512
      - 1024
    - - 1024
      - 1024
    - - 1024
      - 2048
    dtypes:
    - float16
tolerances:
  float32:
    atol: 1.0e-05
    rtol: 1.0e-05
  float16:
    atol: 0.02
    rtol: 0.02
baselines:
  pytorch_autograd: backward_ref.py
  naive_triton_backward: backward_naive_triton.py
  expert_triton_backward: null
metrics:
- compile_success
- runtime_success
- backward_correctness
- dx_correctness
- dbias_correctness
- autograd_integration_success
- backward_latency_ms
- speedup_vs_naive_triton_baseline
construction_outputs:
  autograd_oracle: backward_ref.py
  naive_triton_candidate: backward_naive_triton.py
  prompts:
  - prompts/backward_formula_prompt.md
  - prompts/naive_triton_backward_prompt.md
  - prompts/repair_prompt_template.md

```

forward_ref.py:

```python
"""PyTorch reference for the bias + ReLU forward operator."""

import torch


def bias_relu_forward_ref(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Compute y = relu(x + bias) for x shaped [rows, cols] and bias [cols]."""
    if x.ndim != 2:
        raise ValueError(f"expected x to be 2D, got shape {tuple(x.shape)}")
    if bias.ndim != 1:
        raise ValueError(f"expected bias to be 1D, got shape {tuple(bias.shape)}")
    if x.shape[1] != bias.shape[0]:
        raise ValueError(f"bias length {bias.shape[0]} must match x cols {x.shape[1]}")
    return torch.relu(x + bias)

```

forward_triton.py, provided only as layout/indexing context:

```python
"""Simple Triton forward kernel for y = relu(x + bias)."""

import torch
import triton
import triton.language as tl


@triton.jit
def _bias_relu_forward_kernel(
    x_ptr,
    bias_ptr,
    y_ptr,
    total: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    cols = offsets % n_cols

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = tl.maximum(x + bias, 0.0)

    tl.store(y_ptr + offsets, y, mask=mask)


def bias_relu_forward_triton(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Compute y = relu(x + bias) for contiguous 2D CUDA tensors."""
    if x.ndim != 2:
        raise ValueError(f"expected x to be 2D, got shape {tuple(x.shape)}")
    if bias.ndim != 1:
        raise ValueError(f"expected bias to be 1D, got shape {tuple(bias.shape)}")
    if x.shape[1] != bias.shape[0]:
        raise ValueError(f"bias length {bias.shape[0]} must match x cols {x.shape[1]}")
    if not x.is_cuda or not bias.is_cuda:
        raise ValueError("bias_relu_forward_triton requires CUDA tensors")

    x = x.contiguous()
    bias = bias.contiguous()
    y = torch.empty_like(x)

    total = x.numel()
    block_size = 256
    grid = (triton.cdiv(total, block_size),)
    _bias_relu_forward_kernel[grid](
        x,
        bias,
        y,
        total,
        x.shape[1],
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return y

```


Return only the complete Python source file. Do not include Markdown prose.
