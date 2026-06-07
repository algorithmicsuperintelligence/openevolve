# Backward Formula Derivation Prompt

You are deriving a deliberately naive backward decomposition for a training-oriented
Triton benchmark task. This is Stage 1 of benchmark construction, not optimization.

Operator: `bias_relu`

Forward semantic ground truth:

```text
y = relu(x + bias)
```

Inputs:

- `x`: shape `[rows, cols]`, layout `contiguous`, differentiable
- `bias`: shape `[cols]`, layout `contiguous`, differentiable

Upstream gradient: `dy`

Required gradients:

```text
dx, dbias
```

Known formula hints from metadata:

- `dx`: `dy * ((x + bias) > 0)`
- `dbias`: `sum(dx, dim=0)`

Task:

1. Derive clear mathematical formulas for every required gradient.
2. Identify reductions, broadcasting semantics, saved tensors, and fp32 accumulation requirements.
3. Break the backward into a sequence of simple named intermediate tensors.
4. Prefer an unfused decomposition suitable for a readable multi-kernel Triton baseline.
5. For each intermediate, state its shape, whether it is pointwise or a reduction, and which axis is reduced.
6. Do not write Triton code here.
7. Do not claim the derivation is ground truth; the oracle is PyTorch autograd over `forward_ref.py`.



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


Return only Markdown describing the naive backward decomposition. Do not write code.
