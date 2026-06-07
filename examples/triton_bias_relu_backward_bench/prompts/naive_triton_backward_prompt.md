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

Canonical naive kernel plan:

Use this exact canonical naive kernel plan:

1. `_bias_relu_backward_dx_kernel`
   - inputs: `dy`, `x`, `bias`
   - outputs: `dx = dy * ((x + bias) > 0)`
   - pointwise over `[rows, cols]`
2. `_bias_relu_backward_dbias_kernel`
   - inputs: `dx`
   - outputs: `dbias = sum(dx, dim=0)`
   - column-wise reduction over rows

Forbidden fusions for this baseline:

- Do not compute `dbias` inside the `dx` kernel.
- Do not combine the pointwise and reduction kernels.

Correctness requirements:

- Match `backward_ref.py`, which uses PyTorch autograd over `forward_ref.py`.
- Prioritize clarity and faithful decomposition over speed.
- Use simple pointwise and reduction kernels.
- Deliberately avoid fusion optimizations.
- Follow the canonical naive kernel plan exactly when one is provided.
- Materialize named intermediates from the decomposition when reasonable.
- Use fp32 accumulation for reductions.
- Support the correctness workloads in `stage1_spec.yaml`.
- Return gradients in the order listed above.
- Do not add argument-order compatibility hacks. Use exactly the public API
  requested by the synthesis prompt.
- Do not special-case only benchmark shapes.

Do not use Liger or expert fused code as ground truth. Liger may be used later
only as an expert performance baseline.
