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

Canonical naive decomposition policy:

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

Task:

1. Derive clear mathematical formulas for every required gradient.
2. Identify reductions, broadcasting semantics, saved tensors, and fp32 accumulation requirements.
3. Break the backward into a sequence of simple named intermediate tensors.
4. Prefer an unfused decomposition suitable for a readable multi-kernel Triton baseline.
5. For each intermediate, state its shape, whether it is pointwise or a reduction, and which axis is reduced.
6. Do not write Triton code here.
7. Do not claim the derivation is ground truth; the oracle is PyTorch autograd over `forward_ref.py`.
