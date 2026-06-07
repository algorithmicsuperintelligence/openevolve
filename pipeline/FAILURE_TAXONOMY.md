# AtenIR Agent Failure Taxonomy

This document records recurring failure modes from AtenIR lowering/fusion agent
runs. The goal is to turn failed verification reports into prompt guidance so a
lower-cost model can repair like a stronger model.

## Triton Compile-Time Failures

### `triton_constexpr_arange`

Symptom:

```text
arange's arguments must be of type tl.constexpr
```

Root cause:

The generated kernel used a runtime dimension as a `tl.arange` bound, for
example `tl.arange(0, N)`.

Repair guidance:

Use a meta-parameter such as `BLOCK_N: tl.constexpr`, compute it on the Python
launch side with `triton.next_power_of_2(N)`, then mask with `cols < N`.

### `triton_global_scalar_in_jit`

Symptom:

```text
Cannot access global variable eps from within @jit'ed function
```

Root cause:

The generated `@triton.jit` function closed over Python globals instead of
receiving scalars as kernel parameters.

Repair guidance:

Pass `eps`, dimensions, strides, and other scalar values explicitly to the
kernel launch.

### `triton_constexpr_block_shape`

Symptom:

```text
Shape element 0 must have type `constexpr[int]`
```

Root cause:

The generated kernel used a runtime dimension as a Triton block shape, for
example `tl.zeros((C,), ...)` where `C` is a runtime scalar.

Repair guidance:

Pass a compile-time tile size such as `BLOCK_C: tl.constexpr`, create tensors
with `tl.zeros((BLOCK_C,), ...)`, and mask lanes with `cols < C`.

### `dtype_mismatch`

Symptom:

```text
Float did not match Half
```

Root cause:

The generated program accumulates in fp32 but returns fp32 tensors for outputs
whose public API expects the input/output dtype.

Repair guidance:

Allocate output tensors with `torch.empty_like` / `torch.zeros_like` for the
expected output dtype, or explicitly cast outputs before returning.

### `trace_shape_constant_leak`

Symptom:

```text
shape '[64]' is invalid for input of size 128
```

Root cause:

The generated kernel or wrapper preserved a traced shape constant and reused it
for dynamic inputs.

Repair guidance:

Derive reshape/view sizes from runtime tensor shapes or graph scalar overrides.
Do not bake example-input dimensions into reusable seed programs.

## LayerNorm Semantic Failures

### `layernorm_hidden_dim_not_fully_reduced`

Symptom:

The program compiles, but all outputs have large numerical errors. Generated
code often contains fixed `BLOCK_SIZE = 64` while tests use hidden sizes such as
768, 1024, 1536, 4096, or 8192.

Root cause:

The row-wise mean/variance reduction covers only a tile instead of the full
hidden dimension.

Repair guidance:

For the common LayerNorm benchmark shapes, one program per row should reduce
across all hidden columns using `BLOCK_N = triton.next_power_of_2(N)` and
`cols = tl.arange(0, BLOCK_N)`.

### `layernorm_row_vs_column_reduction_confusion`

Symptom:

`dx`, `dweight`, and `dbias` all have large numerical errors. Generated code may
use `dbias / N` or `dweight / N` inside the `dx` formula.

Root cause:

The model confused row-wise reductions over hidden columns with column-wise
parameter gradients over rows. `dweight` and `dbias` are not substitutes for
`mean(xhat * one)` and `mean(one)`.

Repair guidance:

Use:

```text
one = dy * weight
dx = (one - mean(one) - xhat * mean(xhat * one)) * rstd
dweight = sum(dy * xhat, dim=0)
dbias = sum(dy, dim=0)
```

The means in the `dx` formula are row-wise hidden-dimension means. `dweight` and
`dbias` are column-wise sums across rows.

## Process Guidance

- First classify a failure as compile-time, dtype/casting, shape/tiling, or
  numerical formula error.
- Fix compile-time failures before changing math.
- If attempts repeat the same numerical errors, force a structural rewrite
  instead of small local edits.
- Add new recurring failures here and mirror high-confidence fixes in the agent
  prompt's "Common Triton mistakes" section.
