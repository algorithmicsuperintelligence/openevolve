# AtenIR Unified Correctness Tester

This directory contains the correctness tester for AtenIR-backed backward kernel
workflows. The tester compares generated or composed backward implementations
against PyTorch autograd, per gradient.

## Files

- `run_correctness.py`: CLI for checking a backward implementation against a
  PyTorch forward reference.
- `harness.py`: shared test utilities and backend adapters.
- `test_unified_correctness.py`: pytest smoke tests for the generic AtenIR path.

## What This Builds On

Haochen's original AtenIR tests validate this round-trip:

```text
PyTorch forward
-> extract_autograd
-> AtenIR graph
-> make_registry
-> run_graph
-> compare with PyTorch autograd
```

This tester keeps that path, but adds a more general interface for testing
different backward backends, shape suites, and dtypes.

## Added Coverage

Compared with the original AtenIR tests, this adds:

- A unified CLI for correctness checks.
- Support for both generic AtenIR compose and explicit backward functions.
- Static, dynamic, and non-tile-aligned shape suites.
- dtype-specific testing, including fp32 and fp16.
- Per-gradient correctness reports.
- Structured JSON output.
- Fallback reporting for the AtenIR compose path.
- A reusable harness for future fused or optimized kernels.

## Backend Modes

### AtenIR Compose Backend

This mode tests the generic AtenIR execution path:

```text
forward function
-> extract_autograd
-> make_registry
-> run_graph
-> compare with PyTorch autograd
```

Example:

```bash
python -m tests.atenir_correctness.run_correctness \
  --backend atenir_compose \
  --forward atenir._examples:layernorm \
  --mode dynamic \
  --dtype float32 \
  --atol 2e-5 \
  --rtol 2e-5
```

### Callable Backward Backend

This mode tests an explicit backward implementation:

```text
forward reference
-> PyTorch autograd oracle
-> backward candidate
-> compare all gradients
```

Example:

```bash
python -m tests.atenir_correctness.run_correctness \
  --forward benchmark.triton_layernorm_backward_bench.forward_ref:layernorm_forward_ref \
  --backward benchmark.triton_layernorm_backward_bench.backward_naive_triton:layernorm_backward_naive_triton \
  --mode dynamic \
  --scalar 1e-5 \
  --dtype float32 \
  --atol 2e-5 \
  --rtol 2e-5
```

Generated files can be checked by path:

```bash
python -m tests.atenir_correctness.run_correctness \
  --forward atenir._examples:layernorm \
  --backward-file /path/to/generated_program.py \
  --backward-fn layernorm_backward_triton \
  --mode dynamic \
  --dtype float32 \
  --dtype float16
```

## Shape Modes

If explicit `--shape` arguments are not provided, the runner infers a preset
from `--forward`, `--backward`, or `--task-spec`.

- `static`: small smoke-test shapes.
- `dynamic`: multiple realistic shapes for shape generalization.
- `nontile`: non-power-of-two or non-tile-aligned shapes.

For LayerNorm, `dynamic` currently covers:

```text
(B, H) = (1, 768)
(B, H) = (8, 1024)
(B, H) = (32, 1536)
(B, H) = (8, 4096)
(B, H) = (1, 8192)
```

`nontile` is useful for catching kernels that assume power-of-two hidden sizes.

## Dtype Testing

Run one dtype:

```bash
--dtype float32
```

or multiple dtypes:

```bash
--dtype float32 --dtype float16
```

For fp32, typical LayerNorm tolerances are:

```bash
--atol 2e-5 --rtol 2e-5
```

fp16 generally needs looser tolerances depending on the op and accumulation
strategy.

## Output

The runner prints structured JSON:

```json
{
  "passed": true,
  "passed_cases": 5,
  "failed_cases": 0,
  "total_cases": 5,
  "reports": []
}
```

Each gradient is checked separately. For LayerNorm, this means:

```text
dx
dweight
dbias
```

## Current Coverage

The unified tester currently covers:

- Generic AtenIR compose examples:
  - `square_sum`
  - `rmsnorm`
  - `attention_block`
  - `topk_gather`
  - `layernorm`
- Explicit backward callables:
  - manually written Triton backward kernels
  - AtenIR-backed backward adapters
  - generated or optimized backward kernels

## Note On `compose.py`

`compose.py` is not the final optimized implementation. It is an executable
correctness baseline for AtenIR: it runs the extracted graph node-by-node using
the primitive registry and checks that the graph semantics match PyTorch
autograd before later lowering, fusion, or optimization work.
