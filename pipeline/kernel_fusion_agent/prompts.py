"""Prompt templates for fusing verified per-op Triton kernels."""

from __future__ import annotations


SYSTEM_MESSAGE = """You are a Triton compiler engineer specializing in kernel fusion.
You are given a verified AtenIR graph and verified per-op Triton kernels. Your
job is to synthesize a small number of fused Triton kernels for the public API.
Correctness is required, but the seed must also be suitable for OpenEvolve
performance optimization."""


KERNEL_FUSION_RULES = """## Kernel-aware fusion rules

The per-op kernels are semantic evidence, not an implementation template.

Hard constraints:
- Do NOT call the per-op graph runner.
- Do NOT stitch the per-op kernels together as many launches.
- Do NOT call PyTorch core tensor ops for the backward math: no `torch.mean`,
  `torch.var`, `torch.sum`, `torch.sqrt`, `torch.rsqrt`, `torch.nn`, or autograd
  inside the generated public API.
- Do NOT use Python loops over rows/columns to compute tensor values.
- Implement the backward math with Triton kernels inside the EVOLVE-BLOCK.

Performance guidance:
- Fuse adjacent elementwise/reduction/scalar ops when their data dependencies
  allow it, but preserve the exact graph semantics.
- Prefer computing short-lived intermediates inside fused kernels instead of
  materializing them as global-memory tensors.
- Avoid designs that perform one global atomic update per input element when a
  block-wise partial reduction followed by a second reduction would express the
  same graph semantics with less contention.
- For small correctness shapes, correctness is more important than launch count;
  for benchmark shapes, avoid PyTorch fallback and excessive atomics.

Common Triton pitfalls:
- `tl.arange` bounds must be compile-time constants. Use `BLOCK_N:
  tl.constexpr`, compute it in Python with `triton.next_power_of_2(N)`, and mask
  with `cols < N`.
- Do not read Python globals inside `@triton.jit`; pass `eps`, dimensions, and
  strides as kernel arguments.
- Reductions must cover exactly the graph-specified axes and preserve keepdim /
  reshape / view semantics. Do not substitute a different reduction merely
  because output names look similar.
- Do not hard-code traced dimensions such as 64, 128, or 256 unless they are
  semantically constant in the graph.
- Tile sizes must cover the runtime extent being processed. If a runtime
  dimension can exceed the initial trace size, compute a power-of-two
  `BLOCK_*` from that runtime dimension on the Python launch side and mask
  inactive lanes. Do not process only the first fixed tile of a larger tensor.
- Every `tl.zeros`, `tl.full`, and `tl.arange` shape element must be a
  `tl.constexpr` integer. Never use a runtime scalar/tensor dimension directly
  as a Triton block shape.
- Output dtypes must match the public API contract. If accumulation uses fp32,
  allocate outputs with `torch.empty_like` / `torch.zeros_like` of the requested
  output tensor when possible, or explicitly cast before returning.
"""


def render_kernel_fusion_plan_prompt(
    *,
    forward: str,
    public_api: str,
    graph_summary: str,
    lowering_context: str,
) -> str:
    return f"""# Kernel-Aware Backward Fusion Planning

Forward reference:

```text
{forward}
```

Public API to generate:

```python
def {public_api}(dy, x, weight, bias, eps=1e-5):
    return dx, dweight, dbias
```

The AtenIR graph below is verified, and the per-op Triton kernels below are
also verified for the extracted graph. Use them to recover precise semantics
and dtype/shape handling, then design a fused implementation.

{KERNEL_FUSION_RULES}

## AtenIR graph summary

{graph_summary}

## Verified per-op lowering context

```text
{lowering_context}
```

Return Markdown with:

1. Fused Triton kernel groups.
2. Inputs/outputs for each fused kernel.
3. Which intermediates are recomputed versus materialized.
4. Reduction strategy for every graph output and required intermediate.
5. Explicitly state how the design avoids PyTorch fallback and excessive global
   atomic contention.
"""


def render_kernel_fusion_codegen_prompt(
    *,
    public_api: str,
    graph_summary: str,
    fusion_plan: str,
    lowering_context: str,
) -> str:
    return f"""# Kernel-Aware Backward Fusion Codegen

Generate a complete Python module implementing:

```python
def {public_api}(dy, x, weight, bias, eps=1e-5):
    return dx, dweight, dbias
```

Requirements:
- Return only Python source, no Markdown.
- Include imports for `torch`, `triton`, and `triton.language as tl`.
- Include an `EVOLVE-BLOCK` around generated Triton kernels and launch helpers.
- Support float32 and float16 CUDA tensors.
- Return `dx` with `x` dtype, `dweight` with `weight` dtype, and `dbias` with
  `bias` dtype.
- Preserve the public API and output order.
- Use the verified per-op kernels only as semantic references. Do not call them.

{KERNEL_FUSION_RULES}

## Fusion plan

```markdown
{fusion_plan}
```

## AtenIR graph summary

{graph_summary}

## Verified per-op lowering context

```text
{lowering_context}
```
"""


def render_kernel_fusion_repair_prompt(
    *,
    public_api: str,
    graph_summary: str,
    fusion_plan: str,
    previous_code: str,
    verifier_report: str,
    lowering_context: str,
) -> str:
    return f"""# Kernel-Aware Backward Fusion Repair

The fused Triton program failed verification.

Public API:

```python
def {public_api}(dy, x, weight, bias, eps=1e-5):
    return dx, dweight, dbias
```

Verifier report:

```json
{verifier_report}
```

Before editing, classify the root cause internally:
- Triton compile-time error
- dtype/casting error
- dynamic shape/tiling error
- graph semantic / formula error
- performance-antipattern correctness fallback

Then rewrite the source to fix the root cause. If the previous code uses
PyTorch tensor ops for core backward math or excessive global atomics for graph
reductions, replace that structure instead of making a local patch.

{KERNEL_FUSION_RULES}

## Fusion plan

```markdown
{fusion_plan}
```

## AtenIR graph summary

{graph_summary}

## Verified per-op lowering context

```text
{lowering_context}
```

## Previous code

```python
{previous_code}
```

Return only the repaired Python source. No Markdown.
"""
