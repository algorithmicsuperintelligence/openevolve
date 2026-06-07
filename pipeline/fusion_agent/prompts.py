"""Prompt templates for the AtenIR backward fusion synthesis agent."""

from __future__ import annotations


SYSTEM_MESSAGE = """You are a Triton compiler engineer.
You fuse a correct but fine-grained AtenIR backward graph into a small number of
readable Triton kernels. Correctness is more important than speed."""


COMMON_TRITON_MISTAKES = """## Common Triton mistakes to avoid

- `tl.arange` bounds must be compile-time constants. Do not write
  `tl.arange(0, N)` when `N` is a runtime tensor dimension. Use a
  `BLOCK_*: tl.constexpr` parameter and mask with `offs < N`.
- Do not read Python globals inside `@triton.jit` kernels. Pass `eps`, strides,
  dimensions, and other scalars as kernel arguments.
- For reductions, reduce over exactly the dimensions specified by the AtenIR
  graph metadata. Do not infer reduction axes from output names or local
  variable names.
- Do not hard-code traced dimensions such as 64, 128, or 256 unless they are
  semantically constant in the graph. Derive runtime dimensions from input
  tensor shapes and use masks for tail elements.
- Tile sizes must cover the runtime extent being processed. If a runtime
  dimension can exceed the initial trace size, compute a power-of-two
  `BLOCK_*` from that runtime dimension on the Python launch side and mask
  inactive lanes. Do not process only the first fixed tile of a larger tensor.
- Every `tl.zeros`, `tl.full`, and `tl.arange` shape element must be a
  `tl.constexpr` integer. Never use a runtime scalar/tensor dimension directly
  as a Triton block shape.
- If using `tl.atomic_add`, initialize the destination tensor first and make
  sure the atomic accumulation axis matches the graph's reduction semantics.
- Output dtypes must match the public API contract. If accumulation uses fp32,
  allocate outputs with `torch.empty_like` / `torch.zeros_like` of the requested
  output tensor when possible, or explicitly cast before returning.
- For dynamic shapes, derive shapes/strides at runtime in Python, but pass
  tile sizes as `tl.constexpr` meta-parameters to Triton kernels.
"""


def render_fusion_plan_prompt(
    *,
    forward: str,
    public_api: str,
    graph_summary: str,
    lowering_context: str = "",
) -> str:
    lowering_section = ""
    if lowering_context:
        lowering_section = f"""

Additional per-op lowering context:

```text
{lowering_context}
```

Use this context as implementation evidence for individual AtenIR ops, but still
produce a fused implementation. Do not simply call the per-op graph runner unless
the prompt explicitly asks for a lowering-only baseline.
"""

    return f"""# Backward Fusion Planning

Forward reference:

```text
{forward}
```

Public API to generate:

```python
def {public_api}(dy, x, weight, bias, eps=1e-5):
    return dx, dweight, dbias
```

The AtenIR backward graph below is already verified by primitive composition.
Your task is to propose a semantic-preserving fusion plan that groups nodes into
a small number of Triton kernels. Do not optimize launch parameters yet.
The generated program must support both float32 and float16 inputs; reductions
should accumulate in fp32 and outputs should match the input/parameter dtypes.

{graph_summary}
{lowering_section}

Return Markdown with:

1. Proposed fused kernel groups.
2. Inputs and outputs of each group.
3. Reductions and accumulation dtype.
4. Any intermediate tensors that must be materialized.
"""


def render_codegen_prompt(
    *,
    public_api: str,
    graph_summary: str,
    fusion_plan: str,
    lowering_context: str = "",
) -> str:
    lowering_section = ""
    if lowering_context:
        lowering_section = f"""

Per-op lowering context:

```text
{lowering_context}
```

The context above contains verified per-op implementations from a lowering
agent. Use it to understand local AtenIR semantics and tricky dtype/shape
handling, while still writing fused Triton kernels for the public API.
"""

    return f"""# Backward Fusion Codegen

Generate a complete Python module implementing:

```python
def {public_api}(dy, x, weight, bias, eps=1e-5):
    return dx, dweight, dbias
```

Requirements:

- Use Triton kernels for the fused backward implementation.
- Preserve the semantics of the AtenIR graph.
- Use fp32 accumulation for reductions.
- Support both float32 and float16 input tensors. Do not reject float16 tensors.
- Return `dx` with `x` dtype, `dweight` with `weight` dtype, and `dbias` with `bias` dtype.
- Use output tensors allocated with `torch.empty_like(x)`, `torch.empty_like(weight)`,
  and `torch.empty_like(bias)` so stores cast back to the expected dtype.
- Do not check that inputs are exactly `torch.float32`; only require CUDA floating tensors.
- Include an `EVOLVE-BLOCK` around generated Triton kernels and launch helpers.
- Do not call PyTorch autograd or high-level PyTorch reference operators in the
  generated backward.
- Return only Python source code, no Markdown.

{COMMON_TRITON_MISTAKES}

Fusion plan:

```markdown
{fusion_plan}
```

AtenIR graph summary:

{graph_summary}
{lowering_section}
"""


def render_repair_prompt(
    *,
    public_api: str,
    graph_summary: str,
    fusion_plan: str,
    previous_code: str,
    verifier_report: str,
    lowering_context: str = "",
) -> str:
    lowering_section = ""
    if lowering_context:
        lowering_section = f"""

Per-op lowering context:

```text
{lowering_context}
```
"""

    return f"""# Backward Fusion Repair

The generated fused backward program failed correctness.

Public API:

```python
def {public_api}(dy, x, weight, bias, eps=1e-5):
    return dx, dweight, dbias
```

Verifier report:

```json
{verifier_report}
```

Fusion plan:

```markdown
{fusion_plan}
```

AtenIR graph summary:

{graph_summary}
{lowering_section}

Previous code:

```python
{previous_code}
```

{COMMON_TRITON_MISTAKES}

Before writing the repaired code, classify the failure internally as one of:
compile-time Triton error, dtype/casting error, shape/tiling error, or numerical
formula error. Then fix the root cause in the source code; do not make cosmetic
changes that leave the same formula and tiling structure intact.

Repair only the Python source. Preserve the public API and output order. Return
only Python source code, no Markdown. If the failure is dtype-related, remove
hard-coded float32 checks, use `torch.empty_like` outputs, and keep fp32
accumulation internally instead.
"""
