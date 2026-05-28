"""Prompt templates for the AtenIR backward fusion synthesis agent."""

from __future__ import annotations


SYSTEM_MESSAGE = """You are a Triton compiler engineer.
You fuse a correct but fine-grained AtenIR backward graph into a small number of
readable Triton kernels. Correctness is more important than speed."""


def render_fusion_plan_prompt(
    *,
    forward: str,
    public_api: str,
    graph_summary: str,
) -> str:
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
) -> str:
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
- Do not call PyTorch autograd or PyTorch LayerNorm in the generated backward.
- Return only Python source code, no Markdown.

Fusion plan:

```markdown
{fusion_plan}
```

AtenIR graph summary:

{graph_summary}
"""


def render_repair_prompt(
    *,
    public_api: str,
    graph_summary: str,
    fusion_plan: str,
    previous_code: str,
    verifier_report: str,
) -> str:
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

Previous code:

```python
{previous_code}
```

Repair only the Python source. Preserve the public API and output order. Return
only Python source code, no Markdown. If the failure is dtype-related, remove
hard-coded float32 checks, use `torch.empty_like` outputs, and keep fp32
accumulation internally instead.
"""
