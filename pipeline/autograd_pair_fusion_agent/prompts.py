"""Prompt templates for autograd-pair saved-tensor fusion synthesis."""

from __future__ import annotations


SYSTEM_MESSAGE = """You are a Triton compiler engineer.
You synthesize a forward/backward autograd pair.  The forward may save tensors
that the backward reuses.  Correctness is required; efficiency should balance
backward latency, forward+backward latency, and saved-tensor memory."""


PAIR_RULES = """## Autograd-pair rules

Public API to implement:

```python
def layernorm_forward_with_saved(x, weight, bias, eps=1e-5):
    return y, saved_tensors

def layernorm_backward_from_saved(dy, saved_tensors, eps=1e-5):
    return dx, dweight, dbias
```

Hard constraints:
- Return only Python source, no Markdown.
- Include imports for `torch`, `triton`, and `triton.language as tl`.
- Include an `EVOLVE-BLOCK` around generated Triton kernels and launch helpers.
- Do not call PyTorch autograd or PyTorch reference LayerNorm in the generated math.
- Forward must produce the same `y` as row-wise LayerNorm over the last dimension.
- Backward must consume only `dy`, `saved_tensors`, and `eps`.
- `saved_tensors` must be a tensor or tuple/list of tensors, because the evaluator
  stores them via `ctx.save_for_backward`.
- Return `dx` with `x` dtype, `dweight` with `weight` dtype, and `dbias` with
  `bias` dtype.

Saved-tensor guidance:
- The saved tensor tuple is part of the evolvable program state.  The initial
  seed may save only original inputs; OpenEvolve may add, remove, or reorder
  saved tensors as long as the forward and backward agree.
- You may save forward intermediates if doing so improves the forward+backward
  tradeoff, but the prompt does not prescribe which intermediates to save.
- Prefer compact saved state such as small per-row/per-block statistics over
  full activation-sized tensors when the backward can cheaply reconstruct the
  larger intermediate.
- Avoid saving tensors with the same shape as a large activation unless the
  latency benefit clearly outweighs the memory cost.
- Do not save excessive large intermediates unless they clearly improve
  forward+backward latency.  The evaluator reports saved memory.
- It is acceptable to save original inputs such as `x`, `weight`, and `bias` if
  the backward needs them.

Triton pitfalls:
- `tl.arange` bounds must be compile-time constants. Use `BLOCK_*: tl.constexpr`.
- Do not read Python globals inside `@triton.jit`; pass eps, dimensions, and
  strides as arguments or meta-parameters.
- Use fp32 accumulation for reductions.
- Avoid global atomic contention when a partial-buffer reduction is better.
"""


def render_plan_prompt(*, forward: str, graph_summary: str, lowering_context: str = "") -> str:
    lowering_section = ""
    if lowering_context:
        lowering_section = f"""

Additional lowering context:

```text
{lowering_context}
```
"""

    return f"""# Autograd-Pair Planning

Forward reference:

```text
{forward}
```

The AtenIR backward graph below describes the reference backward semantics, but
the generated implementation is allowed to change the forward/backward contract
by saving forward intermediates.

{graph_summary}
{lowering_section}

Return Markdown with:

1. Initial saved tensor contract and which parts should remain evolvable.
2. Triton kernels for forward and backward.
3. Backward formula and reduction strategy.
4. Expected memory overhead of saved tensors and why it is worth the latency tradeoff.
"""


def render_codegen_prompt(*, graph_summary: str, pair_plan: str, lowering_context: str = "") -> str:
    lowering_section = ""
    if lowering_context:
        lowering_section = f"""

Additional lowering context:

```text
{lowering_context}
```
"""

    return f"""# Autograd-Pair Codegen

Generate a complete Python module for an autograd pair for the provided forward reference.

{PAIR_RULES}

## Plan

```markdown
{pair_plan}
```

## AtenIR backward graph summary

{graph_summary}
{lowering_section}
"""


def render_repair_prompt(
    *,
    graph_summary: str,
    pair_plan: str,
    previous_code: str,
    verifier_report: str,
    lowering_context: str = "",
) -> str:
    lowering_section = ""
    if lowering_context:
        lowering_section = f"""

Additional lowering context:

```text
{lowering_context}
```
"""

    return f"""# Autograd-Pair Repair

The generated autograd-pair program failed verification.

Verifier report:

```json
{verifier_report}
```

{PAIR_RULES}

## Plan

```markdown
{pair_plan}
```

## AtenIR backward graph summary

{graph_summary}
{lowering_section}

## Previous code

```python
{previous_code}
```

Return only the repaired Python source. No Markdown.
"""
