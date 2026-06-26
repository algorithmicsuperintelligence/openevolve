"""Prompt templates for autograd-pair saved-tensor fusion synthesis."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OperatorSpec:
    """Everything that varies between operators in the autograd-pair pipeline.

    Pass an instance to all render_* functions so the prompts describe the
    right function names, tensor shapes, and semantics for the target benchmark.
    """

    # Public function signatures the LLM must implement.
    forward_fn_name: str       # e.g. "layernorm_forward_with_saved"
    forward_args: str          # e.g. "x, weight, bias, eps=1e-5"
    backward_fn_name: str      # e.g. "layernorm_backward_from_saved"
    backward_args: str         # e.g. "dy, saved_tensors, eps=1e-5"
    backward_returns: str      # e.g. "dx, dweight, dbias"

    # One-liner each: what the forward computes, what the backward returns.
    # Shown verbatim in the "Hard constraints" block.
    forward_semantics: str
    backward_semantics: str

    # Inputs with no gradient. The AtenIR graph may include their grads as
    # outputs; tell the LLM to discard them.
    no_grad_inputs: tuple[str, ...] = ()

    # Optional freeform text appended after the Triton pitfalls block.
    extra_constraints: str = ""


LAYERNORM_SPEC = OperatorSpec(
    forward_fn_name="layernorm_forward_with_saved",
    forward_args="x, weight, bias, eps=1e-5",
    backward_fn_name="layernorm_backward_from_saved",
    backward_args="dy, saved_tensors, eps=1e-5",
    backward_returns="dx, dweight, dbias",
    forward_semantics=(
        "Do not call PyTorch autograd or PyTorch reference LayerNorm in the generated math. "
        "Forward must produce the same `y` as row-wise LayerNorm over the last dimension."
    ),
    backward_semantics=(
        "Backward must consume only `dy`, `saved_tensors`, and `eps`. "
        "Return `dx` with `x` dtype, `dweight` with `weight` dtype, and `dbias` with `bias` dtype."
    ),
)


SYSTEM_MESSAGE = """You are a Triton compiler engineer.
You synthesize a forward/backward autograd pair.  The forward may save tensors
that the backward reuses.  Correctness is required; efficiency should balance
backward latency, forward+backward latency, and saved-tensor memory."""

_SAVED_TENSOR_GUIDANCE = """\
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
- It is acceptable to save original inputs if the backward needs them."""

_TRITON_PITFALLS = """\
Triton pitfalls:
- `tl.arange` bounds must be compile-time constants. Use `BLOCK_*: tl.constexpr`.
- Do not read Python globals inside `@triton.jit`; pass dimensions, strides, and
  scalar constants as arguments or meta-parameters.
- Use fp32 accumulation for reductions.
- Avoid global atomic contention when a partial-buffer reduction is better."""


def render_pair_rules(spec: OperatorSpec) -> str:
    no_grad_lines = ""
    if spec.no_grad_inputs:
        names = ", ".join(f"`{n}`" for n in spec.no_grad_inputs)
        no_grad_lines = (
            f"- The following inputs carry NO gradient and must not appear in the "
            f"backward output: {names}. "
            f"The AtenIR graph may include their gradients as outputs — discard them.\n"
        )
    extra = f"\n{spec.extra_constraints}" if spec.extra_constraints else ""
    return f"""\
## Autograd-pair rules

Public API to implement:

```python
def {spec.forward_fn_name}({spec.forward_args}):
    return y, saved_tensors

def {spec.backward_fn_name}({spec.backward_args}):
    return {spec.backward_returns}
```

Hard constraints:
- Return only Python source, no Markdown.
- Include imports for `torch`, `triton`, and `triton.language as tl`.
- Include an `EVOLVE-BLOCK` around generated Triton kernels and launch helpers.
- `saved_tensors` must be a tensor or tuple/list of tensors, because the evaluator
  stores them via `ctx.save_for_backward`.
- {spec.forward_semantics}
- {spec.backward_semantics}
{no_grad_lines}
{_SAVED_TENSOR_GUIDANCE}

{_TRITON_PITFALLS}
{extra}"""


def render_plan_prompt(
    *,
    forward: str,
    graph_summary: str,
    lowering_context: str = "",
    spec: OperatorSpec = LAYERNORM_SPEC,
) -> str:
    lowering_section = (
        f"\nAdditional lowering context:\n\n```text\n{lowering_context}\n```\n"
        if lowering_context else ""
    )
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


def render_codegen_prompt(
    *,
    graph_summary: str,
    pair_plan: str,
    lowering_context: str = "",
    spec: OperatorSpec = LAYERNORM_SPEC,
) -> str:
    lowering_section = (
        f"\nAdditional lowering context:\n\n```text\n{lowering_context}\n```\n"
        if lowering_context else ""
    )
    return f"""# Autograd-Pair Codegen

Generate a complete Python module for an autograd pair for the provided forward reference.

{render_pair_rules(spec)}

## Plan

```markdown
{pair_plan}
```

## AtenIR backward graph summary

{graph_summary}
{lowering_section}"""


def render_repair_prompt(
    *,
    graph_summary: str,
    pair_plan: str,
    previous_code: str,
    verifier_report: str,
    lowering_context: str = "",
    spec: OperatorSpec = LAYERNORM_SPEC,
) -> str:
    lowering_section = (
        f"\nAdditional lowering context:\n\n```text\n{lowering_context}\n```\n"
        if lowering_context else ""
    )
    return f"""# Autograd-Pair Repair

The generated autograd-pair program failed verification.

Verifier report:

```json
{verifier_report}
```

{render_pair_rules(spec)}

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
