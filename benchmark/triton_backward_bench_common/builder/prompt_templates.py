"""Prompt artifact templates for Stage 1 backward benchmark construction."""

from __future__ import annotations

from .spec import BenchmarkSpec


def _format_inputs(spec: BenchmarkSpec) -> str:
    lines = []
    for input_spec in spec.inputs:
        diff = "differentiable" if input_spec.differentiable else "non-differentiable"
        lines.append(f"- `{input_spec.name}`: shape `{input_spec.shape}`, layout `{input_spec.layout}`, {diff}")
    return "\n".join(lines)


def _format_formulas(spec: BenchmarkSpec) -> str:
    if not spec.backward_formulas:
        return "- No formula hints provided; derive from the forward definition."
    return "\n".join(f"- `{name}`: `{formula}`" for name, formula in spec.backward_formulas.items())


def canonical_naive_plan(spec: BenchmarkSpec) -> str:
    """Return a concise canonical multi-kernel plan for known operators."""
    if spec.operator == "layernorm":
        return """Use this exact canonical naive kernel plan:

1. `_compute_xhat_kernel`
   - inputs: `x`
   - outputs: `xhat`, `rstd`
   - computes row-wise mean, variance, reciprocal std, and xhat
2. `_compute_one_kernel`
   - inputs: `dy`, `weight`
   - outputs: `one = dy * weight`
3. `_mean_one_kernel`
   - inputs: `one`
   - outputs: `mean_one`
   - row-wise reduction over hidden dimension
4. `_mean_xhat_one_kernel`
   - inputs: `xhat`, `one`
   - outputs: `mean_xhat_one`
   - row-wise reduction over hidden dimension
5. `_dx_kernel`
   - inputs: `one`, `xhat`, `rstd`, `mean_one`, `mean_xhat_one`
   - outputs: `dx`
6. `_dweight_kernel`
   - inputs: `dy`, `xhat`
   - outputs: `dweight`
   - column-wise reduction over rows
7. `_dbias_kernel`
   - inputs: `dy`
   - outputs: `dbias`
   - column-wise reduction over rows

Forbidden fusions for this baseline:

- Do not combine `_mean_one_kernel` and `_mean_xhat_one_kernel`.
- Do not combine `_dweight_kernel` and `_dbias_kernel`.
- Do not combine `_compute_xhat_kernel` and `_dx_kernel`.
- Do not materialize `dweight_partial`.
- Do not store `var` unless needed only inside `_compute_xhat_kernel`."""

    if spec.operator == "bias_relu":
        return """Use this exact canonical naive kernel plan:

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
- Do not combine the pointwise and reduction kernels."""

    return "No canonical kernel plan is registered for this operator. Use the simplest unfused decomposition."


def render_backward_formula_prompt(spec: BenchmarkSpec) -> str:
    return f"""# Backward Formula Derivation Prompt

You are deriving a deliberately naive backward decomposition for a training-oriented
Triton benchmark task. This is Stage 1 of benchmark construction, not optimization.

Operator: `{spec.operator}`

Forward semantic ground truth:

```text
{spec.forward_formula}
```

Inputs:

{_format_inputs(spec)}

Upstream gradient: `{spec.upstream_gradient}`

Required gradients:

```text
{", ".join(spec.required_gradients)}
```

Known formula hints from metadata:

{_format_formulas(spec)}

Canonical naive decomposition policy:

{canonical_naive_plan(spec)}

Task:

1. Derive clear mathematical formulas for every required gradient.
2. Identify reductions, broadcasting semantics, saved tensors, and fp32 accumulation requirements.
3. Break the backward into a sequence of simple named intermediate tensors.
4. Prefer an unfused decomposition suitable for a readable multi-kernel Triton baseline.
5. For each intermediate, state its shape, whether it is pointwise or a reduction, and which axis is reduced.
6. Do not write Triton code here.
7. Do not claim the derivation is ground truth; the oracle is PyTorch autograd over `forward_ref.py`.
"""


def render_naive_triton_prompt(spec: BenchmarkSpec) -> str:
    return f"""# Naive Triton Backward Generation Prompt

You are writing a naive, readable Triton backward baseline for `{spec.operator}`.

Forward semantic ground truth:

```text
{spec.forward_formula}
```

Inputs:

{_format_inputs(spec)}

Required outputs:

```text
{", ".join(spec.required_gradients)}
```

Formula hints:

{_format_formulas(spec)}

Canonical naive kernel plan:

{canonical_naive_plan(spec)}

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
"""


def render_repair_prompt_template(spec: BenchmarkSpec) -> str:
    return f"""# Repair Prompt Template

The naive Triton backward candidate for `{spec.operator}` failed verification.

Forward semantic ground truth:

```text
{spec.forward_formula}
```

Required gradients:

```text
{", ".join(spec.required_gradients)}
```

Expected oracle:

```text
PyTorch autograd over forward_ref.py
```

Insert verifier feedback below before sending this prompt to an agent:

```text
{{VERIFIER_ERROR_REPORT}}
```

Repair task:

1. Explain the likely mathematical or indexing error.
2. Modify only the naive Triton backward candidate.
3. Preserve the public API and output order.
4. Keep the implementation simple, readable, and unfused.
5. Do not weaken tests or tolerances.
6. Do not add argument-order compatibility hacks.
"""


def render_all_prompts(spec: BenchmarkSpec) -> dict[str, str]:
    return {
        "backward_formula_prompt.md": render_backward_formula_prompt(spec),
        "naive_triton_backward_prompt.md": render_naive_triton_prompt(spec),
        "repair_prompt_template.md": render_repair_prompt_template(spec),
    }


def render_oracle_guidance(spec: BenchmarkSpec) -> str:
    differentiable = ", ".join(input_spec.name for input_spec in spec.differentiable_inputs())
    gradients = ", ".join(spec.required_gradients)
    return f"""# Autograd Oracle Guidance

`backward_ref.py` should be the correctness oracle for `{spec.task_name}`.
It must compute `{gradients}` by running PyTorch autograd over `forward_ref.py`.

Semantic source:

```text
{spec.forward_formula}
```

Differentiable inputs:

```text
{differentiable}
```

Required gradients:

```text
{gradients}
```

Policy:

- Do not use LLM-derived formulas as ground truth.
- Clone inputs before setting `requires_grad_(True)`.
- Call the PyTorch forward reference.
- Call `.backward(dy)` or `torch.autograd.grad(...)`.
- Return gradients in the exact order listed in `required_gradients`.
"""


def check_oracle_source(source: str) -> list[str]:
    warnings = []
    for marker in ["requires_grad_(True)", "backward", "forward_ref", ".grad"]:
        if marker not in source:
            warnings.append(f"missing marker: {marker}")
    if "triton" in source.lower():
        warnings.append("oracle source references Triton; backward_ref should use PyTorch autograd only")
    return warnings
