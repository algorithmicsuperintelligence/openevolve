"""Prompt templates for the AtenIR per-op Triton kernel lowering agent."""

from __future__ import annotations

_TRITON_REFERENCE = """
TRITON KERNEL RULES — READ CAREFULLY:

Triton is NOT Python/NumPy. You CANNOT index pointers like Python arrays.
The ONLY correct way to read/write memory is with tl.load / tl.store on vectorised offsets.

WRONG (will fail with CompilationError):
    idx = tl.program_id(0)
    if idx < N:
        out_ptr[idx] = a_ptr[idx] + b_ptr[idx]   # ← illegal

CORRECT elementwise kernel pattern:
    @triton.jit
    def _add_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)   # tile of BLOCK indices
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, a + b, mask=mask)

    def kernel_add(*args):
        a, b = args[0].contiguous(), args[1].contiguous()
        out = torch.empty_like(a)
        N = a.numel()
        BLOCK = 1024
        _add_kernel[(triton.cdiv(N, BLOCK),)](a, b, out, N, BLOCK=BLOCK)
        return out

CORRECT row-reduction kernel pattern (e.g. sum over last dim):
    @triton.jit
    def _row_sum_kernel(a_ptr, out_ptr, R, C, BLOCK_C: tl.constexpr):
        row = tl.program_id(0)                          # one program per row
        cols = tl.arange(0, BLOCK_C)
        mask = cols < C
        a = tl.load(a_ptr + row * C + cols, mask=mask, other=0.0)
        tl.store(out_ptr + row, tl.sum(a.to(tl.float32), axis=0))

    def kernel_sum(*args):
        a = args[0].contiguous()
        R, C = a.shape
        out = torch.empty(R, device=a.device, dtype=torch.float32)
        BLOCK_C = triton.next_power_of_2(C)
        _row_sum_kernel[(R,)](a, out, R, C, BLOCK_C=BLOCK_C)
        return out
"""

SYSTEM_MESSAGE = f"""You are a Triton compiler engineer.
You implement AtenIR primitive ops as Triton kernels.
Correctness is more important than speed. Every kernel must match PyTorch semantics exactly.

{_TRITON_REFERENCE}"""


# ── per-node generation ───────────────────────────────────────────────────────


def render_op_codegen_prompt(
    *,
    node_summary: str,
    fn_name: str,
    graph_context: str = "",
) -> str:
    """Per-node prompt. graph_context is intentionally short (a few predecessor lines)
    to avoid burning TPM on repeated full-graph summaries."""
    node_name = fn_name[len("kernel_") :] if fn_name.startswith("kernel_") else fn_name
    context_section = f"\nPredecessor context:\n\n{graph_context}\n" if graph_context else ""
    return f"""# AtenIR Op Lowering: {fn_name}

{node_summary}{context_section}
Generate a Python function that implements this single AtenIR op.

The function **must** be named exactly `{fn_name}` with this signature:

```python
def {fn_name}(*args):
    ...  # returns one torch.Tensor
```

`args` contains the arguments **in the positional order listed above** — tensors
and scalars interleaved exactly as shown. For reduction ops, `reduction_dims` and
`keepdim` are encoded in the graph metadata and do NOT appear in `args`; bake them
as constants inside the kernel.

Requirements:
- Use a `@triton.jit` kernel for the computation when practical.
- For ops that are impractical in Triton (view, reshape, expand, clone, transpose,
  permute, matmul, bmm, topk, scatter, gather), use PyTorch directly instead.
- For shape-only ops such as view/reshape/expand: shape arguments in the graph
  may contain traced example dimensions. Do not blindly bake positive shape
  constants into reusable code when the same dimension can be derived from a
  runtime input tensor. Preserve `-1` semantics and prefer runtime shapes such
  as `tensor.shape[i]` when they correspond to the requested view/expand size.
- Output dtype must match the expected output dtype shown above.
- Use fp32 accumulation for reductions; cast back to input dtype on store.
- Derive ALL shapes and strides from input tensors at runtime — never hardcode numbers.
- Do NOT add import statements (they are already in the enclosing file).
- Do NOT use `HAS_TRITON` guards inside the function.
- CRITICAL — unique helper names: every `@triton.jit` kernel you write MUST be
  named starting with `_{node_name}_`, e.g. `_{node_name}_kernel` or
  `_{node_name}_row_sum`. NEVER use generic names like `_mul_kernel`, `sum_kernel`,
  `add_kernel`, or `neg_kernel`. This file assembles kernels from many nodes into
  one Python module; duplicate names cause silent misdispatch because Python
  overwrites earlier definitions with later ones.

Example of the required naming style for node `{node_name}`:

```python
@triton.jit
def _{node_name}_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, a * b, mask=mask)

def {fn_name}(*args):
    a, b = args[0], args[1]
    a, b = torch.broadcast_tensors(a.contiguous(), b.contiguous())
    out = torch.empty_like(a)
    N = a.numel()
    BLOCK = 1024
    _{node_name}_kernel[(triton.cdiv(N, BLOCK),)](a, b, out, N, BLOCK=BLOCK)
    return out
```

Return only the Python source. No markdown, no module-level imports.
"""


# ── per-node repair ──────────────────────────────────────────────────────────


def render_op_repair_prompt(
    *,
    node_summary: str,
    fn_name: str,
    previous_code: str,
    error_report: str,
) -> str:
    return f"""# AtenIR Kernel Repair: {fn_name}

The kernel below failed verification. Fix it.

## Node specification

{node_summary}

## Error report

```json
{error_report}
```

## Previous (broken) code

```python
{previous_code}
```

## Common Triton mistakes to check

- Using Python indexing (`ptr[i] = val`) instead of `tl.load` / `tl.store`.
- Missing `BLOCK: tl.constexpr` parameter — required for block size.
- Wrong grid: must be a tuple, e.g. `(triton.cdiv(N, BLOCK),)`.
- `tl.math.pow` does not exist — use `tl.exp(exp * tl.log(base))`.
- Hardcoded shapes — derive from input tensors at runtime.
- For view/reshape/expand, do not blindly reuse traced shape constants such as
  `[64]` when the correct dimension is available from an input tensor at
  runtime. Preserve `-1` semantics and derive dynamic dimensions from
  `tensor.shape`.
- `tl.log` and `tl.exp` only accept fp32/fp64 — they fail with a CompilationError
  on fp16 inputs. Always cast to fp32 before calling them:
  `x = tl.load(...).to(tl.float32)`, then compute, then store via
  `torch.empty_like(input)` so the output is automatically cast back.
- `tl.exp(exp * tl.log(base))` produces `NaN` for negative `base`, even when
  PyTorch `base ** exp` is valid (e.g. `(-2.0) ** 2 = 4.0`). Fix: use
  `tl.abs(base)` for the log; for even exponents the sign is always positive.
  For special exponents prefer direct formulas:
    - `exp == 0` → `tl.full(shape, 1.0, dtype)`
    - `exp == 1` → `base`
    - `exp == 2` → `base * base`
    - `exp == 3` → `base * base * base`
  Or fall back to PyTorch: `return torch.pow(tensor, scalar)`.
- Mask shape must match the pointer shape in `tl.load`: a scalar pointer needs
  no mask (or a scalar mask); a 1-D tile pointer needs a 1-D vector mask.
- For row-broadcast ops (e.g. `x[row, col] - mean[row]`): load `mean` as a
  scalar once per row program, then subtract from the tile — do not load it
  with a vector mask.
- `tl.arange` only accepts `tl.constexpr` bounds. Use `BLOCK_*` constants and masks, not runtime dimensions.
- Every `tl.zeros`, `tl.full`, and `tl.arange` shape element must be a
  `tl.constexpr` integer. If an output width like `C` is dynamic, pass
  `BLOCK_C: tl.constexpr` and use masks; do not write `tl.zeros((C,), ...)`.
- For reductions, reduce over exactly `reduction_dims` and preserve `keepdim`.
  For example, a reduction over dim 0 of a 2-D tensor should produce one value
  per column when `keepdim=True`, with output shape `[1, C]`.
- Do not use Python `if` on Triton tensor values inside `@triton.jit`; use
  masks and `tl.where` instead.

Return only the Python source for `{fn_name}` and its `@triton.jit` helpers.
No markdown, no imports.
"""
