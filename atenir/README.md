# AtenIR — a small front-end for multi-agent Triton backward synthesis

AtenIR is the layer that sits between PyTorch's autograd dispatcher and a
multi-agent Triton kernel synthesizer.  Its job is narrow:

1. **Extract** — turn a backward op (or any differentiable forward) into a
   JSON graph of primitive aten nodes that an LLM agent can read.
2. **Compose** — take that JSON, an `env` dict of placeholder tensors, and a
   kernel registry, and run the graph by dispatching each node to the
   registered kernel.
3. **Adapt** — bridge a specific benchmark's frozen function API to the
   placeholder schema the graph expects (recompute missing inputs, cast
   dtypes, override shape-specialised constants).

Only the third piece is op-specific.  Extraction and composition are
op-agnostic.

## Files

| File | Role |
|---|---|
| `extract.py` | CLI for graph extraction — see "Extract" below |
| `compose.py` | IR runtime: `run_graph(graph, env, registry, scalar_overrides)` |
| `_examples.py` | small forward callables used by extraction tests |
| `_diff.py` | structural diff for two AtenIR JSON files (name-agnostic) |
| `__init__.py` | makes `atenir` a package |

This package lives at the repo root (`openevolve/atenir/`) so any bench can
import it as `from atenir.compose import run_graph` once `openevolve` is on
the import path.  LayerNorm-specific glue (the worked-example adapter) lives
at the bench root as
`examples/triton_layernorm_backward_bench/backward_atenir.py`, alongside the
existing `backward_naive_triton.py`.

## Extract

Two modes, both produce a JSON graph using the same schema (placeholders,
call_function nodes with `input_nodes` / `predecessor_ids` / `scalar_args` /
`reduction_dims` / `keepdim`, and a single `output` node).

**Named aten op** — for backwards that PyTorch already exposes as an op:

```
python -m atenir.extract \
  --op aten._softmax_backward_data \
  --example-args "[(64,128) f32, (64,128) f32, -1, f32]" \
  --out softmax_bwd.json
```

The extractor wraps the op so that only tensor positions become FX
placeholders; Python list / scalar / dtype arguments are closed over and
remain inline constants.

**Autograd-driven** — for ops without a named aten backward.  Provide the
forward callable and the extractor will trace `torch.autograd.grad(...)`
through it:

```
python -m atenir.extract \
  --fn atenir._examples:square_sum \
  --example-input "[(32,64) f32]" \
  --out square_sum_bwd.json
```

Spec grammar for `--example-args` / `--example-input`:

```
spec   := '[' item (',' item)* ']'
item   := tensor | list | scalar | bool | dtype
tensor := '(' int (',' int)* [','] ')' dtype
list   := '[' item (',' item)* ']'
scalar := int | float
dtype  := f16 | f32 | f64 | bf16 | i32 | i64 | bool
```

## Compose

`compose.py` is one function plus a JSON loader.  It walks `graph["nodes"]`
in order, skips `placeholder` / `output`, and for each `call_function`
node looks up `kernel_registry[node["name"]]`, calls it with the
predecessor tensors followed by the scalar args, and stores the result in
`env`.  At the end it reads `graph["nodes"][-1]["args"][0]` and returns
those tensors in order.

```python
from atenir.compose import run_graph

dx, dw, db = run_graph(
    "path/to/graph.json",
    env={"grad_out_1": dy, "x_1": x, ...},
    kernel_registry={"sub": sub_run, "mul": mul_run, ...},
    scalar_overrides={"mul_2": [cols], "div": [cols]},
)
```

The runtime knows nothing about LayerNorm, fp32, mean, rstd, or any
particular API.  All of that lives in adapters.

## Adapters

`examples/triton_layernorm_backward_bench/backward_atenir.py` is the worked
example.  It implements the bench's frozen public function
`layernorm_backward_triton(dy, x, weight, bias, eps)`, does the LayerNorm-
specific prep (mean/rstd recompute, dtype casting, N-shape override), builds
the `env` dict, calls `run_graph`, and casts the outputs back to the
original dtypes.

A new bench / op gets its own adapter file alongside `backward_atenir.py`;
`compose.py` and `extract.py` do not change.

## Known limitation — shape-specialised constants

`make_fx` traces at a concrete shape and bakes integer dims into
`scalar_args` (e.g. `mul_2: scalar_args=[256]` for a graph captured at
`cols=256`).  The composer currently exposes `scalar_overrides` so an
adapter can swap those constants for the runtime value.  The proper fix
is **symbolic-mode extraction** (`make_fx(..., tracing_mode="symbolic")`)
so the graph carries SymInt nodes that the runtime computes from input
shapes — that work is TODO and would remove the need for adapter-side
overrides entirely.
