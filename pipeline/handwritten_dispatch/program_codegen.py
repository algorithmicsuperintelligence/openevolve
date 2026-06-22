"""Generate a self-contained, dispatch-free Triton program for one AtenIR graph.

`generate_dispatch_program(graph)` resolves every call_function node to its concrete
kernel via `atenir.primitive_triton.dispatch.make_kernel` *at generation time only* —
the big string-matching `make_kernel`/`make_registry` machinery and the runtime
`_REGISTRY` dict never appear in the output. What's emitted instead:

  * only the Triton kernels / launcher functions actually reachable from the ops
    used by this graph (pulled transitively: a launcher that calls a helper that
    launches a `@triton.jit` kernel brings both along), deduplicated;
  * dispatch.py closures that just bake a scalar into a kernel call (e.g.
    `lambda a: elementwise.mul_scalar(a, 2.0)`) are inlined as ordinary functions
    with the baked values as literal default arguments — no closure, no registry;
  * one literal Python statement per AtenIR call_function node, in topological
    order, calling the resolved kernel directly.

The result is a plain Triton program: kernels + the one function that runs the graph.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import textwrap
from typing import Any, Callable

import torch

_SUBMODULE_NAMES = ("elementwise", "reduction", "gemm", "scatter_gather")


def _unwrap_jit(fn: Callable) -> Callable:
    """Triton wraps the user function in a JITFunction; get back the plain def."""
    for candidate in (getattr(fn, "fn", None), getattr(fn, "__wrapped__", None)):
        if candidate is not None:
            return candidate
    return fn


def _extract_def_or_lambda(stmt: ast.stmt) -> ast.FunctionDef:
    """Normalize a parsed top-level statement into a FunctionDef.

    Closures in dispatch.py are written either as `def f(...): ...` or as
    `return lambda a: ...` / `lambda a: ...` one-liners; both end up as a single
    FunctionDef so the rest of the pipeline doesn't need to special-case lambdas.
    """
    if isinstance(stmt, ast.FunctionDef):
        return stmt
    node = stmt
    if isinstance(node, ast.Return):
        node = node.value
    elif isinstance(node, ast.Expr):
        node = node.value
    if isinstance(node, ast.Lambda):
        return ast.FunctionDef(
            name="_lambda",
            args=node.args,
            body=[ast.Return(value=node.body)],
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
    raise ValueError(f"unsupported source shape for closure: {ast.dump(stmt)[:200]}")


class _UnqualifyTransformer(ast.NodeTransformer):
    """Rewrite `elementwise.div_scalar(...)` to `div_scalar(...)`.

    Once a cross-module helper is pulled into the flat output namespace it no
    longer needs (or has) a module object to hang off of — every kernel/helper
    this graph touches lives as a bare top-level name in the same file.
    """

    def __init__(self, submodule_names: frozenset[str]) -> None:
        self.submodule_names = submodule_names

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id in self.submodule_names:
            return ast.Name(id=node.attr, ctx=node.ctx)
        return node


class _ProgramBuilder:
    """Resolves AtenIR nodes to direct calls, pulling in only what's referenced."""

    def __init__(self) -> None:
        self.submodules: dict[str, Any] = {
            name: importlib.import_module(f"atenir.primitive_triton.{name}")
            for name in _SUBMODULE_NAMES
        }
        self.dispatch_mod = importlib.import_module("atenir.primitive_triton.dispatch")

        # The @triton.jit kernel functions only exist as module attributes inside
        # `if HAS_TRITON:` blocks. If triton isn't importable *in this process*,
        # those names are silently absent and dependency collection below would
        # quietly skip them — producing a file whose launchers call kernels that
        # were never emitted. Fail loudly instead of emitting a broken seed.
        missing = [
            name for name, mod in self.submodules.items() if not getattr(mod, "HAS_TRITON", False)
        ]
        if missing:
            raise RuntimeError(
                "generate_dispatch_program requires triton to be importable in this "
                f"process (HAS_TRITON is False for: {', '.join(missing)}); run it where "
                "`import triton` succeeds, otherwise kernel bodies are silently omitted."
            )

        self.helper_sources: dict[tuple[str, str], str] = {}  # (module, name) -> source
        self.helper_order: list[tuple[str, str]] = []
        self.constants: dict[str, str] = {}  # name -> "name = literal"
        self.generated_sources: dict[str, str] = {}  # generated name -> source
        self.generated_order: list[str] = []
        # id(fn) -> (fn, generated name). The strong ref to `fn` is required, not
        # cosmetic: without it the closure is GC'd as soon as the caller's local
        # goes out of scope, and CPython is free to hand that id to the very next
        # (unrelated) closure it allocates — silently aliasing two different ops.
        self._by_identity: dict[int, tuple[Callable, str]] = {}
        self._by_structure: dict[tuple, str] = {}
        self._counter = 0

    # ── public entry point for one graph node ──────────────────────────────────

    def call_expr_for_node(self, node: dict, kernel: Callable, arg_exprs: list[str]) -> str:
        """Return the literal Python call expression for this node, e.g.

        'elementwise.mul_tt(a, b)' or '_k3(a)' — and as a side effect, register
        every kernel/helper/constant this call needs so it ends up in the output.
        """
        target = node.get("target", "")
        if self._is_generic_fallback(kernel):
            return self._generic_fallback_call(target, arg_exprs)
        if self._is_unimplemented(kernel):
            raise NotImplementedError(f"No hand-written kernel for target {target!r}")
        callee = self._resolve(kernel)
        return f"{callee}({', '.join(arg_exprs)})"

    # ── resolving a callable to a call-site expression ──────────────────────────

    def _resolve(self, fn: Callable) -> str:
        qualname = getattr(fn, "__qualname__", "") or getattr(fn, "__name__", "")
        modname = getattr(fn, "__module__", "") or ""
        if "<locals>" not in qualname:
            short = self._submodule_short(modname)
            if short is not None:
                self._collect_module_helper(fn)
                return fn.__name__
            # Plain top-level callable outside the four kernel submodules (rare;
            # e.g. a bound torch function). Fall back to its repr/import path.
            raise ValueError(f"cannot reference top-level function outside known submodules: {fn!r}")

        key = id(fn)
        cached = self._by_identity.get(key)
        if cached is not None and cached[0] is fn:
            return cached[1]

        struct_key = self._structural_key(fn)
        if struct_key is not None and struct_key in self._by_structure:
            name = self._by_structure[struct_key]
            self._by_identity[key] = (fn, name)
            return name

        name = f"_k{self._counter}"
        self._counter += 1
        self._by_identity[key] = (fn, name)
        if struct_key is not None:
            self._by_structure[struct_key] = name
        func_def = self._build_standalone_def(fn, name)
        self.generated_sources[name] = ast.unparse(func_def)
        self.generated_order.append(name)
        return name

    def _structural_key(self, fn: Callable) -> tuple | None:
        """Two closures with the same source and the same captured values are the
        same kernel for codegen purposes (e.g. three `alias` nodes all resolve to
        `lambda a: a.contiguous()`) — dedupe them instead of emitting `_k0`, `_k1`,
        `_k2` for identical bodies. `None` opts a closure out of dedup (defensive:
        only dedupe when every captured value reprs to something stable).
        """
        try:
            src = inspect.getsource(fn)
        except (OSError, TypeError):
            return None
        freevars = fn.__code__.co_freevars
        cells = fn.__closure__ or ()
        try:
            captured = tuple(repr(cell.cell_contents) for cell in cells)
        except Exception:
            return None
        return (src, freevars, captured)

    def _build_standalone_def(self, fn: Callable, new_name: str) -> ast.FunctionDef:
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
        func_def = _extract_def_or_lambda(tree.body[0])
        func_def.name = new_name
        func_def.decorator_list = []

        self._collect_body_dependencies(func_def, fn)
        func_def = _UnqualifyTransformer(frozenset(self.submodules)).visit(func_def)

        existing_params = {a.arg for a in func_def.args.args} | {
            a.arg for a in func_def.args.kwonlyargs
        }
        freevars = fn.__code__.co_freevars
        cells = fn.__closure__ or ()
        for fv_name, cell in zip(freevars, cells):
            if fv_name in existing_params:
                continue
            default_expr = self._default_expr_for(cell.cell_contents)
            func_def.args.args.append(ast.arg(arg=fv_name))
            func_def.args.defaults.append(ast.parse(default_expr, mode="eval").body)

        ast.fix_missing_locations(func_def)
        return func_def

    def _default_expr_for(self, value: Any) -> str:
        if callable(value) and not isinstance(value, type):
            return self._resolve(value)
        if isinstance(value, torch.dtype):
            return repr(value)  # e.g. "torch.float32" — valid given `import torch`
        text = repr(value)
        try:
            ast.literal_eval(text)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"cannot represent captured value {value!r} as a literal default") from exc
        return text

    # ── pulling in module-level helpers (triton kernels, plain functions) ──────

    def _collect_module_helper(self, fn: Callable) -> None:
        real_fn = _unwrap_jit(fn)
        modname = getattr(real_fn, "__module__", "")
        name = getattr(real_fn, "__name__", None)
        if name is None:
            return
        key = (modname, name)
        if key in self.helper_sources:
            return
        try:
            src = textwrap.dedent(inspect.getsource(real_fn))
        except (OSError, TypeError):
            return
        self.helper_sources[key] = src  # placeholder; prevents recursive re-entry
        self.helper_order.append(key)
        try:
            tree = ast.parse(src)
            fdef = tree.body[0]
        except SyntaxError:
            return
        if isinstance(fdef, ast.FunctionDef):
            self._collect_body_dependencies(fdef, real_fn)
            fdef = _UnqualifyTransformer(frozenset(self.submodules)).visit(fdef)
            ast.fix_missing_locations(fdef)
            self.helper_sources[key] = ast.unparse(fdef)

    def _collect_body_dependencies(self, func_def: ast.AST, origin_fn: Callable) -> None:
        origin_module = inspect.getmodule(_unwrap_jit(origin_fn))
        local_bound = {a.arg for a in func_def.args.args} | {
            a.arg for a in getattr(func_def.args, "kwonlyargs", [])
        }
        for n in ast.walk(func_def):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                local_bound.add(n.id)

        for n in ast.walk(func_def):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
                base = n.value.id
                target_mod = self.submodules.get(base)
                if target_mod is None:
                    continue
                attr_val = getattr(target_mod, n.attr, None)
                if attr_val is None:
                    continue
                if callable(attr_val):
                    self._collect_module_helper(attr_val)
                else:
                    self.constants[n.attr] = f"{n.attr} = {attr_val!r}"
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                name = n.id
                if name in local_bound or name in self.submodules:
                    continue
                if name in ("torch", "tl", "triton", "libdevice"):
                    continue
                if origin_module is None or not hasattr(origin_module, name):
                    continue
                candidate = getattr(origin_module, name)
                if callable(candidate):
                    cand_mod = getattr(candidate, "__module__", "")
                    cand_qual = getattr(_unwrap_jit(candidate), "__qualname__", "") or ""
                    if cand_mod == origin_module.__name__ and "<locals>" not in cand_qual:
                        self._collect_module_helper(candidate)
                else:
                    try:
                        ast.literal_eval(repr(candidate))
                        self.constants[name] = f"{name} = {candidate!r}"
                    except Exception:
                        pass

    # ── special dispatch.py cases that aren't real per-op kernels ───────────────

    def _is_generic_fallback(self, kernel: Callable) -> bool:
        # dispatch.py's _generic_fallback overwrites __name__/__qualname__ to
        # "pytorch_fallback[target]" for display purposes, so qualname can't be
        # used to recognize it — _dispatch_tag is the stable marker.
        tag = getattr(kernel, "_dispatch_tag", "")
        return tag.startswith("pytorch:fallback[")

    def _is_unimplemented(self, kernel: Callable) -> bool:
        tag = getattr(kernel, "_dispatch_tag", "")
        return tag.startswith("pytorch:unimplemented[")

    def _generic_fallback_call(self, target: str, arg_exprs: list[str]) -> str:
        """Targets dispatch.py routes to plain `torch.ops.aten.*` (no Triton kernel
        exists for them yet). Resolve via attribute traversal — not string-matching
        dispatch logic, just the aten op the target string already names.
        """
        self.constants.setdefault(
            "_resolve_aten", inspect.getsource(self.dispatch_mod._resolve_aten)
        )
        return f"_resolve_aten({target!r})({', '.join(arg_exprs)})"

    def _submodule_short(self, modname: str) -> str | None:
        for short in _SUBMODULE_NAMES:
            if modname == self.submodules[short].__name__:
                return short
        return None

    # ── final assembly ──────────────────────────────────────────────────────────

    def render_kernel_library(self) -> str:
        parts: list[str] = []
        if self.constants:
            for name in sorted(self.constants):
                if name == "_resolve_aten":
                    continue
                parts.append(self.constants[name])
            parts.append("")
        if "_resolve_aten" in self.constants:
            parts.append(self.constants["_resolve_aten"])
            parts.append("")
        for key in self.helper_order:
            parts.append(self.helper_sources[key])
            parts.append("")
        for name in self.generated_order:
            parts.append(self.generated_sources[name])
            parts.append("")
        return "\n".join(parts).strip() + "\n"


def _pyname(node_name: str) -> str:
    """Namespace a graph node name so it can never collide with a kernel/helper
    function name living in the same flat module (e.g. a node literally named
    "mm" would otherwise shadow the `mm` kernel function: `mm = mm(...)` raises
    UnboundLocalError since the assignment makes `mm` local for the whole def).
    """
    return f"t_{node_name}"


def generate_dispatch_program(graph: dict) -> str:
    """Build the full dispatch_program.py source text for an AtenIR graph dict."""
    from atenir.primitive_triton.dispatch import make_kernel

    placeholders = [n for n in graph["nodes"] if n["op"] == "placeholder"]
    call_nodes = [n for n in graph["nodes"] if n["op"] == "call_function"]
    out_node = graph["nodes"][-1]
    if out_node.get("op") != "output":
        raise ValueError("graph's last node is not an output node; cannot generate program")

    builder = _ProgramBuilder()
    call_lines: list[str] = []
    for node in call_nodes:
        name = node["name"]
        target = node["target"]
        args_ordered = node.get("args_ordered")
        if args_ordered is not None:
            arg_names = [e["name"] for e in args_ordered if e["kind"] == "node"]
        else:
            arg_names = list(node.get("predecessor_ids") or [])
        kernel = make_kernel(node)
        call_expr = builder.call_expr_for_node(node, kernel, [_pyname(a) for a in arg_names])
        call_lines.append(f"    {_pyname(name)} = {call_expr}  # {name}: {target}")

    output_names = out_node["args"][0]
    trailing_comma = "," if len(output_names) == 1 else ""
    call_lines.append(
        f"    return ({', '.join(_pyname(n) for n in output_names)}{trailing_comma})"
    )

    placeholder_names = [_pyname(p["name"]) for p in placeholders]
    kernel_library = builder.render_kernel_library()

    header = '''"""Auto-generated by Pipeline D (handwritten_dispatch) — a self-contained,
dispatch-free Triton program for one AtenIR graph.

Only the Triton kernels actually used by this graph are included below (pulled
transitively from atenir/primitive_triton/), wrapped in an EVOLVE-BLOCK so
OpenEvolve can rewrite/fuse them. There is no string-matching op dispatch table
and no runtime registry: every call in `run_graph_program` below the block is a
direct call to one of these kernels, with any baked scalar arguments inlined as
literal defaults.

`run_graph_program` is the literal, unrolled call sequence for this graph — one
statement per AtenIR call_function node, in topological order.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
'''

    lines = [
        header,
        "# EVOLVE-BLOCK-START",
        "",
        kernel_library,
        "# EVOLVE-BLOCK-END",
        "",
        "",
        f"def run_graph_program({', '.join(placeholder_names)}):",
        *call_lines,
    ]
    return "\n".join(lines) + "\n"
