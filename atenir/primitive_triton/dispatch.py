"""Build an AtenIR kernel registry from a graph JSON.

``make_registry(graph)`` walks every call_function node and assigns a kernel:
  - Category 1 (view/shape ops) → PyTorch, output_shape captured in closure.
  - Category 2 (pointwise) → Triton kernels in elementwise module.
  - Category 3 (reductions/scan) → Triton kernels in reduction module.
  - Category 4 (GEMM) → PyTorch torch.mm / torch.bmm.
  - Category 5 (skip / complex) → PyTorch generic fallback.

The returned dict maps node name → callable compatible with compose.run_graph.
"""

from __future__ import annotations

import sys
from typing import Callable

import torch

from . import elementwise, gemm, reduction, scatter_gather

# ── helpers ───────────────────────────────────────────────────────────────────


def _resolve_aten(target: str):
    rest = target[len("aten.") :] if target.startswith("aten.") else target
    parts = rest.split(".")
    obj = torch.ops.aten
    for p in parts:
        obj = getattr(obj, p)
    return obj


def _generic_fallback(target: str) -> Callable:
    try:
        op = _resolve_aten(target)
    except (AttributeError, ValueError):

        def fn(*args):
            raise NotImplementedError(f"No kernel for target {target!r}")

        fn.__name__ = fn.__qualname__ = f"unimplemented[{target}]"
        fn._dispatch_tag = f"pytorch:unimplemented[{target}]"
        return fn

    def fn(*args):
        return op(*args)

    fn.__name__ = fn.__qualname__ = f"pytorch_fallback[{target}]"
    fn._dispatch_tag = f"pytorch:fallback[{target}]"
    return fn


def _parse_dtype(dtype_str: str):
    return {
        "torch.float32": torch.float32,
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.int64": torch.int64,
        "torch.int32": torch.int32,
        "torch.bool": torch.bool,
    }.get(dtype_str)


# ── main dispatch ──────────────────────────────────────────────────────────────


def make_kernel(node: dict) -> Callable:  # noqa: C901 (complex but intentionally exhaustive)
    target: str = node["target"]
    ao = node.get("args_ordered") or []

    if ao:
        n_tensors = sum(1 for e in ao if e.get("kind") == "node")
        all_scalars = [e["value"] for e in ao if e.get("kind") == "scalar"]
        scalar_args = [v for v in all_scalars if v is not None]
        scalar_first = (
            len(ao) >= 2 and ao[0].get("kind") == "scalar" and ao[1].get("kind") == "node"
        )
    else:
        # Backward compat: graphs serialised before args_ordered was added.
        n_tensors = len(node.get("predecessor_ids") or [])
        scalar_args = list(node.get("scalar_args") or [])
        all_scalars = scalar_args
        scalar_first = False

    dims = node.get("reduction_dims")
    keepdim = bool(node.get("keepdim") or False)
    output_shape = node.get("output_shape")

    # ── Category 2: existing pointwise ops ────────────────────────────────────

    if "aten.mul" in target:
        if n_tensors >= 2:
            return elementwise.mul_tt
        s = float(scalar_args[0]) if scalar_args else 1.0

        def _mul_s(a):
            return elementwise.mul_scalar(a, s)

        return _mul_s

    # "aten.add." (trailing dot) so this does NOT also match "aten.addmm.default";
    # addmm is handled in the matmul group below.
    if "aten.add." in target:
        if n_tensors >= 2:
            alpha = float(scalar_args[0]) if scalar_args else 1.0
            if alpha == 1.0:
                return elementwise.add_tt

            def _add_alpha(a, b):
                return elementwise.add_tt(a, b, alpha)

            return _add_alpha
        s = float(scalar_args[0]) if scalar_args else 0.0

        def _add_s(a):
            return elementwise.add_scalar(a, s)

        return _add_s

    if "aten.sub" in target:
        if n_tensors >= 2:
            alpha = float(scalar_args[0]) if scalar_args else 1.0
            if alpha == 1.0:
                return elementwise.sub_tt

            def _sub_alpha(a, b):
                return elementwise.sub_tt(a, b, alpha)

            return _sub_alpha
        s = float(scalar_args[0]) if scalar_args else 0.0
        if scalar_first:

            def _rsub_s(a):
                return elementwise.add_scalar(elementwise.neg(a), s)

            return _rsub_s

        def _sub_s(a):
            return elementwise.add_scalar(a, -s)

        return _sub_s

    if "aten.div" in target:
        # rounding_mode is a str in scalar_args; numeric scalar comes first for Scalar variants
        rounding_mode = next((v for v in scalar_args if isinstance(v, str)), None)
        numeric = [v for v in scalar_args if not isinstance(v, str) and v is not None]
        if rounding_mode == "trunc":
            if n_tensors >= 2:
                return lambda a, b: elementwise.trunc_(elementwise.div_tt(a, b))
            s = float(numeric[0]) if numeric else 1.0

            def _div_trunc_s(a):
                return elementwise.trunc_(elementwise.div_scalar(a, s))

            return _div_trunc_s
        if rounding_mode == "floor":
            if n_tensors >= 2:
                return lambda a, b: elementwise.floor_(elementwise.div_tt(a, b))
            s = float(numeric[0]) if numeric else 1.0

            def _div_floor_s(a):
                return elementwise.floor_(elementwise.div_scalar(a, s))

            return _div_floor_s
        if n_tensors >= 2:
            return elementwise.div_tt
        s = float(numeric[0]) if numeric else 1.0
        if scalar_first:
            # div.Tensor(scalar, tensor) = scalar / tensor = scalar * reciprocal(tensor)
            def _rdiv_s(a):
                return elementwise.mul_scalar(elementwise.reciprocal(a), s)

            return _rdiv_s

        def _div_s(a):
            return elementwise.div_scalar(a, s)

        return _div_s

    if "aten.rsqrt" in target:
        return elementwise.rsqrt

    if "aten.neg" in target:
        return elementwise.neg

    if target == "aten.exp.default":
        return elementwise.exp

    if "aten.pow" in target:
        if n_tensors >= 2:
            return elementwise.pow_tt
        if "Scalar_Tensor" in target or (
            "Scalar" in target and "Tensor_Scalar" not in target and "Tensor" not in target
        ):
            import math

            base = float(scalar_args[0]) if scalar_args else math.e
            log_base = math.log(base) if base > 0 else 0.0

            def _pow_base(tensor_exp):
                return elementwise.exp(
                    elementwise.mul_scalar(tensor_exp.to(torch.float32), log_base)
                )

            return _pow_base
        exp_val = float(scalar_args[0]) if scalar_args else 2.0

        def _pow_s(a):
            return elementwise.pow_scalar(a, exp_val)

        return _pow_s

    # ── Category 2: new pointwise ops ─────────────────────────────────────────

    if "aten.abs" in target:
        return elementwise.abs_

    if "aten.sqrt" in target:
        return elementwise.sqrt_

    if "aten.reciprocal" in target:
        return elementwise.reciprocal

    if "aten.sign" in target:
        return elementwise.sign

    # More-specific (longer) names MUST come before their shorter prefixes
    # e.g. "aten.tan" in "aten.tanh.default" is True — tanh must win.
    if "aten.atanh" in target:
        return elementwise.atanh_

    if "aten.acosh" in target:
        return elementwise.acosh_

    if "aten.asinh" in target:
        return elementwise.asinh_

    if "aten.tanh" in target:
        return elementwise.tanh_

    if "aten.sinh" in target:
        return elementwise.sinh_

    if "aten.cosh" in target:
        return elementwise.cosh_

    if "aten.atan2" in target:
        return elementwise.atan2_

    if "aten.atan" in target:
        return elementwise.atan_

    if "aten.asin" in target:
        return elementwise.asin_

    if "aten.acos" in target:
        return elementwise.acos_

    if "aten.tan" in target:
        return elementwise.tan_

    if "aten.sin" in target:
        return elementwise.sin_

    if "aten.cos" in target:
        return elementwise.cos_

    if "aten.expm1" in target:
        return elementwise.expm1_

    if target == "aten.log.default":
        return elementwise.log_

    if "aten.log1p" in target:
        return elementwise.log1p_

    if "aten.log2" in target:
        return elementwise.log2_

    if "aten.log10" in target:
        return elementwise.log10_

    if "aten.erf" in target and "erfc" not in target:
        return elementwise.erf_

    if "aten.ceil" in target:
        return elementwise.ceil_

    if "aten.floor" in target and "floor_divide" not in target:
        return elementwise.floor_

    if "aten.trunc" in target and "trunc_divide" not in target:
        return elementwise.trunc_

    if "aten.round" in target:
        return elementwise.round_

    if "aten.sigmoid" in target:
        return elementwise.sigmoid_

    if "aten.relu" in target:
        return elementwise.relu

    if "aten.gelu" in target:
        return elementwise.gelu

    if "aten.silu" in target:
        return elementwise.silu

    if "aten.mish" in target:
        return elementwise.mish

    if "aten.softplus" in target:
        # Default beta=1, threshold=20; parametric forms fall back to PyTorch.
        beta = float(scalar_args[0]) if len(scalar_args) > 0 else 1.0
        threshold = float(scalar_args[1]) if len(scalar_args) > 1 else 20.0
        if beta == 1.0 and threshold == 20.0:

            def _softplus(a):
                return elementwise.softplus(a)

            return _softplus
        return _generic_fallback(target)

    if "aten.hardsigmoid" in target:
        return elementwise.hardsigmoid

    if "aten.hardswish" in target:
        return elementwise.hardswish

    if "aten.special_erfc" in target or target == "aten.erfc.default":
        return elementwise.erfc_

    if "aten.exp2" in target:
        return elementwise.exp2_

    if "aten.lerp" in target:
        if n_tensors >= 3:
            return elementwise.lerp_tt
        weight = float(scalar_args[0]) if scalar_args else 0.5

        def _lerp_s(a, b):
            return elementwise.lerp_scalar(a, b, weight)

        return _lerp_s

    if "aten.copysign" in target:
        return elementwise.copysign_tt

    if "aten.leaky_relu" in target:
        neg_slope = float(scalar_args[0]) if scalar_args else 0.01

        def _leaky(a):
            return elementwise.leaky_relu(a, neg_slope)

        return _leaky

    if "aten.elu" in target:
        alpha = float(scalar_args[0]) if scalar_args else 1.0

        def _elu(a):
            return elementwise.elu(a, alpha)

        return _elu

    if "aten.hardtanh" in target or "aten.clamp" in target:
        # all_scalars preserves None slots so positions map correctly:
        # clamp(t, None, 6) → all_scalars=[None, 6] → lo=None, hi=6.
        lo = float(all_scalars[0]) if len(all_scalars) > 0 and all_scalars[0] is not None else None
        hi = float(all_scalars[1]) if len(all_scalars) > 1 and all_scalars[1] is not None else None
        if n_tensors >= 2:
            return _generic_fallback(target)

        def _clamp(a):
            return elementwise.clamp_(a, lo, hi)

        return _clamp

    if "aten.fmod" in target:
        if n_tensors >= 2:
            return elementwise.fmod_tt
        s = float(scalar_args[0]) if scalar_args else 1.0

        def _fmod_s(a):
            return elementwise.fmod_scalar(a, s)

        return _fmod_s

    if "aten.remainder" in target:
        if n_tensors >= 2:
            return elementwise.remainder_tt
        s = float(scalar_args[0]) if scalar_args else 1.0

        def _rem_s(a):
            return elementwise.remainder_scalar(a, s)

        return _rem_s

    # Comparisons: tensor-tensor or tensor-scalar
    def _cmp_scalar_baked(fn_tt, fn_s):
        if n_tensors >= 2:
            return fn_tt
        s = float(scalar_args[0]) if scalar_args else 0.0

        def _op(a):
            return fn_s(a, s)

        return _op

    if "aten.eq" in target:
        return _cmp_scalar_baked(elementwise.eq_tt, elementwise.eq_scalar)

    if "aten.ne" in target:
        return _cmp_scalar_baked(elementwise.ne_tt, elementwise.ne_scalar)

    if "aten.ge" in target:
        return _cmp_scalar_baked(elementwise.ge_tt, elementwise.ge_scalar)

    if "aten.gt" in target:
        return _cmp_scalar_baked(elementwise.gt_tt, elementwise.gt_scalar)

    if "aten.le" in target:
        return _cmp_scalar_baked(elementwise.le_tt, elementwise.le_scalar)

    if "aten.lt" in target:
        return _cmp_scalar_baked(elementwise.lt_tt, elementwise.lt_scalar)

    if "aten.isinf" in target:
        return elementwise.isinf_

    if "aten.isnan" in target:
        return elementwise.isnan_

    if "aten.logical_not" in target:
        return elementwise.logical_not

    if "aten.logical_and" in target:
        return elementwise.logical_and

    if "aten.logical_or" in target:
        return elementwise.logical_or

    if "aten.logical_xor" in target:
        return elementwise.logical_xor

    if "aten.bitwise_not" in target:
        return elementwise.bitwise_not

    if "aten.bitwise_and" in target:
        return elementwise.bitwise_and

    if "aten.bitwise_or" in target:
        return elementwise.bitwise_or

    if "aten.bitwise_xor" in target:
        return elementwise.bitwise_xor

    if "aten.maximum" in target and "aten.amax" not in target:
        return elementwise.maximum_tt

    if "aten.minimum" in target and "aten.amin" not in target:
        return elementwise.minimum_tt

    if "aten.where" in target:
        return elementwise.where

    if "aten.fill" in target:
        fill_val = float(scalar_args[0]) if scalar_args else 0.0

        def _fill(a):
            return elementwise.fill_scalar(a, fill_val)

        return _fill

    if "aten._to_copy" in target:
        out_dtype = _parse_dtype(node.get("output_dtype") or "")

        def _to_copy(a):
            return elementwise.to_copy(a, out_dtype)

        return _to_copy

    # ── Category 3: existing reductions ───────────────────────────────────────

    if "aten.sum" in target:
        return reduction.make_sum_kernel(dims, keepdim)

    if "aten.mean" in target:
        return reduction.make_mean_kernel(dims, keepdim)

    # exact match to avoid matching aten.exp
    if target == "aten._softmax.default":
        return reduction.make_softmax_kernel(dims[0] if dims else None)

    if "aten.softmax" in target:
        return reduction.make_softmax_kernel(dims[0] if dims else None)

    # ── Category 3: new reductions ────────────────────────────────────────────

    if "aten.amax" in target:
        return reduction.make_amax_kernel(dims, keepdim)

    if "aten.amin" in target:
        return reduction.make_amin_kernel(dims, keepdim)

    if "aten.max" in target and dims is not None:
        # aten.max.dim — scalar_args[0] is dim, scalar_args[1] may be keepdim
        dim_arg = int(dims[0]) if dims else (int(scalar_args[0]) if scalar_args else 0)
        return reduction.make_max_dim_kernel(dim_arg, keepdim)

    if "aten.min" in target and dims is not None:
        dim_arg = int(dims[0]) if dims else (int(scalar_args[0]) if scalar_args else 0)
        return reduction.make_min_dim_kernel(dim_arg, keepdim)

    if "aten.argmax" in target:
        dim_arg = int(dims[0]) if dims else (int(scalar_args[0]) if scalar_args else None)
        return reduction.make_argmax_kernel(dim_arg, keepdim)

    if "aten.argmin" in target:
        dim_arg = int(dims[0]) if dims else (int(scalar_args[0]) if scalar_args else None)
        return reduction.make_argmin_kernel(dim_arg, keepdim)

    if "aten.prod" in target:
        return reduction.make_prod_kernel(dims, keepdim)

    if "aten.any" in target:
        return reduction.make_any_kernel(dims, keepdim)

    if "aten.var" in target and "var_mean" not in target:
        correction = int(scalar_args[0]) if scalar_args else 1
        return reduction.make_var_kernel(dims, keepdim, correction)

    if "aten.cumsum" in target:
        dim_arg = int(dims[0]) if dims else (int(scalar_args[0]) if scalar_args else 0)
        return reduction.make_cumsum_kernel(dim_arg)

    # ── allocation op with positional-only args ────────────────────────────────

    if "aten.empty_permuted" in target:
        out_dtype = _parse_dtype(node.get("output_dtype") or "") or torch.float32
        size = list(scalar_args[0]) if scalar_args else list(output_shape or [])
        physical_layout = list(scalar_args[1]) if len(scalar_args) > 1 else list(range(len(size)))

        def _empty_permuted():
            return torch.ops.aten.empty_permuted(
                size, physical_layout, dtype=out_dtype, device="cuda"
            )

        return _empty_permuted

    # ── Category 1: view / shape ops ──────────────────────────────────────────

    if "aten.expand" in target:
        shape = list(scalar_args[0]) if scalar_args else list(output_shape or [])

        def expand_kernel(a):
            return a.expand(tuple(shape)).contiguous()

        return expand_kernel

    if "aten.unsqueeze" in target:
        dim = int(scalar_args[0]) if scalar_args else 0

        def unsqueeze_kernel(a):
            return a.unsqueeze(dim)

        return unsqueeze_kernel

    if "aten.squeeze" in target:
        dim = int(scalar_args[0]) if scalar_args else None
        if dim is not None:

            def squeeze_kernel(a):
                return a.squeeze(dim)

        else:

            def squeeze_kernel(a):
                return a.squeeze()

        return squeeze_kernel

    if "aten.view" in target or "aten._unsafe_view" in target or "aten.reshape" in target:
        shape = list(scalar_args[0]) if scalar_args else list(output_shape or [])

        def view_kernel(a):
            return a.reshape(shape).contiguous()

        return view_kernel

    if "aten.alias" in target:
        return lambda a: a.contiguous()

    if "aten.copy_" in target:
        # in-place copy: copy src values into dst; preserve dst dtype
        def _copy_(dst, src):
            return src.to(dtype=dst.dtype, device=dst.device).contiguous()

        return _copy_

    if "aten.clone" in target or "aten.copy" in target or "aten.contiguous" in target:
        return lambda a: a.contiguous().clone()

    if target == "aten.t.default":
        return lambda a: a.t().contiguous()

    if "aten.transpose" in target:
        dim0 = int(scalar_args[0]) if len(scalar_args) > 0 else 0
        dim1 = int(scalar_args[1]) if len(scalar_args) > 1 else 1

        def transpose_kernel(a):
            return a.transpose(dim0, dim1).contiguous()

        return transpose_kernel

    if "aten.permute" in target:
        dims = list(scalar_args[0]) if scalar_args else []

        def permute_kernel(a):
            return a.permute(tuple(dims)).contiguous()

        return permute_kernel

    if "aten.select" in target or "aten.slice" in target:
        return _generic_fallback(target)

    if "aten.repeat" in target:
        return _generic_fallback(target)

    # ── Category 4: GEMM — Triton kernels (mm/bmm/addmm) ─────────────────────

    if "aten.mm" in target:
        return gemm.mm

    if "aten.bmm" in target:
        return gemm.bmm

    if "aten.matmul" in target:
        # matmul dispatches through mm or bmm depending on rank at runtime.
        def _matmul(a, b):
            if a.ndim == 2 and b.ndim == 2:
                return gemm.mm(a, b)
            if a.ndim == 3 and b.ndim == 3:
                return gemm.bmm(a, b)
            return torch.matmul(a, b)

        return _matmul

    if "aten.addmm" in target:
        beta = float(scalar_args[0]) if len(scalar_args) > 0 else 1.0
        alpha = float(scalar_args[1]) if len(scalar_args) > 1 else 1.0

        def _addmm(bias, a, b):
            return gemm.addmm(bias, a, b, beta=beta, alpha=alpha)

        return _addmm

    # ── init / allocation ops ─────────────────────────────────────────────────

    if "zeros_like" in target:
        return torch.zeros_like

    if "ones_like" in target:
        return torch.ones_like

    if "full_like" in target:
        fill_val = float(scalar_args[0]) if scalar_args else 0.0

        def full_like_kernel(t):
            return torch.full_like(t, fill_val)

        return full_like_kernel

    if "aten.full" in target:
        dtype = _parse_dtype(node.get("output_dtype") or "") or torch.float32
        shape = list(scalar_args[0]) if scalar_args else list(output_shape or [])
        fill_val = scalar_args[1] if len(scalar_args) > 1 else 0.0

        def full_kernel():
            return torch.full(tuple(shape), fill_val, dtype=dtype, device="cuda")

        return full_kernel

    if "aten.scalar_tensor" in target:
        out_dtype = _parse_dtype(node.get("output_dtype") or "") or torch.float32
        val = float(scalar_args[0]) if scalar_args else 0.0

        def _scalar_tensor():
            return torch.tensor(val, dtype=out_dtype, device="cuda")

        return _scalar_tensor

    # ── scatter / gather ──────────────────────────────────────────────────────

    if "aten.scatter_add" in target:
        dim = int(scalar_args[0]) if scalar_args else 0

        def scatter_add_kernel(self_t, index, src):
            return scatter_gather.scatter_add(self_t, dim, index, src)

        return scatter_add_kernel

    if "aten.scatter" in target:
        return _generic_fallback(target)

    if "aten.gather" in target:
        dim = int(scalar_args[0]) if scalar_args else 0

        def gather_kernel(self_t, index):
            return scatter_gather.gather(self_t, dim, index)

        return gather_kernel

    # ── misc ──────────────────────────────────────────────────────────────────

    if "aten.topk" in target:
        k = int(scalar_args[0]) if scalar_args else 1
        dim_k = int(scalar_args[1]) if len(scalar_args) > 1 else -1

        def topk_kernel(a):
            return torch.topk(a, k=k, dim=dim_k)

        return topk_kernel

    if "aten._softmax" in target or "aten.log_softmax" in target:
        return _generic_fallback(target)

    if "aten.index" in target:
        return _generic_fallback(target)

    if "getitem" in target:
        idx = int(scalar_args[0]) if scalar_args else 0

        def getitem_kernel(container):
            return container[idx]

        return getitem_kernel

    return _generic_fallback(target)


def _kernel_label(fn: Callable) -> str:
    """Short label for verbose output, matching the logic in compose._kernel_label."""
    tag = getattr(fn, "_dispatch_tag", None)
    if tag:
        return tag
    module = getattr(fn, "__module__", "") or ""
    qname = getattr(fn, "__qualname__", "") or getattr(fn, "__name__", repr(fn))
    for seg in ("elementwise", "gemm", "reduction", "scatter_gather"):
        if seg in module:
            short = seg.replace("_gather", "")
            return f"triton/{short}:{qname}"
    return f"pytorch:{qname}"


def make_registry(graph: dict, verbose: bool = False) -> dict[str, Callable]:
    registry: dict[str, Callable] = {}
    if verbose:
        print(
            f"\n[AtenIR] dispatch table  ({len(graph['nodes'])} nodes)", file=sys.stderr, flush=True
        )
        print(f"  {'node':45s} {'target':50s}  kernel", file=sys.stderr, flush=True)
        print(f"  {'-'*45} {'-'*50}  {'-'*30}", file=sys.stderr, flush=True)
    for node in graph["nodes"]:
        if node.get("op") != "call_function":
            continue
        k = make_kernel(node)
        if verbose:
            print(
                f"  {node['name']:45s} {node['target']:50s}  {_kernel_label(k)}",
                file=sys.stderr,
                flush=True,
            )
        registry[node["name"]] = k
    return registry
