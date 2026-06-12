"""Build an AtenIR kernel registry from a graph JSON.

``make_registry(graph)`` walks every call_function node and assigns a kernel:
  - Arithmetic ops → Triton kernels in elementwise / reduction modules.
  - View / shape ops (expand, unsqueeze, view, permute, t, …) → PyTorch, with
    the target output_shape captured in a closure so that args dropped by the
    AtenIR JSON schema (list-valued args for expand / view) can be recovered.
  - Complex ops (mm, bmm, scatter_add, pow, topk, …) → PyTorch fallback.

The returned dict is keyed by node name and maps directly to compose.run_graph's
``kernel_registry`` parameter.
"""

from __future__ import annotations

from typing import Callable

import torch

from . import elementwise, reduction, scatter_gather


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

        return fn

    def fn(*args):
        return op(*args)

    return fn


def make_kernel(node: dict) -> Callable:
    target: str = node["target"]
    n_tensors = len(node.get("predecessor_ids") or [])
    dims = node.get("reduction_dims")
    keepdim = bool(node.get("keepdim") or False)
    output_shape = node.get("output_shape")

    # elementwise
    if "aten.mul" in target:
        return elementwise.mul_tt if n_tensors >= 2 else elementwise.mul_scalar

    # "aten.add." (trailing dot) so this does NOT also match "aten.addmm.default";
    # addmm is handled in the matmul group below.
    if "aten.add." in target:
        if n_tensors >= 2:

            def add_with_alpha(a, b, *rest):
                alpha = float(rest[0]) if rest else 1.0
                return elementwise.add_tt(a, b, alpha)

            return add_with_alpha
        return elementwise.add_scalar

    if "aten.sub" in target:
        if n_tensors >= 2:

            def sub_with_alpha(a, b, *rest):
                alpha = float(rest[0]) if rest else 1.0
                return elementwise.sub_tt(a, b, alpha)

            return sub_with_alpha

        def sub_scalar(*args):
            # tensor - scalar
            if isinstance(args[0], torch.Tensor):
                a, s = args[0], args[1]
                return elementwise.add_scalar(a, -float(s))

            # scalar - tensor
            s, a = args[0], args[1]
            return elementwise.add_scalar(elementwise.neg(a), float(s))

        return sub_scalar

    if "aten.div" in target:
        return elementwise.div_tt if n_tensors >= 2 else elementwise.div_scalar

    if "aten.rsqrt" in target:
        return elementwise.rsqrt

    if "aten.neg" in target:
        return elementwise.neg

    # exact match, otherwise aten.expand also matches "aten.exp"
    if target == "aten.exp.default":
        return elementwise.exp

    # reductions
    if "aten.sum" in target:
        return reduction.make_sum_kernel(dims, keepdim)

    if "aten.mean" in target:
        return reduction.make_mean_kernel(dims, keepdim)

    if "aten.pow" in target:
        if n_tensors >= 2:
            return elementwise.pow_tt
        # aten.pow.Scalar(base_scalar, tensor_exp) means base**tensor — args arrive
        # in correct order now (base first), so we need a different path.
        if "Scalar_Tensor" in target or (
            "Scalar" in target and "Tensor_Scalar" not in target and "Tensor" not in target
        ):

            def pow_base_scalar(base, tensor_exp):
                import math

                log_base = math.log(float(base))
                return elementwise.exp(
                    elementwise.mul_scalar(tensor_exp.to(torch.float32), log_base)
                )

            return pow_base_scalar
        return elementwise.pow_scalar

    # shape/view ops
    if "aten.expand" in target:

        def expand_kernel(a, shape=None):
            if shape is None:
                shape = output_shape
            return a.expand(tuple(shape)).contiguous()

        return expand_kernel

    if "aten.unsqueeze" in target:

        def unsqueeze_kernel(a, dim):
            return a.unsqueeze(int(dim))

        return unsqueeze_kernel

    if "aten.squeeze" in target:

        def squeeze_kernel(a, *args):
            return a.squeeze(int(args[0])) if args else a.squeeze()

        return squeeze_kernel

    if "aten.view" in target or "aten._unsafe_view" in target or "aten.reshape" in target:

        def view_kernel(a, shape=None):
            if shape is None:
                shape = output_shape
            return a.reshape(tuple(shape)).contiguous()

        return view_kernel

    if "aten.alias" in target:
        return lambda a: a.contiguous()

    if "aten.clone" in target or "aten.copy" in target or "aten.contiguous" in target:
        return lambda a: a.contiguous().clone()

    if target == "aten.t.default":
        return lambda a: a.t().contiguous()

    if "aten.transpose" in target:

        def transpose_kernel(a, dim0, dim1):
            return a.transpose(int(dim0), int(dim1)).contiguous()

        return transpose_kernel

    if "aten.permute" in target:

        def permute_kernel(a, dims):
            return a.permute(tuple(dims)).contiguous()

        return permute_kernel

    if "aten.select" in target or "aten.slice" in target:
        return _generic_fallback(target)

    # matmul
    if "aten.addmm" in target:

        def addmm_kernel(bias, mat1, mat2, *rest):
            return torch.addmm(bias, mat1, mat2)

        return addmm_kernel

    if "aten.mm" in target:
        return torch.mm

    if "aten.bmm" in target:
        return torch.bmm

    if "aten.matmul" in target:
        return torch.matmul

    # scatter / gather
    if "aten.scatter_add" in target:

        def scatter_add_kernel(self_t, dim, index, src):
            return scatter_gather.scatter_add(self_t, int(dim), index, src)

        return scatter_add_kernel

    if "aten.scatter" in target:
        return _generic_fallback(target)

    if "aten.gather" in target:

        def gather_kernel(self_t, dim, index):
            return scatter_gather.gather(self_t, int(dim), index)

        return gather_kernel

    # init ops
    if "zeros_like" in target:
        return torch.zeros_like

    if "ones_like" in target:
        return torch.ones_like

    if "full_like" in target:

        def full_like_kernel(t, fill_val=0.0):
            return torch.full_like(t, float(fill_val))

        return full_like_kernel

    if "aten.full" in target:
        dtype = _parse_dtype(node.get("output_dtype") or "") or torch.float32

        def full_kernel(shape, fill_val, *rest):
            return torch.full(
                tuple(shape),
                fill_val,
                dtype=dtype,
                device="cuda",
            )

        return full_kernel

    if "aten.topk" in target:

        def topk_kernel(a, k):
            return torch.topk(a, k=int(k), dim=-1)

        return topk_kernel

    # softmax
    if "aten._softmax" in target or "aten.softmax" in target:
        return reduction.make_softmax_kernel(dims[0] if dims else None)

    if "aten.index" in target or "aten.log_softmax" in target:
        return _generic_fallback(target)

    # tuple indexing
    if "getitem" in target:

        def getitem_kernel(container, index):
            return container[int(index)]

        return getitem_kernel

    return _generic_fallback(target)


def make_registry(graph: dict) -> dict[str, Callable]:
    registry: dict[str, Callable] = {}
    for node in graph["nodes"]:
        if node.get("op") != "call_function":
            continue
        registry[node["name"]] = make_kernel(node)
    return registry


def _parse_dtype(dtype_str: str):
    return {
        "torch.float32": torch.float32,
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.int64": torch.int64,
        "torch.int32": torch.int32,
        "torch.bool": torch.bool,
    }.get(dtype_str)
