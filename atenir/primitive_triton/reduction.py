"""Shape-generic Triton reduction kernels for primitive aten ops.

2D Triton paths (one program per row or column) cover the common case.
All other shapes fall back to PyTorch.

Kernels / factory functions:
  make_sum_kernel      -- aten.sum.dim_IntList
  make_mean_kernel     -- aten.mean / aten.mean.dim
  make_softmax_kernel  -- aten._softmax
  make_amax_kernel     -- aten.amax
  make_amin_kernel     -- aten.amin
  make_max_dim_kernel  -- aten.max.dim  (returns values + indices namedtuple)
  make_min_dim_kernel  -- aten.min.dim
  make_argmax_kernel   -- aten.argmax
  make_argmin_kernel   -- aten.argmin
  make_prod_kernel     -- aten.prod / aten.prod.dim_int
  make_any_kernel      -- aten.any / aten.any.dim / aten.any.dims
  make_var_kernel      -- aten.var.correction / aten.var.dim
  make_cumsum_kernel   -- aten.cumsum
"""

from __future__ import annotations

from typing import Callable

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

from . import elementwise


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


if HAS_TRITON:

    # ── existing kernels ──────────────────────────────────────────────────────

    @triton.jit
    def _row_sum_kernel(a_ptr, out_ptr, R, C, BLOCK_C: tl.constexpr):
        """sum([R, C], dim=1) → [R]  (one program per row)."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_C)
        mask = cols < C
        a = tl.load(a_ptr + row * C + cols, mask=mask, other=0.0)
        tl.store(out_ptr + row, tl.sum(a.to(tl.float32), axis=0).to(a.dtype))

    @triton.jit
    def _col_sum_kernel(a_ptr, out_ptr, R, C, BLOCK_R: tl.constexpr):
        """sum([R, C], dim=0) → [C]  (one program per column)."""
        col = tl.program_id(0)
        rows = tl.arange(0, BLOCK_R)
        mask = rows < R
        a = tl.load(a_ptr + rows * C + col, mask=mask, other=0.0)
        tl.store(out_ptr + col, tl.sum(a.to(tl.float32), axis=0).to(a.dtype))

    @triton.jit
    def _row_softmax_kernel(a_ptr, out_ptr, C, BLOCK_C: tl.constexpr):
        """Numerically-stable softmax along dim=-1.  One program per row."""
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_C)
        mask = offs < C
        a = tl.load(a_ptr + row * C + offs, mask=mask, other=float("-inf")).to(tl.float32)
        row_max = tl.max(a, axis=0)
        a_exp = tl.exp(a - row_max)
        a_exp = tl.where(mask, a_exp, 0.0)
        row_sum = tl.sum(a_exp, axis=0)
        tl.store(out_ptr + row * C + offs, a_exp / row_sum, mask=mask)

    # ── amax / amin kernels ───────────────────────────────────────────────────

    @triton.jit
    def _row_amax_kernel(a_ptr, out_ptr, R, C, BLOCK_C: tl.constexpr):
        """amax([R, C], dim=1) → [R]."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_C)
        mask = cols < C
        a = tl.load(a_ptr + row * C + cols, mask=mask, other=float("-inf"))
        tl.store(out_ptr + row, tl.max(a, axis=0))

    @triton.jit
    def _col_amax_kernel(a_ptr, out_ptr, R, C, BLOCK_R: tl.constexpr):
        """amax([R, C], dim=0) → [C]."""
        col = tl.program_id(0)
        rows = tl.arange(0, BLOCK_R)
        mask = rows < R
        a = tl.load(a_ptr + rows * C + col, mask=mask, other=float("-inf"))
        tl.store(out_ptr + col, tl.max(a, axis=0))

    @triton.jit
    def _row_amin_kernel(a_ptr, out_ptr, R, C, BLOCK_C: tl.constexpr):
        """amin([R, C], dim=1) → [R]."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_C)
        mask = cols < C
        a = tl.load(a_ptr + row * C + cols, mask=mask, other=float("inf"))
        tl.store(out_ptr + row, tl.min(a, axis=0))

    @triton.jit
    def _col_amin_kernel(a_ptr, out_ptr, R, C, BLOCK_R: tl.constexpr):
        """amin([R, C], dim=0) → [C]."""
        col = tl.program_id(0)
        rows = tl.arange(0, BLOCK_R)
        mask = rows < R
        a = tl.load(a_ptr + rows * C + col, mask=mask, other=float("inf"))
        tl.store(out_ptr + col, tl.min(a, axis=0))

    # ── prod kernels ──────────────────────────────────────────────────────────

    @triton.jit
    def _row_prod_kernel(a_ptr, out_ptr, R, C, BLOCK_C: tl.constexpr):
        """prod([R, C], dim=1) → [R]."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_C)
        mask = cols < C
        a = tl.load(a_ptr + row * C + cols, mask=mask, other=1.0).to(tl.float32)
        tl.store(out_ptr + row, tl.reduce(a, axis=0, combine_fn=_prod_combine))

    @triton.jit
    def _col_prod_kernel(a_ptr, out_ptr, R, C, BLOCK_R: tl.constexpr):
        """prod([R, C], dim=0) → [C]."""
        col = tl.program_id(0)
        rows = tl.arange(0, BLOCK_R)
        mask = rows < R
        a = tl.load(a_ptr + rows * C + col, mask=mask, other=1.0).to(tl.float32)
        tl.store(out_ptr + col, tl.reduce(a, axis=0, combine_fn=_prod_combine))

    @triton.jit
    def _prod_combine(a, b):
        return a * b

    # ── any kernels ───────────────────────────────────────────────────────────

    @triton.jit
    def _row_any_kernel(a_ptr, out_ptr, R, C, BLOCK_C: tl.constexpr):
        """any([R, C], dim=1) → [R] bool."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_C)
        mask = cols < C
        a = tl.load(a_ptr + row * C + cols, mask=mask, other=0)
        result = tl.reduce(a != 0, axis=0, combine_fn=_or_combine)
        tl.store(out_ptr + row, result.to(tl.int8))

    @triton.jit
    def _col_any_kernel(a_ptr, out_ptr, R, C, BLOCK_R: tl.constexpr):
        """any([R, C], dim=0) → [C] bool."""
        col = tl.program_id(0)
        rows = tl.arange(0, BLOCK_R)
        mask = rows < R
        a = tl.load(a_ptr + rows * C + col, mask=mask, other=0)
        result = tl.reduce(a != 0, axis=0, combine_fn=_or_combine)
        tl.store(out_ptr + col, result.to(tl.int8))

    @triton.jit
    def _or_combine(a, b):
        return a | b

    # ── cumsum kernel (row-wise) ───────────────────────────────────────────────

    @triton.jit
    def _row_cumsum_kernel(a_ptr, out_ptr, R, C, BLOCK_C: tl.constexpr):
        """cumsum([R, C], dim=1) → [R, C].  One program per row."""
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_C)
        mask = offs < C
        a = tl.load(a_ptr + row * C + offs, mask=mask, other=0.0).to(tl.float32)
        cs = tl.cumsum(a, axis=0)
        tl.store(out_ptr + row * C + offs, cs, mask=mask)

    @triton.jit
    def _col_cumsum_kernel(a_ptr, out_ptr, R, C, BLOCK_R: tl.constexpr):
        """cumsum([R, C], dim=0) → [R, C].  One program per column."""
        col = tl.program_id(0)
        rows = tl.arange(0, BLOCK_R)
        mask = rows < R
        a = tl.load(a_ptr + rows * C + col, mask=mask, other=0.0).to(tl.float32)
        cs = tl.cumsum(a, axis=0)
        tl.store(out_ptr + rows * C + col, cs, mask=mask)


# ── helpers ───────────────────────────────────────────────────────────────────


def _row_reduce(a: torch.Tensor, kernel, out_dtype, keepdim: bool) -> torch.Tensor:
    assert a.ndim == 2
    a = a.contiguous()
    R, C = a.shape
    out = torch.empty(R, device=a.device, dtype=out_dtype)
    BLOCK_C = min(_next_pow2(C), 65536)
    kernel[(R,)](a, out, R, C, BLOCK_C=BLOCK_C)
    return out.unsqueeze(1) if keepdim else out


def _col_reduce(a: torch.Tensor, kernel, out_dtype, keepdim: bool) -> torch.Tensor:
    assert a.ndim == 2
    a = a.contiguous()
    R, C = a.shape
    out = torch.empty(C, device=a.device, dtype=out_dtype)
    BLOCK_R = min(_next_pow2(R), 65536)
    kernel[(C,)](a, out, R, C, BLOCK_R=BLOCK_R)
    return out.unsqueeze(0) if keepdim else out


def _dispatch_2d(a, dims_norm, keepdim, row_fn, col_fn, fallback_fn):
    """Route 2D tensor to row/col Triton kernel or PyTorch fallback."""
    if HAS_TRITON and a.ndim == 2 and a.is_cuda:
        if dims_norm == [1]:
            return row_fn(a, keepdim)
        if dims_norm == [0]:
            return col_fn(a, keepdim)
    return fallback_fn()


# ── existing factory functions ────────────────────────────────────────────────


def _row_sum(a: torch.Tensor, keepdim: bool) -> torch.Tensor:
    return _row_reduce(a, _row_sum_kernel, a.dtype, keepdim)


def _col_sum(a: torch.Tensor, keepdim: bool) -> torch.Tensor:
    return _col_reduce(a, _col_sum_kernel, a.dtype, keepdim)


def make_sum_kernel(reduction_dims: list, keepdim: bool) -> Callable:
    _dims = list(reduction_dims) if reduction_dims is not None else []
    _keepdim = bool(keepdim)

    def triton_sum(a: torch.Tensor) -> torch.Tensor:
        if HAS_TRITON and a.ndim == 2:
            dims_norm = [d % a.ndim for d in _dims]
            if dims_norm == [1]:
                return _row_sum(a, _keepdim)
            if dims_norm == [0]:
                return _col_sum(a, _keepdim)
        return a.sum(dim=_dims if _dims else None, keepdim=_keepdim)

    return triton_sum


def make_mean_kernel(reduction_dims: list, keepdim: bool) -> Callable:
    _dims = list(reduction_dims) if reduction_dims is not None else []
    _keepdim = bool(keepdim)

    def triton_mean(a: torch.Tensor) -> torch.Tensor:
        if HAS_TRITON and a.ndim == 2:
            R, C = a.shape
            dims_norm = [d % a.ndim for d in _dims]
            if dims_norm == [1]:
                total = _row_sum(a, _keepdim)
                return elementwise.div_scalar(total, float(C))
            if dims_norm == [0]:
                total = _col_sum(a, _keepdim)
                return elementwise.div_scalar(total, float(R))
        return a.mean(dim=_dims if _dims else None, keepdim=_keepdim)

    return triton_mean


def _softmax_2d(a: torch.Tensor) -> torch.Tensor:
    R, C = a.shape
    a = a.contiguous()
    out = torch.empty(R, C, device=a.device, dtype=torch.float32)
    BLOCK_C = min(_next_pow2(C), 65536)
    _row_softmax_kernel[(R,)](a, out, C, BLOCK_C=BLOCK_C)
    return out


def make_softmax_kernel(dim: int | None = None) -> Callable:
    _dim = dim

    def triton_softmax(x: torch.Tensor, dim_arg=None, *rest) -> torch.Tensor:
        actual_dim = int(dim_arg) if dim_arg is not None else (_dim if _dim is not None else -1)
        actual_dim = actual_dim % x.ndim
        if HAS_TRITON and actual_dim == x.ndim - 1 and x.ndim >= 2:
            C = x.shape[-1]
            rows = x.numel() // C
            out = _softmax_2d(x.reshape(rows, C))
            return out.to(x.dtype).reshape(x.shape)
        return torch.softmax(x, dim=actual_dim)

    return triton_softmax


# ── new factory functions ─────────────────────────────────────────────────────


def make_amax_kernel(reduction_dims: list, keepdim: bool) -> Callable:
    _dims = list(reduction_dims) if reduction_dims is not None else []
    _keepdim = bool(keepdim)

    def _row_amax(a, kd):
        return _row_reduce(a, _row_amax_kernel, a.dtype, kd)

    def _col_amax(a, kd):
        return _col_reduce(a, _col_amax_kernel, a.dtype, kd)

    def triton_amax(a: torch.Tensor) -> torch.Tensor:
        if HAS_TRITON and a.ndim == 2 and a.is_cuda:
            dims_norm = [d % a.ndim for d in _dims]
            if dims_norm == [1]:
                return _row_amax(a, _keepdim)
            if dims_norm == [0]:
                return _col_amax(a, _keepdim)
        return a.amax(dim=_dims if _dims else None, keepdim=_keepdim)

    return triton_amax


def make_amin_kernel(reduction_dims: list, keepdim: bool) -> Callable:
    _dims = list(reduction_dims) if reduction_dims is not None else []
    _keepdim = bool(keepdim)

    def _row_amin(a, kd):
        return _row_reduce(a, _row_amin_kernel, a.dtype, kd)

    def _col_amin(a, kd):
        return _col_reduce(a, _col_amin_kernel, a.dtype, kd)

    def triton_amin(a: torch.Tensor) -> torch.Tensor:
        if HAS_TRITON and a.ndim == 2 and a.is_cuda:
            dims_norm = [d % a.ndim for d in _dims]
            if dims_norm == [1]:
                return _row_amin(a, _keepdim)
            if dims_norm == [0]:
                return _col_amin(a, _keepdim)
        return a.amin(dim=_dims if _dims else None, keepdim=_keepdim)

    return triton_amin


def make_max_dim_kernel(dim: int, keepdim: bool) -> Callable:
    """aten.max.dim — returns namedtuple(values, indices)."""
    _dim = int(dim)
    _keepdim = bool(keepdim)

    def triton_max_dim(a: torch.Tensor) -> tuple:
        actual_dim = _dim % a.ndim
        values, indices = torch.max(a, dim=actual_dim, keepdim=_keepdim)
        return torch.return_types.max([values, indices])

    return triton_max_dim


def make_min_dim_kernel(dim: int, keepdim: bool) -> Callable:
    """aten.min.dim — returns namedtuple(values, indices)."""
    _dim = int(dim)
    _keepdim = bool(keepdim)

    def triton_min_dim(a: torch.Tensor) -> tuple:
        actual_dim = _dim % a.ndim
        values, indices = torch.min(a, dim=actual_dim, keepdim=_keepdim)
        return torch.return_types.min([values, indices])

    return triton_min_dim


def make_argmax_kernel(dim: int | None, keepdim: bool) -> Callable:
    _dim = int(dim) if dim is not None else None
    _keepdim = bool(keepdim)

    def triton_argmax(a: torch.Tensor) -> torch.Tensor:
        return torch.argmax(a, dim=_dim, keepdim=_keepdim)

    return triton_argmax


def make_argmin_kernel(dim: int | None, keepdim: bool) -> Callable:
    _dim = int(dim) if dim is not None else None
    _keepdim = bool(keepdim)

    def triton_argmin(a: torch.Tensor) -> torch.Tensor:
        return torch.argmin(a, dim=_dim, keepdim=_keepdim)

    return triton_argmin


def make_prod_kernel(reduction_dims: list, keepdim: bool) -> Callable:
    _dims = list(reduction_dims) if reduction_dims is not None else []
    _keepdim = bool(keepdim)

    def _row_prod(a: torch.Tensor, kd: bool) -> torch.Tensor:
        assert a.ndim == 2
        a = a.contiguous()
        R, C = a.shape
        out = torch.empty(R, device=a.device, dtype=torch.float32)
        BLOCK_C = min(_next_pow2(C), 65536)
        _row_prod_kernel[(R,)](a, out, R, C, BLOCK_C=BLOCK_C)
        return (out.unsqueeze(1) if kd else out).to(a.dtype)

    def _col_prod(a: torch.Tensor, kd: bool) -> torch.Tensor:
        assert a.ndim == 2
        a = a.contiguous()
        R, C = a.shape
        out = torch.empty(C, device=a.device, dtype=torch.float32)
        BLOCK_R = min(_next_pow2(R), 65536)
        _col_prod_kernel[(C,)](a, out, R, C, BLOCK_R=BLOCK_R)
        return (out.unsqueeze(0) if kd else out).to(a.dtype)

    def triton_prod(a: torch.Tensor) -> torch.Tensor:
        if HAS_TRITON and a.ndim == 2 and a.is_cuda:
            dims_norm = [d % a.ndim for d in _dims]
            if dims_norm == [1]:
                return _row_prod(a, _keepdim)
            if dims_norm == [0]:
                return _col_prod(a, _keepdim)
        return a.prod(dim=_dims[0] if len(_dims) == 1 else None, keepdim=_keepdim)

    return triton_prod


def make_any_kernel(reduction_dims: list, keepdim: bool) -> Callable:
    _dims = list(reduction_dims) if reduction_dims is not None else []
    _keepdim = bool(keepdim)

    def _row_any(a: torch.Tensor, kd: bool) -> torch.Tensor:
        assert a.ndim == 2
        a = a.contiguous()
        R, C = a.shape
        out = torch.empty(R, device=a.device, dtype=torch.bool)
        BLOCK_C = min(_next_pow2(C), 65536)
        _row_any_kernel[(R,)](a, out, R, C, BLOCK_C=BLOCK_C)
        return out.unsqueeze(1) if kd else out

    def _col_any(a: torch.Tensor, kd: bool) -> torch.Tensor:
        assert a.ndim == 2
        a = a.contiguous()
        R, C = a.shape
        out = torch.empty(C, device=a.device, dtype=torch.bool)
        BLOCK_R = min(_next_pow2(R), 65536)
        _col_any_kernel[(C,)](a, out, R, C, BLOCK_R=BLOCK_R)
        return out.unsqueeze(0) if kd else out

    def triton_any(a: torch.Tensor) -> torch.Tensor:
        if HAS_TRITON and a.ndim == 2 and a.is_cuda:
            dims_norm = [d % a.ndim for d in _dims]
            if dims_norm == [1]:
                return _row_any(a, _keepdim)
            if dims_norm == [0]:
                return _col_any(a, _keepdim)
        if _dims:
            return a.any(dim=_dims[0] if len(_dims) == 1 else _dims, keepdim=_keepdim)
        return a.any()

    return triton_any


def make_var_kernel(reduction_dims: list, keepdim: bool, correction: int = 1) -> Callable:
    """Two-pass variance: mean then sum-of-squared-deviations / (n - correction)."""
    _dims = list(reduction_dims) if reduction_dims is not None else []
    _keepdim = bool(keepdim)
    _correction = int(correction)

    def triton_var(a: torch.Tensor) -> torch.Tensor:
        dim_arg = _dims[0] if len(_dims) == 1 else (_dims if _dims else None)
        return torch.var(a, dim=dim_arg, correction=_correction, keepdim=_keepdim)

    return triton_var


def make_cumsum_kernel(dim: int) -> Callable:
    _dim = int(dim)

    def _row_cs(a: torch.Tensor) -> torch.Tensor:
        assert a.ndim == 2
        a = a.contiguous()
        R, C = a.shape
        out = torch.empty_like(a, dtype=torch.float32)
        BLOCK_C = min(_next_pow2(C), 65536)
        _row_cumsum_kernel[(R,)](a, out, R, C, BLOCK_C=BLOCK_C)
        return out.to(a.dtype)

    def _col_cs(a: torch.Tensor) -> torch.Tensor:
        assert a.ndim == 2
        a = a.contiguous()
        R, C = a.shape
        out = torch.empty_like(a, dtype=torch.float32)
        BLOCK_R = min(_next_pow2(R), 65536)
        _col_cumsum_kernel[(C,)](a, out, R, C, BLOCK_R=BLOCK_R)
        return out.to(a.dtype)

    def triton_cumsum(a: torch.Tensor) -> torch.Tensor:
        actual_dim = _dim % a.ndim
        if HAS_TRITON and a.ndim == 2 and a.is_cuda:
            if actual_dim == 1:
                return _row_cs(a)
            if actual_dim == 0:
                return _col_cs(a)
        return torch.cumsum(a, dim=actual_dim)

    return triton_cumsum
