"""Shape-generic Triton reduction kernels for sum, mean, and softmax.

Kernels:
  _row_sum_kernel    -- sum([R,C], dim=-1/1)  → [R]
  _col_sum_kernel    -- sum([R,C], dim=0)     → [C]
  _row_softmax_kernel -- softmax([R,C], dim=-1) → [R,C]  (numerically stable)

make_sum_kernel / make_mean_kernel / make_softmax_kernel return closures that
dispatch to the Triton path for 2D inputs and fall back to PyTorch otherwise.
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

if HAS_TRITON:

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
        # load; pad out-of-bounds with -inf so they don't affect max/sum
        a = tl.load(a_ptr + row * C + offs, mask=mask, other=float("-inf")).to(tl.float32)
        row_max = tl.max(a, axis=0)
        # calculate exp
        a_exp = tl.exp(a - row_max)
        # zero out padding lanes before summing
        a_exp = tl.where(mask, a_exp, 0.0)
        row_sum = tl.sum(a_exp, axis=0)
        tl.store(out_ptr + row * C + offs, a_exp / row_sum, mask=mask)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _row_sum(a: torch.Tensor, keepdim: bool) -> torch.Tensor:
    assert a.ndim == 2
    a = a.contiguous()
    R, C = a.shape
    out = torch.empty(R, device=a.device, dtype=a.dtype)
    BLOCK_C = min(_next_pow2(C), 65536)
    _row_sum_kernel[(R,)](a, out, R, C, BLOCK_C=BLOCK_C)
    return out.unsqueeze(1) if keepdim else out


def _col_sum(a: torch.Tensor, keepdim: bool) -> torch.Tensor:
    assert a.ndim == 2
    a = a.contiguous()
    R, C = a.shape
    out = torch.empty(C, device=a.device, dtype=a.dtype)
    BLOCK_R = min(_next_pow2(R), 65536)
    _col_sum_kernel[(C,)](a, out, R, C, BLOCK_R=BLOCK_R)
    return out.unsqueeze(0) if keepdim else out


def make_sum_kernel(reduction_dims: list, keepdim: bool) -> Callable:
    """Return a kernel closure for sum along the given dims.

    Triton paths cover 2D tensors reduced along dim 0 or dim -1/1.
    All other cases fall through to PyTorch.
    """
    _dims = list(reduction_dims) if reduction_dims is not None else []
    _keepdim = bool(keepdim)

    def triton_sum(a: torch.Tensor) -> torch.Tensor:
        if HAS_TRITON and a.ndim == 2:
            R, C = a.shape
            # Normalise negative dims
            dims_norm = [d % a.ndim for d in _dims]
            if dims_norm == [1]:
                return _row_sum(a, _keepdim)
            if dims_norm == [0]:
                return _col_sum(a, _keepdim)
        # Fallback: PyTorch handles N-D or multi-dim reductions
        return a.sum(dim=_dims if _dims else None, keepdim=_keepdim)

    return triton_sum


def make_mean_kernel(reduction_dims: list, keepdim: bool) -> Callable:
    """Return a kernel closure for mean along the given dims.

    Reuses the sum kernels and divides by the reduced dimension size via a
    Triton div_scalar kernel.  Falls back to PyTorch for non-2D inputs.
    """
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
    """Triton row-softmax on a 2D contiguous float tensor; output is float32."""
    R, C = a.shape
    a = a.contiguous()
    out = torch.empty(R, C, device=a.device, dtype=torch.float32)
    BLOCK_C = min(_next_pow2(C), 65536)
    _row_softmax_kernel[(R,)](a, out, C, BLOCK_C=BLOCK_C)
    return out


def make_softmax_kernel(dim: int | None = None) -> Callable:
    """Return a kernel closure for softmax.

    Triton path handles last-dim softmax for 2D and any-rank tensors by folding
    outer dims into rows.  Falls back to PyTorch for other dims.

    The returned closure matches the AtenIR calling convention:
      fn(x, dim_scalar, *rest)
    where dim_scalar is the runtime-provided dim argument.
    """
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
