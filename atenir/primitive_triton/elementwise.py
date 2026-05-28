"""Shape-generic Triton elementwise kernels for primitive aten ops.

Each kernel operates on a flat 1D contiguous buffer (BLOCK=1024 fixed
constexpr, masked for tails).  Python wrappers call torch.broadcast_tensors
before dispatching so the kernel always receives same-shape contiguous inputs.

Covered ops (via Python wrapper):
  mul_tt       -- aten.mul.Tensor (tensor × tensor)
  add_tt       -- aten.add.Tensor (tensor + tensor, with optional alpha)
  sub_tt       -- aten.sub.Tensor (tensor - tensor, with optional alpha)
  div_tt       -- aten.div.Tensor (tensor / tensor)
  rsqrt        -- aten.rsqrt.default
  neg          -- aten.neg.default
  exp          -- aten.exp.default
  mul_scalar   -- aten.mul.Tensor / aten.mul.Scalar with 1 tensor + 1 scalar
  add_scalar   -- aten.add.Scalar
  div_scalar   -- aten.div.Tensor / aten.div.Scalar with 1 tensor + 1 scalar
  pow_scalar   -- aten.pow.Tensor_Scalar  via exp(b * log(a))
  pow_tt       -- aten.pow.Tensor_Tensor  via exp(b * log(a))
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

_BLOCK = 1024


if HAS_TRITON:

    @triton.jit
    def _mul_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, a * b, mask=mask)

    @triton.jit
    def _add_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, a + b, mask=mask)

    @triton.jit
    def _sub_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, a - b, mask=mask)

    @triton.jit
    def _div_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        b = tl.load(b_ptr + offs, mask=mask, other=1.0)
        tl.store(out_ptr + offs, a / b, mask=mask)

    @triton.jit
    def _rsqrt_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        tl.store(out_ptr + offs, 1.0 / tl.sqrt(a), mask=mask)

    @triton.jit
    def _neg_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, -a, mask=mask)

    @triton.jit
    def _exp_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, tl.exp(a), mask=mask)

    @triton.jit
    def _mul_scalar_kernel(a_ptr, out_ptr, scalar, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, a * scalar, mask=mask)

    @triton.jit
    def _add_scalar_kernel(a_ptr, out_ptr, scalar, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, a + scalar, mask=mask)

    @triton.jit
    def _div_scalar_kernel(a_ptr, out_ptr, scalar, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, a / scalar, mask=mask)

    @triton.jit
    def _pow_scalar_kernel(a_ptr, out_ptr, exp, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        # exp(exp * tl.log(a)) = a**exp
        tl.store(out_ptr + offs, tl.exp(exp * tl.log(a)), mask=mask)

    @triton.jit
    def _pow_tt_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, tl.exp(b * tl.log(a)), mask=mask)


def _grid(n: int):
    import math

    return (math.ceil(n / _BLOCK),)


def _bc(*tensors):
    """Broadcast all tensors to a common shape and make contiguous."""
    # pytorch default to broadcast tensors to compatible shapes.
    broadcast = torch.broadcast_tensors(*tensors)
    # triton assumes flat 1D contiguous mem.
    return tuple(t.contiguous() for t in broadcast)


# Python wrappers
# ── tensor × tensor ──────────────────────────────────────────────────────────


def mul_tt(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = _bc(a, b)
    out = torch.empty_like(a)
    _mul_kernel[_grid(a.numel())](a, b, out, N=a.numel(), BLOCK=_BLOCK)
    return out


def add_tt(a: torch.Tensor, b: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    a, b = _bc(a, b)
    out = torch.empty_like(a)
    if alpha != 1.0:
        b = mul_scalar(b, alpha)
    _add_kernel[_grid(a.numel())](a, b, out, N=a.numel(), BLOCK=_BLOCK)
    return out


def sub_tt(a: torch.Tensor, b: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    a, b = _bc(a, b)
    out = torch.empty_like(a)
    if alpha != 1.0:
        b = mul_scalar(b, alpha)
    _sub_kernel[_grid(a.numel())](a, b, out, N=a.numel(), BLOCK=_BLOCK)
    return out


def div_tt(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = _bc(a, b)
    out = torch.empty_like(a)
    _div_kernel[_grid(a.numel())](a, b, out, N=a.numel(), BLOCK=_BLOCK)
    return out


# ── single-tensor ops ─────────────────────────────────────────────────────────


def rsqrt(a: torch.Tensor) -> torch.Tensor:
    a = a.contiguous()
    out = torch.empty_like(a)
    _rsqrt_kernel[_grid(a.numel())](a, out, N=a.numel(), BLOCK=_BLOCK)
    return out


def neg(a: torch.Tensor) -> torch.Tensor:
    a = a.contiguous()
    out = torch.empty_like(a)
    _neg_kernel[_grid(a.numel())](a, out, N=a.numel(), BLOCK=_BLOCK)
    return out


def exp(a: torch.Tensor) -> torch.Tensor:
    a = a.contiguous()
    out = torch.empty_like(a)
    _exp_kernel[_grid(a.numel())](a, out, N=a.numel(), BLOCK=_BLOCK)
    return out


# ── tensor × scalar ───────────────────────────────────────────────────────────


def mul_scalar(a: torch.Tensor, scalar) -> torch.Tensor:
    a = a.contiguous()
    out = torch.empty_like(a)
    _mul_scalar_kernel[_grid(a.numel())](a, out, float(scalar), N=a.numel(), BLOCK=_BLOCK)
    return out


def add_scalar(a: torch.Tensor, scalar) -> torch.Tensor:
    a = a.contiguous()
    out = torch.empty_like(a)
    _add_scalar_kernel[_grid(a.numel())](a, out, float(scalar), N=a.numel(), BLOCK=_BLOCK)
    return out


def div_scalar(a: torch.Tensor, scalar) -> torch.Tensor:
    a = a.contiguous()
    out = torch.empty_like(a)
    _div_scalar_kernel[_grid(a.numel())](a, out, float(scalar), N=a.numel(), BLOCK=_BLOCK)
    return out


# ── pow ───────────────────────────────────────────────────────────────────────


def pow_scalar(a: torch.Tensor, exp) -> torch.Tensor:
    a = a.contiguous()
    exp_f = float(exp)
    if exp_f == 0.0:
        return torch.ones_like(a)
    if exp_f == 1.0:
        return a.clone()
    if exp_f == 2.0:
        return mul_tt(a, a)
    if exp_f == 3.0:
        return mul_tt(mul_tt(a, a), a)
    out = torch.empty_like(a, dtype=torch.float32)
    _pow_scalar_kernel[_grid(a.numel())](a, out, exp_f, N=a.numel(), BLOCK=_BLOCK)
    return out.to(a.dtype)


def pow_tt(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = _bc(a, b)
    out = torch.empty_like(a, dtype=torch.float32)
    _pow_tt_kernel[_grid(a.numel())](a, b, out, N=a.numel(), BLOCK=_BLOCK)
    return out.to(a.dtype)
