"""Shape-generic Triton elementwise kernels for primitive aten ops.

Each kernel operates on a flat 1D contiguous buffer (BLOCK=1024 fixed
constexpr, masked for tails). Python wrappers call torch.broadcast_tensors
before dispatching so the kernel always receives same-shape contiguous inputs.

Transcendental functions use triton.language.extra.cuda.libdevice (Triton 3.x
naming, no 'f' suffix). All transcendental wrappers compute in float32 and
cast back to the original dtype.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    import triton.language.extra.cuda.libdevice as libdevice

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

_BLOCK = 1024


# ── Triton JIT kernels ────────────────────────────────────────────────────────

if HAS_TRITON:

    # ── existing binary tt ────────────────────────────────────────────────────

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

    # ── existing unary ────────────────────────────────────────────────────────

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
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, tl.exp(a), mask=mask)

    # ── existing scalar ───────────────────────────────────────────────────────

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
        tl.store(out_ptr + offs, tl.exp(exp * tl.log(a)), mask=mask)

    @triton.jit
    def _pow_tt_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, tl.exp(b * tl.log(a)), mask=mask)

    # ── new unary: simple tl ops ──────────────────────────────────────────────

    @triton.jit
    def _abs_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, tl.abs(a), mask=mask)

    @triton.jit
    def _sqrt_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, tl.sqrt(a), mask=mask)

    @triton.jit
    def _reciprocal_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        tl.store(out_ptr + offs, 1.0 / a, mask=mask)

    @triton.jit
    def _sign_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        result = tl.where(a > 0.0, 1.0, tl.where(a < 0.0, -1.0, 0.0))
        tl.store(out_ptr + offs, result.to(a.dtype), mask=mask)

    @triton.jit
    def _sin_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, tl.sin(a), mask=mask)

    @triton.jit
    def _cos_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, tl.cos(a), mask=mask)

    @triton.jit
    def _log_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        tl.store(out_ptr + offs, tl.log(a), mask=mask)

    @triton.jit
    def _log2_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        # log2(x) = log(x) / log(2)
        tl.store(out_ptr + offs, tl.log(a) * 1.4426950408889634, mask=mask)

    @triton.jit
    def _log10_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        # log10(x) = log(x) / log(10)
        tl.store(out_ptr + offs, tl.log(a) * 0.4342944819032518, mask=mask)

    @triton.jit
    def _sigmoid_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, tl.sigmoid(a), mask=mask)

    # tanh via identity: tanh(x) = 2*sigmoid(2x) - 1
    @triton.jit
    def _tanh_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, 2.0 * tl.sigmoid(2.0 * a) - 1.0, mask=mask)

    # sinh(x) = (exp(x) - exp(-x)) / 2
    @triton.jit
    def _sinh_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, (tl.exp(a) - tl.exp(-a)) * 0.5, mask=mask)

    # cosh(x) = (exp(x) + exp(-x)) / 2
    @triton.jit
    def _cosh_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, (tl.exp(a) + tl.exp(-a)) * 0.5, mask=mask)

    @triton.jit
    def _relu_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, tl.where(a > 0.0, a, 0.0).to(a.dtype), mask=mask)

    # expm1(x) = exp(x) - 1
    @triton.jit
    def _expm1_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, tl.exp(a) - 1.0, mask=mask)

    # log1p(x) = log(1 + x)
    @triton.jit
    def _log1p_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, tl.log(1.0 + a), mask=mask)

    # ── new unary: transcendentals ────────────────────────────────────────────

    @triton.jit
    def _tan_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, libdevice.tan(a), mask=mask)

    @triton.jit
    def _atan_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, libdevice.atan(a), mask=mask)

    @triton.jit
    def _asin_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, libdevice.asin(a), mask=mask)

    @triton.jit
    def _acos_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, libdevice.acos(a), mask=mask)

    @triton.jit
    def _asinh_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, libdevice.asinh(a), mask=mask)

    @triton.jit
    def _acosh_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        tl.store(out_ptr + offs, libdevice.acosh(a), mask=mask)

    @triton.jit
    def _atanh_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, libdevice.atanh(a), mask=mask)

    @triton.jit
    def _erf_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, libdevice.erf(a), mask=mask)

    @triton.jit
    def _ceil_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, libdevice.ceil(a), mask=mask)

    @triton.jit
    def _floor_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, libdevice.floor(a), mask=mask)

    @triton.jit
    def _trunc_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, libdevice.trunc(a), mask=mask)

    @triton.jit
    def _round_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, libdevice.rint(a), mask=mask)

    # ── activations with scalar params ────────────────────────────────────────

    @triton.jit
    def _leaky_relu_kernel(a_ptr, out_ptr, neg_slope, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, tl.where(a > 0.0, a, neg_slope * a).to(a.dtype), mask=mask)

    @triton.jit
    def _elu_kernel(a_ptr, out_ptr, alpha, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, tl.where(a > 0.0, a, alpha * (tl.exp(a) - 1.0)), mask=mask)

    # GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
    @triton.jit
    def _gelu_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(
            out_ptr + offs,
            0.5 * a * (1.0 + libdevice.erf(a * 0.7071067811865476)),
            mask=mask,
        )

    # SiLU / Swish: x * sigmoid(x)
    @triton.jit
    def _silu_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, a * tl.sigmoid(a), mask=mask)

    # softplus: log(1 + exp(x)), capped at x for x > threshold to avoid overflow
    @triton.jit
    def _softplus_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, tl.where(a > 20.0, a, tl.log(1.0 + tl.exp(a))), mask=mask)

    # hardsigmoid: clamp((x + 3) / 6, 0, 1)
    @triton.jit
    def _hardsigmoid_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        v = (a + 3.0) * 0.16666666666666666
        tl.store(out_ptr + offs, tl.where(v < 0.0, 0.0, tl.where(v > 1.0, 1.0, v)), mask=mask)

    # hardswish: x * hardsigmoid(x)
    @triton.jit
    def _hardswish_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        v = (a + 3.0) * 0.16666666666666666
        hs = tl.where(v < 0.0, 0.0, tl.where(v > 1.0, 1.0, v))
        tl.store(out_ptr + offs, a * hs, mask=mask)

    # erfc: 1 - erf(x)
    @triton.jit
    def _erfc_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, 1.0 - libdevice.erf(a), mask=mask)

    # exp2: 2^x = exp(x * ln2)
    @triton.jit
    def _exp2_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, tl.exp(a * 0.6931471805599453), mask=mask)

    # mish: x * tanh(softplus(x))
    @triton.jit
    def _mish_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sp = tl.where(a > 20.0, a, tl.log(1.0 + tl.exp(a)))
        tl.store(out_ptr + offs, a * (2.0 * tl.sigmoid(2.0 * sp) - 1.0), mask=mask)

    # lerp with scalar weight: a + weight * (b - a)
    @triton.jit
    def _lerp_scalar_kernel(a_ptr, b_ptr, out_ptr, weight, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, a + weight * (b - a), mask=mask)

    # lerp with tensor weight
    @triton.jit
    def _lerp_tt_kernel(a_ptr, b_ptr, w_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, a + w * (b - a), mask=mask)

    # copysign: |a| with sign of b
    @triton.jit
    def _copysign_tt_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sign_b = tl.where(b < 0.0, -1.0, 1.0)
        tl.store(out_ptr + offs, tl.abs(a) * sign_b, mask=mask)

    @triton.jit
    def _clamp_kernel(a_ptr, out_ptr, lo, hi, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        clamped = tl.where(a < lo, lo, tl.where(a > hi, hi, a))
        tl.store(out_ptr + offs, clamped.to(a.dtype), mask=mask)

    @triton.jit
    def _clamp_min_kernel(a_ptr, out_ptr, lo, N, BLOCK: tl.constexpr):
        """clamp with only min (no upper bound)."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, tl.where(a < lo, lo, a).to(a.dtype), mask=mask)

    @triton.jit
    def _clamp_max_kernel(a_ptr, out_ptr, hi, N, BLOCK: tl.constexpr):
        """clamp with only max (no lower bound)."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, tl.where(a > hi, hi, a).to(a.dtype), mask=mask)

    # ── new binary tt ─────────────────────────────────────────────────────────

    @triton.jit
    def _atan2_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        y = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(b_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        tl.store(out_ptr + offs, libdevice.atan2(y, x), mask=mask)

    @triton.jit
    def _fmod_tt_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
        """C-style fmod: result has the sign of the dividend."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        result = a - b * (a / b).to(tl.int32).to(tl.float32)
        tl.store(out_ptr + offs, result, mask=mask)

    @triton.jit
    def _fmod_scalar_kernel(a_ptr, out_ptr, scalar, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        result = a - scalar * (a / scalar).to(tl.int32).to(tl.float32)
        tl.store(out_ptr + offs, result, mask=mask)

    @triton.jit
    def _remainder_tt_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
        """Python-style remainder: result has the sign of the divisor."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        ratio = a / b
        trunc_r = ratio.to(tl.int32).to(tl.float32)
        floor_r = trunc_r - tl.where(ratio < trunc_r, 1.0, 0.0)
        tl.store(out_ptr + offs, a - b * floor_r, mask=mask)

    @triton.jit
    def _remainder_scalar_kernel(a_ptr, out_ptr, scalar, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        ratio = a / scalar
        trunc_r = ratio.to(tl.int32).to(tl.float32)
        floor_r = trunc_r - tl.where(ratio < trunc_r, 1.0, 0.0)
        tl.store(out_ptr + offs, a - scalar * floor_r, mask=mask)

    @triton.jit
    def _maximum_tt_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=float("-inf"))
        b = tl.load(b_ptr + offs, mask=mask, other=float("-inf"))
        tl.store(out_ptr + offs, tl.where(a >= b, a, b), mask=mask)

    @triton.jit
    def _minimum_tt_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=float("inf"))
        b = tl.load(b_ptr + offs, mask=mask, other=float("inf"))
        tl.store(out_ptr + offs, tl.where(a <= b, a, b), mask=mask)

    # ── comparison ops → bool output ──────────────────────────────────────────

    @triton.jit
    def _eq_tt_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, (a == b).to(tl.int8), mask=mask)

    @triton.jit
    def _ne_tt_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, (a != b).to(tl.int8), mask=mask)

    @triton.jit
    def _ge_tt_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, (a >= b).to(tl.int8), mask=mask)

    @triton.jit
    def _gt_tt_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, (a > b).to(tl.int8), mask=mask)

    @triton.jit
    def _le_tt_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, (a <= b).to(tl.int8), mask=mask)

    @triton.jit
    def _lt_tt_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, (a < b).to(tl.int8), mask=mask)

    @triton.jit
    def _eq_scalar_kernel(a_ptr, out_ptr, scalar, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, (a == scalar).to(tl.int8), mask=mask)

    @triton.jit
    def _ne_scalar_kernel(a_ptr, out_ptr, scalar, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, (a != scalar).to(tl.int8), mask=mask)

    @triton.jit
    def _ge_scalar_kernel(a_ptr, out_ptr, scalar, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, (a >= scalar).to(tl.int8), mask=mask)

    @triton.jit
    def _gt_scalar_kernel(a_ptr, out_ptr, scalar, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, (a > scalar).to(tl.int8), mask=mask)

    @triton.jit
    def _le_scalar_kernel(a_ptr, out_ptr, scalar, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, (a <= scalar).to(tl.int8), mask=mask)

    @triton.jit
    def _lt_scalar_kernel(a_ptr, out_ptr, scalar, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, (a < scalar).to(tl.int8), mask=mask)

    # ── logical ops ───────────────────────────────────────────────────────────

    @triton.jit
    def _logical_not_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0)
        tl.store(out_ptr + offs, (a == 0).to(tl.int8), mask=mask)

    @triton.jit
    def _logical_and_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0)
        b = tl.load(b_ptr + offs, mask=mask, other=0)
        tl.store(out_ptr + offs, ((a != 0) & (b != 0)).to(tl.int8), mask=mask)

    @triton.jit
    def _logical_or_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0)
        b = tl.load(b_ptr + offs, mask=mask, other=0)
        tl.store(out_ptr + offs, ((a != 0) | (b != 0)).to(tl.int8), mask=mask)

    @triton.jit
    def _logical_xor_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0)
        b = tl.load(b_ptr + offs, mask=mask, other=0)
        tl.store(out_ptr + offs, ((a != 0) ^ (b != 0)).to(tl.int8), mask=mask)

    # ── bitwise ops (integer tensors) ─────────────────────────────────────────

    @triton.jit
    def _bitwise_not_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0).to(tl.int64)
        tl.store(out_ptr + offs, ~a, mask=mask)

    @triton.jit
    def _bitwise_and_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0).to(tl.int64)
        b = tl.load(b_ptr + offs, mask=mask, other=0).to(tl.int64)
        tl.store(out_ptr + offs, a & b, mask=mask)

    @triton.jit
    def _bitwise_or_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0).to(tl.int64)
        b = tl.load(b_ptr + offs, mask=mask, other=0).to(tl.int64)
        tl.store(out_ptr + offs, a | b, mask=mask)

    @triton.jit
    def _bitwise_xor_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0).to(tl.int64)
        b = tl.load(b_ptr + offs, mask=mask, other=0).to(tl.int64)
        tl.store(out_ptr + offs, a ^ b, mask=mask)

    # ── special: where / fill / isinf / isnan ─────────────────────────────────

    @triton.jit
    def _where_kernel(cond_ptr, x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        cond = tl.load(cond_ptr + offs, mask=mask, other=0)
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        y = tl.load(y_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, tl.where(cond != 0, x, y), mask=mask)

    @triton.jit
    def _fill_scalar_kernel(out_ptr, scalar, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        tl.store(out_ptr + offs, scalar, mask=mask)

    @triton.jit
    def _isinf_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        # IEEE 754: isinf iff (bits & 0x7FFFFFFF) == 0x7F800000
        bits = a.to(tl.int32, bitcast=True)
        result = ((bits & 0x7FFFFFFF) == 0x7F800000).to(tl.int8)
        tl.store(out_ptr + offs, result, mask=mask)

    @triton.jit
    def _isnan_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        # IEEE 754: isnan iff (bits & 0x7FFFFFFF) > 0x7F800000
        bits = a.to(tl.int32, bitcast=True)
        result = ((bits & 0x7FFFFFFF) > 0x7F800000).to(tl.int8)
        tl.store(out_ptr + offs, result, mask=mask)


# ── helpers ───────────────────────────────────────────────────────────────────


def _grid(n: int):
    import math

    return (math.ceil(n / _BLOCK),)


def _bc(*tensors):
    """Broadcast all tensors to a common shape and make contiguous."""
    broadcast = torch.broadcast_tensors(*tensors)
    return tuple(t.contiguous() for t in broadcast)


def _f32_out(a: torch.Tensor) -> torch.Tensor:
    """Allocate a float32 output with the same shape/device as a."""
    return torch.empty_like(a, dtype=torch.float32)


def _bool_out(a: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(a, dtype=torch.bool)


# ── existing Python wrappers ──────────────────────────────────────────────────


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
    return _unary_f32(_exp_kernel, a)


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


# ── new unary wrappers ────────────────────────────────────────────────────────


def _unary_f32(kernel, a: torch.Tensor) -> torch.Tensor:
    """Run a unary kernel that computes in float32; cast result back to a.dtype."""
    orig = a.dtype
    a = a.contiguous()
    out = _f32_out(a)
    kernel[_grid(a.numel())](a, out, N=a.numel(), BLOCK=_BLOCK)
    return out.to(orig)


def abs_(a: torch.Tensor) -> torch.Tensor:
    a = a.contiguous()
    out = torch.empty_like(a)
    _abs_kernel[_grid(a.numel())](a, out, N=a.numel(), BLOCK=_BLOCK)
    return out


def sqrt_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_sqrt_kernel, a)


def reciprocal(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_reciprocal_kernel, a)


def sign(a: torch.Tensor) -> torch.Tensor:
    a = a.contiguous()
    out = torch.empty_like(a)
    _sign_kernel[_grid(a.numel())](a, out, N=a.numel(), BLOCK=_BLOCK)
    return out


def sin_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_sin_kernel, a)


def cos_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_cos_kernel, a)


def tan_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_tan_kernel, a)


def asin_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_asin_kernel, a)


def acos_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_acos_kernel, a)


def atan_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_atan_kernel, a)


def sinh_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_sinh_kernel, a)


def cosh_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_cosh_kernel, a)


def tanh_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_tanh_kernel, a)


def asinh_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_asinh_kernel, a)


def acosh_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_acosh_kernel, a)


def atanh_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_atanh_kernel, a)


def expm1_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_expm1_kernel, a)


def log_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_log_kernel, a)


def log1p_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_log1p_kernel, a)


def log2_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_log2_kernel, a)


def log10_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_log10_kernel, a)


def erf_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_erf_kernel, a)


def ceil_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_ceil_kernel, a)


def floor_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_floor_kernel, a)


def trunc_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_trunc_kernel, a)


def round_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_round_kernel, a)


def relu(a: torch.Tensor) -> torch.Tensor:
    a = a.contiguous()
    out = torch.empty_like(a)
    _relu_kernel[_grid(a.numel())](a, out, N=a.numel(), BLOCK=_BLOCK)
    return out


def sigmoid_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_sigmoid_kernel, a)


def tanh_activation(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_tanh_kernel, a)


def gelu(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_gelu_kernel, a)


def silu(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_silu_kernel, a)


def softplus(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_softplus_kernel, a)


def hardsigmoid(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_hardsigmoid_kernel, a)


def hardswish(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_hardswish_kernel, a)


def erfc_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_erfc_kernel, a)


def exp2_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_exp2_kernel, a)


def mish(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_mish_kernel, a)


def lerp_scalar(a: torch.Tensor, b: torch.Tensor, weight: float) -> torch.Tensor:
    a, b = _bc(a, b)
    out = torch.empty_like(a)
    _lerp_scalar_kernel[_grid(a.numel())](a, b, out, float(weight), N=a.numel(), BLOCK=_BLOCK)
    return out


def lerp_tt(a: torch.Tensor, b: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    a, b, weight = _bc(a, b, weight)
    out = torch.empty_like(a)
    _lerp_tt_kernel[_grid(a.numel())](a, b, weight, out, N=a.numel(), BLOCK=_BLOCK)
    return out


def copysign_tt(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    orig = a.dtype
    a, b = _bc(a, b)
    out = _f32_out(a)
    _copysign_tt_kernel[_grid(a.numel())](a, b, out, N=a.numel(), BLOCK=_BLOCK)
    return out.to(orig)


def leaky_relu(a: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    a = a.contiguous()
    out = torch.empty_like(a)
    _leaky_relu_kernel[_grid(a.numel())](a, out, float(negative_slope), N=a.numel(), BLOCK=_BLOCK)
    return out


def elu(a: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    orig = a.dtype
    a = a.contiguous()
    out = _f32_out(a)
    _elu_kernel[_grid(a.numel())](a, out, float(alpha), N=a.numel(), BLOCK=_BLOCK)
    return out.to(orig)


def clamp_(a: torch.Tensor, lo=None, hi=None) -> torch.Tensor:
    a = a.contiguous()
    out = torch.empty_like(a)
    N = a.numel()
    if lo is not None and hi is not None:
        _clamp_kernel[_grid(N)](a, out, float(lo), float(hi), N, BLOCK=_BLOCK)
    elif lo is not None:
        _clamp_min_kernel[_grid(N)](a, out, float(lo), N, BLOCK=_BLOCK)
    elif hi is not None:
        _clamp_max_kernel[_grid(N)](a, out, float(hi), N, BLOCK=_BLOCK)
    else:
        out = a.clone()
    return out


def isinf_(a: torch.Tensor) -> torch.Tensor:
    a = a.contiguous()
    out = _bool_out(a)
    _isinf_kernel[_grid(a.numel())](a, out, N=a.numel(), BLOCK=_BLOCK)
    return out


def isnan_(a: torch.Tensor) -> torch.Tensor:
    a = a.contiguous()
    out = _bool_out(a)
    _isnan_kernel[_grid(a.numel())](a, out, N=a.numel(), BLOCK=_BLOCK)
    return out


# ── new binary wrappers ───────────────────────────────────────────────────────


def atan2_(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    orig = a.dtype
    a, b = _bc(a, b)
    out = _f32_out(a)
    _atan2_kernel[_grid(a.numel())](a, b, out, N=a.numel(), BLOCK=_BLOCK)
    return out.to(orig)


def fmod_tt(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    orig = a.dtype
    a, b = _bc(a, b)
    out = _f32_out(a)
    _fmod_tt_kernel[_grid(a.numel())](a, b, out, N=a.numel(), BLOCK=_BLOCK)
    return out.to(orig)


def fmod_scalar(a: torch.Tensor, scalar) -> torch.Tensor:
    orig = a.dtype
    a = a.contiguous()
    out = _f32_out(a)
    _fmod_scalar_kernel[_grid(a.numel())](a, out, float(scalar), N=a.numel(), BLOCK=_BLOCK)
    return out.to(orig)


def remainder_tt(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    orig = a.dtype
    a, b = _bc(a, b)
    out = _f32_out(a)
    _remainder_tt_kernel[_grid(a.numel())](a, b, out, N=a.numel(), BLOCK=_BLOCK)
    return out.to(orig)


def remainder_scalar(a: torch.Tensor, scalar) -> torch.Tensor:
    orig = a.dtype
    a = a.contiguous()
    out = _f32_out(a)
    _remainder_scalar_kernel[_grid(a.numel())](a, out, float(scalar), N=a.numel(), BLOCK=_BLOCK)
    return out.to(orig)


def maximum_tt(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = _bc(a, b)
    out = torch.empty_like(a)
    _maximum_tt_kernel[_grid(a.numel())](a, b, out, N=a.numel(), BLOCK=_BLOCK)
    return out


def minimum_tt(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = _bc(a, b)
    out = torch.empty_like(a)
    _minimum_tt_kernel[_grid(a.numel())](a, b, out, N=a.numel(), BLOCK=_BLOCK)
    return out


# ── comparison wrappers ───────────────────────────────────────────────────────


def _cmp_tt(kernel, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = _bc(a, b)
    out = _bool_out(a)
    kernel[_grid(a.numel())](a, b, out, N=a.numel(), BLOCK=_BLOCK)
    return out


def _cmp_scalar(kernel, a: torch.Tensor, scalar) -> torch.Tensor:
    a = a.contiguous()
    out = _bool_out(a)
    kernel[_grid(a.numel())](a, out, float(scalar), N=a.numel(), BLOCK=_BLOCK)
    return out


def eq_tt(a, b):
    return _cmp_tt(_eq_tt_kernel, a, b)


def ne_tt(a, b):
    return _cmp_tt(_ne_tt_kernel, a, b)


def ge_tt(a, b):
    return _cmp_tt(_ge_tt_kernel, a, b)


def gt_tt(a, b):
    return _cmp_tt(_gt_tt_kernel, a, b)


def le_tt(a, b):
    return _cmp_tt(_le_tt_kernel, a, b)


def lt_tt(a, b):
    return _cmp_tt(_lt_tt_kernel, a, b)


def eq_scalar(a, s):
    return _cmp_scalar(_eq_scalar_kernel, a, s)


def ne_scalar(a, s):
    return _cmp_scalar(_ne_scalar_kernel, a, s)


def ge_scalar(a, s):
    return _cmp_scalar(_ge_scalar_kernel, a, s)


def gt_scalar(a, s):
    return _cmp_scalar(_gt_scalar_kernel, a, s)


def le_scalar(a, s):
    return _cmp_scalar(_le_scalar_kernel, a, s)


def lt_scalar(a, s):
    return _cmp_scalar(_lt_scalar_kernel, a, s)


# ── logical wrappers ──────────────────────────────────────────────────────────


def logical_not(a: torch.Tensor) -> torch.Tensor:
    a = a.contiguous()
    out = _bool_out(a)
    _logical_not_kernel[_grid(a.numel())](a, out, N=a.numel(), BLOCK=_BLOCK)
    return out


def logical_and(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = _bc(a, b)
    out = _bool_out(a)
    _logical_and_kernel[_grid(a.numel())](a, b, out, N=a.numel(), BLOCK=_BLOCK)
    return out


def logical_or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = _bc(a, b)
    out = _bool_out(a)
    _logical_or_kernel[_grid(a.numel())](a, b, out, N=a.numel(), BLOCK=_BLOCK)
    return out


def logical_xor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = _bc(a, b)
    out = _bool_out(a)
    _logical_xor_kernel[_grid(a.numel())](a, b, out, N=a.numel(), BLOCK=_BLOCK)
    return out


# ── bitwise wrappers ──────────────────────────────────────────────────────────


def bitwise_not(a: torch.Tensor) -> torch.Tensor:
    a = a.contiguous()
    out = torch.empty_like(a)
    _bitwise_not_kernel[_grid(a.numel())](a, out, N=a.numel(), BLOCK=_BLOCK)
    return out


def bitwise_and(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = _bc(a, b)
    out = torch.empty_like(a)
    _bitwise_and_kernel[_grid(a.numel())](a, b, out, N=a.numel(), BLOCK=_BLOCK)
    return out


def bitwise_or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = _bc(a, b)
    out = torch.empty_like(a)
    _bitwise_or_kernel[_grid(a.numel())](a, b, out, N=a.numel(), BLOCK=_BLOCK)
    return out


def bitwise_xor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = _bc(a, b)
    out = torch.empty_like(a)
    _bitwise_xor_kernel[_grid(a.numel())](a, b, out, N=a.numel(), BLOCK=_BLOCK)
    return out


# ── special wrappers ──────────────────────────────────────────────────────────


def where(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    condition, x, y = torch.broadcast_tensors(condition, x, y)
    condition = condition.contiguous()
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    N = x.numel()
    _where_kernel[_grid(N)](condition, x, y, out, N, BLOCK=_BLOCK)
    return out


def fill_scalar(a: torch.Tensor, scalar) -> torch.Tensor:
    """Return a new tensor with the same shape/dtype as a, filled with scalar."""
    out = torch.empty_like(a)
    _fill_scalar_kernel[_grid(a.numel())](out, float(scalar), N=a.numel(), BLOCK=_BLOCK)
    return out


def to_copy(a: torch.Tensor, dtype=None, *args) -> torch.Tensor:
    """Cast tensor to dtype (Python-level; no Triton kernel needed)."""
    if dtype is not None and isinstance(dtype, torch.dtype):
        return a.to(dtype=dtype).contiguous()
    return a.contiguous().clone()
