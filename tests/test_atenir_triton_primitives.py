import importlib
import json
import os
import sys
import tempfile
import unittest

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _skip_reason():
    try:
        import triton  # noqa: F401
    except ImportError:
        return "Triton not installed"
    if not torch.cuda.is_available():
        return "CUDA not available"
    return None


_SKIP = _skip_reason()


@unittest.skipIf(_SKIP, _SKIP or "")
class TestAtenIRPrimitivesAndExamples(unittest.TestCase):
    def test_elementwise_primitives(self):
        from atenir.primitive_triton import elementwise

        a = torch.randn(32, 64, device="cuda")
        b = torch.randn(32, 64, device="cuda")
        c = torch.randn(64, device="cuda")
        pos = torch.rand(32, 64, device="cuda") + 0.1
        exp_tensor = torch.rand(32, 64, device="cuda") * 2.0
        exp_broadcast = torch.rand(64, device="cuda") * 2.0

        torch.testing.assert_close(elementwise.mul_tt(a, b), a * b)
        torch.testing.assert_close(elementwise.mul_tt(a, c), a * c)
        torch.testing.assert_close(elementwise.add_tt(a, b), a + b)
        torch.testing.assert_close(elementwise.add_tt(a, b, alpha=0.5), a + 0.5 * b)
        torch.testing.assert_close(elementwise.sub_tt(a, b), a - b)
        torch.testing.assert_close(elementwise.div_tt(a, pos), a / pos)
        torch.testing.assert_close(elementwise.rsqrt(pos), torch.rsqrt(pos), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(elementwise.neg(a), -a)
        torch.testing.assert_close(elementwise.exp(a), torch.exp(a), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(elementwise.mul_scalar(a, 3.0), a * 3.0)
        torch.testing.assert_close(elementwise.add_scalar(a, 2.0), a + 2.0)
        torch.testing.assert_close(elementwise.div_scalar(a, 2.0), a / 2.0)

        torch.testing.assert_close(
            elementwise.pow_scalar(pos, 3.0),
            torch.pow(pos, 3.0),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            elementwise.pow_tt(pos, exp_tensor),
            torch.pow(pos, exp_tensor),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            elementwise.pow_tt(pos, exp_broadcast),
            torch.pow(pos, exp_broadcast),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_reduction_primitives(self):
        from atenir.primitive_triton.reduction import (
            make_mean_kernel,
            make_softmax_kernel,
            make_sum_kernel,
        )

        x = torch.randn(32, 64, device="cuda")

        cases = [
            ([1], False),
            ([1], True),
            ([-1], False),
            ([-1], True),
            ([0], False),
            ([0], True),
        ]

        for dims, keepdim in cases:
            with self.subTest(op="sum", dims=dims, keepdim=keepdim):
                got = make_sum_kernel(dims, keepdim)(x)
                ref = x.sum(dim=dims, keepdim=keepdim)
                torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)

            with self.subTest(op="mean", dims=dims, keepdim=keepdim):
                got = make_mean_kernel(dims, keepdim)(x)
                ref = x.mean(dim=dims, keepdim=keepdim)
                torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)

        y = torch.randn(4, 8, 16, device="cuda")

        got = make_sum_kernel([2], False)(y)
        ref = y.sum(dim=[2], keepdim=False)
        torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)

        got = make_softmax_kernel(-1)(x, -1)
        ref = torch.softmax(x, dim=-1)
        torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)

        got = make_softmax_kernel(-1)(y, -1)
        ref = torch.softmax(y, dim=-1)
        torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)

    def test_scatter_gather_primitives(self):
        from atenir.primitive_triton import scatter_gather

        x1 = torch.randn(10, device="cuda")
        idx1 = torch.tensor([3, 0, 7, 7, 2, 9], device="cuda", dtype=torch.long)

        torch.testing.assert_close(
            scatter_gather.gather(x1, 0, idx1),
            torch.gather(x1, 0, idx1),
        )

        base1 = torch.randn(6, device="cuda")
        src1 = torch.randn(8, device="cuda")
        idx_sc1 = torch.tensor(
            [0, 1, 1, 3, 5, 5, 5, 2],
            device="cuda",
            dtype=torch.long,
        )

        ref = base1.clone()
        ref.scatter_add_(0, idx_sc1, src1)
        torch.testing.assert_close(
            scatter_gather.scatter_add(base1, 0, idx_sc1, src1),
            ref,
        )

        x2 = torch.randn(4, 6, device="cuda")

        idx_dim1 = torch.tensor(
            [
                [0, 2, 5],
                [1, 1, 3],
                [4, 0, 2],
                [5, 5, 1],
            ],
            device="cuda",
            dtype=torch.long,
        )

        torch.testing.assert_close(
            scatter_gather.gather(x2, 1, idx_dim1),
            torch.gather(x2, 1, idx_dim1),
        )

        idx_dim0 = torch.tensor(
            [
                [0, 1, 2, 3, 0, 1],
                [3, 2, 1, 0, 3, 2],
            ],
            device="cuda",
            dtype=torch.long,
        )

        torch.testing.assert_close(
            scatter_gather.gather(x2, 0, idx_dim0),
            torch.gather(x2, 0, idx_dim0),
        )

        base2 = torch.randn(4, 6, device="cuda")
        src_dim1 = torch.randn(4, 4, device="cuda")
        idx_sc_dim1 = torch.tensor(
            [
                [0, 1, 1, 5],
                [2, 2, 3, 3],
                [4, 0, 4, 1],
                [5, 5, 0, 2],
            ],
            device="cuda",
            dtype=torch.long,
        )

        ref = base2.clone()
        ref.scatter_add_(1, idx_sc_dim1, src_dim1)
        torch.testing.assert_close(
            scatter_gather.scatter_add(base2, 1, idx_sc_dim1, src_dim1),
            ref,
        )

        base3 = torch.randn(4, 6, device="cuda")
        src_dim0 = torch.randn(3, 6, device="cuda")
        idx_sc_dim0 = torch.tensor(
            [
                [0, 1, 2, 3, 0, 1],
                [1, 1, 2, 2, 3, 3],
                [3, 0, 0, 1, 2, 2],
            ],
            device="cuda",
            dtype=torch.long,
        )

        ref = base3.clone()
        ref.scatter_add_(0, idx_sc_dim0, src_dim0)
        torch.testing.assert_close(
            scatter_gather.scatter_add(base3, 0, idx_sc_dim0, src_dim0),
            ref,
        )

    # ── new elementwise tests ─────────────────────────────────────────────────

    def test_new_unary_elementwise(self):
        from atenir.primitive_triton import elementwise

        a = torch.randn(32, 64, device="cuda")
        pos = torch.rand(32, 64, device="cuda") + 0.1
        # domain [-1, 1] for asin/acos/atanh
        d11 = torch.rand(32, 64, device="cuda") * 1.8 - 0.9
        # domain >= 1 for acosh
        d1inf = torch.rand(32, 64, device="cuda") + 1.01

        T = 1e-5

        # basic math
        torch.testing.assert_close(elementwise.abs_(a), torch.abs(a))
        torch.testing.assert_close(elementwise.sqrt_(pos), torch.sqrt(pos), atol=T, rtol=T)
        torch.testing.assert_close(
            elementwise.reciprocal(pos), torch.reciprocal(pos), atol=T, rtol=T
        )
        torch.testing.assert_close(elementwise.sign(a), torch.sign(a))
        torch.testing.assert_close(elementwise.neg(a), -a)

        # rounding — exact (nearest-integer results in fp32)
        torch.testing.assert_close(elementwise.ceil_(a), torch.ceil(a), atol=0, rtol=0)
        torch.testing.assert_close(elementwise.floor_(a), torch.floor(a), atol=0, rtol=0)
        torch.testing.assert_close(elementwise.trunc_(a), torch.trunc(a), atol=0, rtol=0)
        torch.testing.assert_close(elementwise.round_(a), torch.round(a), atol=0, rtol=0)

        # trig (tl native)
        torch.testing.assert_close(elementwise.sin_(a), torch.sin(a), atol=T, rtol=T)
        torch.testing.assert_close(elementwise.cos_(a), torch.cos(a), atol=T, rtol=T)

        # trig (libdevice)
        torch.testing.assert_close(elementwise.tan_(a), torch.tan(a), atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(elementwise.asin_(d11), torch.asin(d11), atol=T, rtol=T)
        torch.testing.assert_close(elementwise.acos_(d11), torch.acos(d11), atol=T, rtol=T)
        torch.testing.assert_close(elementwise.atan_(a), torch.atan(a), atol=T, rtol=T)

        # hyperbolic
        torch.testing.assert_close(elementwise.sinh_(a), torch.sinh(a), atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(elementwise.cosh_(a), torch.cosh(a), atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(elementwise.tanh_(a), torch.tanh(a), atol=T, rtol=T)
        torch.testing.assert_close(elementwise.asinh_(a), torch.asinh(a), atol=T, rtol=T)
        torch.testing.assert_close(elementwise.acosh_(d1inf), torch.acosh(d1inf), atol=T, rtol=T)
        torch.testing.assert_close(elementwise.atanh_(d11), torch.atanh(d11), atol=T, rtol=T)

        # exp / log family
        torch.testing.assert_close(elementwise.exp(a), torch.exp(a), atol=T, rtol=T)
        torch.testing.assert_close(elementwise.expm1_(a), torch.expm1(a), atol=T, rtol=T)
        torch.testing.assert_close(elementwise.log_(pos), torch.log(pos), atol=T, rtol=T)
        torch.testing.assert_close(elementwise.log1p_(pos), torch.log1p(pos), atol=T, rtol=T)
        torch.testing.assert_close(elementwise.log2_(pos), torch.log2(pos), atol=T, rtol=T)
        torch.testing.assert_close(elementwise.log10_(pos), torch.log10(pos), atol=T, rtol=T)
        torch.testing.assert_close(elementwise.erf_(a), torch.erf(a), atol=T, rtol=T)

        # activations
        torch.testing.assert_close(elementwise.relu(a), torch.relu(a))
        torch.testing.assert_close(elementwise.sigmoid_(a), torch.sigmoid(a), atol=T, rtol=T)
        torch.testing.assert_close(elementwise.tanh_activation(a), torch.tanh(a), atol=T, rtol=T)
        torch.testing.assert_close(
            elementwise.gelu(a), torch.nn.functional.gelu(a), atol=2e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            elementwise.leaky_relu(a, 0.1),
            torch.nn.functional.leaky_relu(a, 0.1),
            atol=T,
            rtol=T,
        )
        torch.testing.assert_close(
            elementwise.elu(a, 1.0), torch.nn.functional.elu(a, 1.0), atol=T, rtol=T
        )

        # clamp
        torch.testing.assert_close(elementwise.clamp_(a, -0.5, 0.5), torch.clamp(a, -0.5, 0.5))
        torch.testing.assert_close(elementwise.clamp_(a, lo=-0.5), torch.clamp(a, min=-0.5))
        torch.testing.assert_close(elementwise.clamp_(a, hi=0.5), torch.clamp(a, max=0.5))

        # isinf / isnan
        special = torch.tensor(
            [float("inf"), float("-inf"), float("nan"), 0.0, 1.0],
            device="cuda",
        )
        torch.testing.assert_close(elementwise.isinf_(special), torch.isinf(special))
        torch.testing.assert_close(elementwise.isnan_(special), torch.isnan(special))

    def test_new_binary_elementwise(self):
        from atenir.primitive_triton import elementwise

        a = torch.randn(32, 64, device="cuda")
        b = torch.randn(32, 64, device="cuda")
        pos_a = torch.rand(32, 64, device="cuda") + 0.1
        pos_b = torch.rand(32, 64, device="cuda") + 0.1

        T = 1e-5

        torch.testing.assert_close(elementwise.atan2_(a, b), torch.atan2(a, b), atol=T, rtol=T)
        torch.testing.assert_close(
            elementwise.fmod_tt(a, pos_b), torch.fmod(a, pos_b), atol=T, rtol=T
        )
        torch.testing.assert_close(
            elementwise.fmod_scalar(a, 0.3), torch.fmod(a, 0.3), atol=T, rtol=T
        )
        torch.testing.assert_close(
            elementwise.remainder_tt(a, pos_b),
            torch.remainder(a, pos_b),
            atol=T,
            rtol=T,
        )
        torch.testing.assert_close(
            elementwise.remainder_scalar(a, 0.3),
            torch.remainder(a, 0.3),
            atol=T,
            rtol=T,
        )
        torch.testing.assert_close(elementwise.maximum_tt(a, b), torch.maximum(a, b))
        torch.testing.assert_close(elementwise.minimum_tt(a, b), torch.minimum(a, b))

    def test_comparison_elementwise(self):
        from atenir.primitive_triton import elementwise

        a = torch.randn(32, 64, device="cuda")
        b = torch.randn(32, 64, device="cuda")
        s = 0.5

        pairs = [
            ("eq_tt", lambda x, y: x == y),
            ("ne_tt", lambda x, y: x != y),
            ("ge_tt", lambda x, y: x >= y),
            ("gt_tt", lambda x, y: x > y),
            ("le_tt", lambda x, y: x <= y),
            ("lt_tt", lambda x, y: x < y),
        ]
        for fn_name, ref_fn in pairs:
            with self.subTest(fn=fn_name):
                got = getattr(elementwise, fn_name)(a, b)
                ref = ref_fn(a, b)
                self.assertEqual(got.dtype, torch.bool)
                self.assertTrue(torch.all(got == ref), f"{fn_name} tt mismatch")

        scalar_pairs = [
            ("eq_scalar", lambda x: x == s),
            ("ne_scalar", lambda x: x != s),
            ("ge_scalar", lambda x: x >= s),
            ("gt_scalar", lambda x: x > s),
            ("le_scalar", lambda x: x <= s),
            ("lt_scalar", lambda x: x < s),
        ]
        for fn_name, ref_fn in scalar_pairs:
            with self.subTest(fn=fn_name):
                got = getattr(elementwise, fn_name)(a, s)
                ref = ref_fn(a)
                self.assertEqual(got.dtype, torch.bool)
                self.assertTrue(torch.all(got == ref), f"{fn_name} scalar mismatch")

    def test_logical_bitwise_elementwise(self):
        from atenir.primitive_triton import elementwise

        a = torch.randn(32, 64, device="cuda")
        b = torch.randn(32, 64, device="cuda")
        # logical ops work on any dtype; result is bool
        bool_a = a > 0
        bool_b = b > 0

        self.assertTrue(torch.all(elementwise.logical_not(bool_a) == ~bool_a))
        self.assertTrue(torch.all(elementwise.logical_and(bool_a, bool_b) == (bool_a & bool_b)))
        self.assertTrue(torch.all(elementwise.logical_or(bool_a, bool_b) == (bool_a | bool_b)))
        self.assertTrue(torch.all(elementwise.logical_xor(bool_a, bool_b) == (bool_a ^ bool_b)))

        # bitwise ops on integers
        int_a = torch.randint(-100, 100, (32, 64), device="cuda", dtype=torch.int32)
        int_b = torch.randint(-100, 100, (32, 64), device="cuda", dtype=torch.int32)

        torch.testing.assert_close(elementwise.bitwise_not(int_a), ~int_a)
        torch.testing.assert_close(elementwise.bitwise_and(int_a, int_b), int_a & int_b)
        torch.testing.assert_close(elementwise.bitwise_or(int_a, int_b), int_a | int_b)
        torch.testing.assert_close(elementwise.bitwise_xor(int_a, int_b), int_a ^ int_b)

    def test_special_elementwise(self):
        from atenir.primitive_triton import elementwise

        a = torch.randn(32, 64, device="cuda")
        x = torch.randn(32, 64, device="cuda")
        y = torch.randn(32, 64, device="cuda")
        cond = a > 0

        # where
        torch.testing.assert_close(elementwise.where(cond, x, y), torch.where(cond, x, y))

        # fill_scalar
        torch.testing.assert_close(elementwise.fill_scalar(a, 3.14), torch.full_like(a, 3.14))

        # to_copy (dtype cast)
        torch.testing.assert_close(elementwise.to_copy(a, torch.float16), a.to(torch.float16))
        torch.testing.assert_close(elementwise.to_copy(a, None), a.contiguous().clone())

    # ── new reduction tests ───────────────────────────────────────────────────

    def test_amax_amin_kernels(self):
        from atenir.primitive_triton.reduction import make_amax_kernel, make_amin_kernel

        x = torch.randn(32, 64, device="cuda")

        for dims, keepdim in [([1], False), ([1], True), ([0], False), ([0], True)]:
            with self.subTest(op="amax", dims=dims, keepdim=keepdim):
                got = make_amax_kernel(dims, keepdim)(x)
                ref = x.amax(dim=dims, keepdim=keepdim)
                torch.testing.assert_close(got, ref)

            with self.subTest(op="amin", dims=dims, keepdim=keepdim):
                got = make_amin_kernel(dims, keepdim)(x)
                ref = x.amin(dim=dims, keepdim=keepdim)
                torch.testing.assert_close(got, ref)

        # N-D fallback path
        y = torch.randn(4, 8, 16, device="cuda")
        torch.testing.assert_close(make_amax_kernel([2], False)(y), y.amax(dim=[2]))

    def test_max_min_dim_kernels(self):
        from atenir.primitive_triton.reduction import make_max_dim_kernel, make_min_dim_kernel

        x = torch.randn(16, 32, device="cuda")

        for dim, keepdim in [(0, False), (0, True), (1, False), (1, True)]:
            with self.subTest(op="max.dim", dim=dim, keepdim=keepdim):
                got = make_max_dim_kernel(dim, keepdim)(x)
                ref = torch.max(x, dim=dim, keepdim=keepdim)
                torch.testing.assert_close(got.values, ref.values)
                torch.testing.assert_close(got.indices, ref.indices)

            with self.subTest(op="min.dim", dim=dim, keepdim=keepdim):
                got = make_min_dim_kernel(dim, keepdim)(x)
                ref = torch.min(x, dim=dim, keepdim=keepdim)
                torch.testing.assert_close(got.values, ref.values)
                torch.testing.assert_close(got.indices, ref.indices)

    def test_argmax_argmin_kernels(self):
        from atenir.primitive_triton.reduction import make_argmax_kernel, make_argmin_kernel

        x = torch.randn(16, 32, device="cuda")

        for dim in [None, 0, 1]:
            with self.subTest(op="argmax", dim=dim):
                got = make_argmax_kernel(dim, False)(x)
                ref = torch.argmax(x, dim=dim)
                torch.testing.assert_close(got, ref)

            with self.subTest(op="argmin", dim=dim):
                got = make_argmin_kernel(dim, False)(x)
                ref = torch.argmin(x, dim=dim)
                torch.testing.assert_close(got, ref)

    def test_prod_kernel(self):
        from atenir.primitive_triton.reduction import make_prod_kernel

        x = torch.rand(16, 32, device="cuda") + 0.1

        for dims, keepdim in [([1], False), ([1], True), ([0], False), ([0], True)]:
            with self.subTest(dims=dims, keepdim=keepdim):
                got = make_prod_kernel(dims, keepdim)(x)
                ref = x.prod(dim=dims[0], keepdim=keepdim)
                torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)

    def test_any_kernel(self):
        from atenir.primitive_triton.reduction import make_any_kernel

        x = torch.randint(0, 3, (16, 32), device="cuda")

        for dims, keepdim in [([1], False), ([1], True), ([0], False), ([0], True)]:
            with self.subTest(dims=dims, keepdim=keepdim):
                got = make_any_kernel(dims, keepdim)(x)
                ref = x.any(dim=dims[0], keepdim=keepdim)
                self.assertTrue(torch.all(got == ref), f"any mismatch dims={dims}")

        # scalar any (no dim)
        got_scalar = make_any_kernel([], False)(x)
        self.assertEqual(got_scalar.item(), x.any().item())

    def test_var_kernel(self):
        from atenir.primitive_triton.reduction import make_var_kernel

        x = torch.randn(16, 32, device="cuda")

        for dims, keepdim, correction in [
            ([1], False, 1),
            ([0], True, 0),
            ([1], True, 1),
        ]:
            with self.subTest(dims=dims, keepdim=keepdim, correction=correction):
                got = make_var_kernel(dims, keepdim, correction)(x)
                ref = torch.var(x, dim=dims[0], correction=correction, keepdim=keepdim)
                torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)

    def test_cumsum_kernel(self):
        from atenir.primitive_triton.reduction import make_cumsum_kernel

        x = torch.randn(16, 32, device="cuda")

        for dim in [0, 1]:
            with self.subTest(dim=dim):
                got = make_cumsum_kernel(dim)(x)
                ref = torch.cumsum(x, dim=dim)
                torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)

        # N-D fallback (dim=2 on 3D tensor)
        y = torch.randn(4, 8, 16, device="cuda")
        got = make_cumsum_kernel(2)(y)
        ref = torch.cumsum(y, dim=2)
        torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)

    # ── GEMM tests ────────────────────────────────────────────────────────────

    def test_mm(self):
        from atenir.primitive_triton.gemm import mm

        cases = [
            (64, 64, 64),  # square, fits exactly in one tile
            (128, 256, 64),  # rectangular
            (100, 200, 150),  # non-power-of-2, exercises K-tail masking
            (1, 64, 64),  # M=1 edge case
            (64, 1, 64),  # N=1 edge case
        ]
        for M, N, K in cases:
            with self.subTest(M=M, N=N, K=K):
                a = torch.randn(M, K, device="cuda")
                b = torch.randn(K, N, device="cuda")
                # Use float64 reference to avoid TF32 vs FP32 disagreement
                ref = torch.mm(a.double(), b.double()).float()
                got = mm(a, b)
                torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)

        # float16
        a16 = torch.randn(64, 64, device="cuda", dtype=torch.float16)
        b16 = torch.randn(64, 64, device="cuda", dtype=torch.float16)
        torch.testing.assert_close(mm(a16, b16), torch.mm(a16, b16), atol=1e-2, rtol=1e-2)

    def test_bmm(self):
        from atenir.primitive_triton.gemm import bmm

        cases = [
            (4, 64, 64, 64),  # batch=4, square
            (8, 32, 48, 16),  # batch=8, rectangular
            (3, 100, 50, 70),  # non-power-of-2 K (odd K, tests EVEN_K=False)
            (1, 64, 64, 64),  # batch=1
        ]
        for B, M, N, K in cases:
            with self.subTest(B=B, M=M, N=N, K=K):
                a = torch.randn(B, M, K, device="cuda")
                b = torch.randn(B, K, N, device="cuda")
                ref = torch.bmm(a.double(), b.double()).float()
                got = bmm(a, b)
                torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)

    def test_addmm(self):
        from atenir.primitive_triton.gemm import addmm

        M, N, K = 64, 128, 96

        a = torch.randn(M, K, device="cuda")
        b = torch.randn(K, N, device="cuda")

        # 2-D bias [M, N]
        bias_2d = torch.randn(M, N, device="cuda")
        ref = torch.addmm(bias_2d.double(), a.double(), b.double()).float()
        got = addmm(bias_2d, a, b)
        torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)

        # 1-D bias [N] — broadcast over rows (typical linear layer use)
        bias_1d = torch.randn(N, device="cuda")
        ref = torch.addmm(bias_1d.double(), a.double(), b.double()).float()
        got = addmm(bias_1d, a, b)
        torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)

        # Non-unit alpha/beta
        ref = torch.addmm(bias_2d.double(), a.double(), b.double(), beta=0.5, alpha=2.0).float()
        got = addmm(bias_2d, a, b, beta=0.5, alpha=2.0)
        torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)

        # Non-power-of-2 sizes (K-tail masking)
        a2 = torch.randn(70, 50, device="cuda")
        b2 = torch.randn(50, 90, device="cuda")
        bias2 = torch.randn(90, device="cuda")
        ref = torch.addmm(bias2.double(), a2.double(), b2.double()).float()
        got = addmm(bias2, a2, b2)
        torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)

    def _roundtrip(self, fn_spec, input_shapes, atol=1e-4, rtol=1e-4):
        from atenir.extract import extract_autograd, _parse_spec, _serialise
        from atenir.compose import run_graph
        from atenir.primitive_triton.dispatch import make_registry

        torch.manual_seed(0)
        fwd_inputs = [
            torch.randn(shape, device="cuda", requires_grad=True) for shape in input_shapes
        ]

        mod_name, fn_name = fn_spec.split(":")
        fn = getattr(importlib.import_module(mod_name), fn_name)

        ref_inputs = [t.detach().clone().requires_grad_(True) for t in fwd_inputs]
        out = fn(*ref_inputs)
        if isinstance(out, (tuple, list)):
            out = out[0]

        grad_out = torch.randn_like(out)
        ref_grads = torch.autograd.grad(out, ref_inputs, grad_outputs=grad_out)

        spec = (
            "["
            + ", ".join("(" + ",".join(str(d) for d in shape) + ") f32" for shape in input_shapes)
            + "]"
        )

        graph = _serialise(extract_autograd(fn_spec, _parse_spec(spec), device="cpu"))

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(graph, tmp)
        tmp.close()

        try:
            placeholders = [n for n in graph["nodes"] if n["op"] == "placeholder"]
            env_values = [grad_out.contiguous()] + [t.detach().contiguous() for t in fwd_inputs]
            self.assertEqual(len(placeholders), len(env_values))

            env = {p["name"]: v for p, v in zip(placeholders, env_values)}
            registry = make_registry(graph)
            got_grads = run_graph(tmp.name, env, registry)

            self.assertEqual(len(got_grads), len(ref_grads))

            for i, (got, ref) in enumerate(zip(got_grads, ref_grads)):
                torch.testing.assert_close(
                    got.float(),
                    ref.float(),
                    atol=atol,
                    rtol=rtol,
                    msg=f"{fn_name} grad[{i}] mismatch",
                )
        finally:
            os.unlink(tmp.name)


if __name__ == "__main__":
    unittest.main()
