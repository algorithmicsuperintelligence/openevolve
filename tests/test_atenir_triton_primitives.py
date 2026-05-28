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
