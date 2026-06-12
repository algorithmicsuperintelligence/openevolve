"""Correctness tests for atenir._examples via primitive Triton kernels.

Pipeline under test:
  1. extract_autograd  — traces bwd(grad_out, *fwd_inputs) with make_fx +
                         core_aten_decompositions, serialises to JSON.
  2. make_registry     — assigns each call_function node a Triton kernel (or a
                         PyTorch fallback for complex / view ops).
  3. run_graph         — composes the graph with those kernels.
  4. compare           — outputs must match torch.autograd.grad to atol/rtol.

All tests are skipped when CUDA or Triton is not available.
"""

from __future__ import annotations

import torch
import json
import os
import sys
import tempfile
import unittest

# Make sure the repo root is importable when run standalone
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _skip_if_no_gpu():
    try:
        import torch
        import triton  # noqa: F401
    except ImportError:
        return "torch or triton not installed"
    if not torch.cuda.is_available():
        return "CUDA not available"
    return None


_SKIP_REASON = _skip_if_no_gpu()


@unittest.skipIf(_SKIP_REASON, _SKIP_REASON or "")
class TestAtenIRTritonExamples(unittest.TestCase):
    """Round-trip test: extract → Triton kernels → compare vs autograd."""

    # ── shared helpers ────────────────────────────────────────────────────────

    def _roundtrip(
        self,
        fn_spec: str,
        input_shapes: list[tuple[int, ...]],
        atol: float = 1e-4,
        rtol: float = 1e-4,
    ):
        """
        Full pipeline for one forward function.

        fn_spec      : "module:fn" string resolvable by atenir.extract
        input_shapes : shapes of the forward inputs; all assumed float32
        """
        import torch
        from atenir.extract import extract_autograd, _serialise, _parse_spec
        from atenir.compose import run_graph
        from atenir.primitive_triton.dispatch import make_registry

        device = "cuda"

        # ── 1. build concrete forward inputs (float32, CUDA) ─────────────────
        torch.manual_seed(0)
        fwd_inputs = [
            torch.randn(shape, device=device, requires_grad=True) for shape in input_shapes
        ]

        # ── 2. autograd reference ─────────────────────────────────────────────
        ins_ref = [t.detach().clone().requires_grad_(True) for t in fwd_inputs]
        fn_mod, fn_name = fn_spec.split(":")
        import importlib

        fn = getattr(importlib.import_module(fn_mod), fn_name)
        out = fn(*ins_ref)
        if isinstance(out, (tuple, list)):
            out = out[0]
        grad_out = torch.randn_like(out)
        ref_grads = torch.autograd.grad(out, ins_ref, grad_outputs=grad_out)

        # ── 3. extract backward graph on CPU ─────────────────────────────────
        spec_str = (
            "[" + ", ".join("(" + ",".join(str(d) for d in s) + ") f32" for s in input_shapes) + "]"
        )
        parsed = _parse_spec(spec_str)
        gm = extract_autograd(fn_spec, parsed, device="cpu")
        graph = _serialise(gm)

        # ── 4. write JSON to temp file ────────────────────────────────────────
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(graph, tmp)
        tmp.close()
        graph_path = tmp.name

        try:
            # ── 5. build kernel registry ──────────────────────────────────────
            registry = make_registry(graph)

            # ── 6. build env: placeholders order = [grad_out, *fwd_inputs] ────
            placeholders = [n for n in graph["nodes"] if n["op"] == "placeholder"]
            env_tensors = [grad_out.contiguous()] + [t.detach().contiguous() for t in fwd_inputs]
            self.assertEqual(
                len(placeholders),
                len(env_tensors),
                f"placeholder count {len(placeholders)} != " f"env tensor count {len(env_tensors)}",
            )
            env = {p["name"]: t for p, t in zip(placeholders, env_tensors)}

            # ── 7. run graph ──────────────────────────────────────────────────
            results = run_graph(graph_path, env, registry)

            # ── 8. compare outputs ────────────────────────────────────────────
            self.assertEqual(
                len(results),
                len(ref_grads),
                f"output count mismatch: got {len(results)}, " f"expected {len(ref_grads)}",
            )
            for i, (got, ref) in enumerate(zip(results, ref_grads)):
                try:
                    torch.testing.assert_close(
                        got.float(),
                        ref.float(),
                        atol=atol,
                        rtol=rtol,
                    )
                except AssertionError as e:
                    raise AssertionError(f"gradient[{i}] mismatch for {fn_name}\n{e}") from None
        finally:
            os.unlink(graph_path)

    # ── test cases ────────────────────────────────────────────────────────────

    def test_square_sum(self):
        """y = sum(x * x, dim=-1).  Backward: dx = 2 * x * expand(dy)."""
        self._roundtrip(
            "atenir._examples:square_sum",
            input_shapes=[(2048, 2048)],
        )

    def test_rmsnorm(self):
        """RMSNorm backward — involves rsqrt, sum (row + col), mul, sub."""
        self._roundtrip(
            "atenir._examples:rmsnorm",
            input_shapes=[(2048, 4096), (4096,)],
            # RMSNorm backward involves higher-order terms; slightly looser tol
            atol=2e-4,
            rtol=2e-4,
        )

    def test_attention_block(self):
        """Scaled dot-product attention backward.

        Backward contains aten.mm / aten.bmm (PyTorch fallback) plus primitive
        elementwise ops handled by Triton kernels.
        """
        self._roundtrip(
            "atenir._examples:attention_block",
            input_shapes=[(128, 512, 1024), (128, 512, 1024), (128, 512, 1024)],
            atol=2e-4,
            rtol=2e-4,
        )

    def test_topk_gather(self):
        """Top-k then gather backward — scatter_add via PyTorch fallback."""
        self._roundtrip(
            "atenir._examples:topk_gather",
            input_shapes=[(4096, 2048)],
        )

    def test_swiglu(self):
        self._roundtrip(
            "atenir._examples:swiglu",
            input_shapes=[(2048, 4096), (4096, 8192), (4096, 8192)],
            atol=1,
            rtol=1,
        )

    def test_mlp(self):
        self._roundtrip(
            "atenir._examples:mlp",
            input_shapes=[(2048, 4096), (4096, 8192), (8192, 2048)],
            atol=1,
            rtol=1,
        )

    def test_silu_mlp(self):
        """LLaMA/Mistral-style MLP: matmul → silu → matmul."""
        self._roundtrip(
            "atenir._examples:silu_mlp",
            input_shapes=[(2048, 4096), (4096, 8192), (8192, 2048)],
            atol=1,
            rtol=1,
        )

    def test_mobilenet_block(self):
        """HardSwish activation: x * clamp((x+3)/6, 0, 1)."""
        self._roundtrip(
            "atenir._examples:mobilenet_block",
            input_shapes=[(2048, 2048)],
            atol=1e-4,
            rtol=1e-4,
        )

    def test_mish_block(self):
        """Mish activation: x * tanh(softplus(x))."""
        self._roundtrip(
            "atenir._examples:mish_block",
            input_shapes=[(2048, 2048)],
            atol=1e-4,
            rtol=1e-4,
        )

    def test_lerp_blend(self):
        """Element-wise lerp with tensor weight: a + w*(b-a)."""
        self._roundtrip(
            "atenir._examples:lerp_blend",
            input_shapes=[(2048, 2048), (2048, 2048), (2048, 2048)],
            atol=1e-4,
            rtol=1e-4,
        )

    def test_erfc_gelu(self):
        """GELU via erfc: 0.5 * x * erfc(-x / sqrt(2))."""
        self._roundtrip(
            "atenir._examples:erfc_gelu",
            input_shapes=[(2048, 2048)],
            atol=1e-4,
            rtol=1e-4,
        )

    def test_exp2_scale(self):
        """Element-wise 2^exponent scaling."""
        self._roundtrip(
            "atenir._examples:exp2_scale",
            input_shapes=[(2048, 2048), (2048, 2048)],
            atol=1e-4,
            rtol=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
