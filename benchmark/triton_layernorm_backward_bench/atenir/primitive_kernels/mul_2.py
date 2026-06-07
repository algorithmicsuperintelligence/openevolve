"""Node: mul_2  |  aten.mul.Tensor
Inputs:  mul_1 [64, 256] f32,  scalar=256
Output:  [64, 256] f32
"""
import torch
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    triton = None
    HAS_TRITON = False

    class _TLStub:
        constexpr = type(None)
    tl = _TLStub()

_jit = triton.jit if HAS_TRITON else (lambda fn: fn)

ROWS = 64
COLS = 256
SCALAR = 256


@_jit
def _kernel(a_ptr, out_ptr, SCALAR: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr):
    """[R,C] * scalar  →  [R,C]"""
    row = tl.program_id(0)
    cols = tl.arange(0, COLS)
    a = tl.load(a_ptr + row * COLS + cols)
    tl.store(out_ptr + row * COLS + cols, a * SCALAR)


def run(a: torch.Tensor, scalar: int = SCALAR) -> torch.Tensor:
    """a: [R,C]  →  [R,C]"""
    R, C = a.shape
    out = torch.empty_like(a)
    _kernel[(R,)](a, out, SCALAR=scalar, ROWS=R, COLS=C)
    return out


def test():
    if not torch.cuda.is_available() or not HAS_TRITON:
        print("SKIP: no CUDA or Triton")
        return
    torch.manual_seed(42)
    a = torch.randn(ROWS, COLS, device="cuda")
    ref = a * SCALAR
    got = run(a)
    torch.testing.assert_close(got, ref, atol=2e-5, rtol=2e-5)
    print("PASS: mul_2")


if __name__ == "__main__":
    test()
