"""Node: mul_5  |  aten.mul.Tensor
Inputs:  div [64, 1] f32,  sub_2 [64, 256] f32  (div broadcast over cols)
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


@_jit
def _kernel(a_ptr, b_ptr, out_ptr, ROWS: tl.constexpr, COLS: tl.constexpr):
    """[R,1] * [R,C]  →  [R,C]"""
    row = tl.program_id(0)
    cols = tl.arange(0, COLS)
    a = tl.load(a_ptr + row)               # one scalar per row
    b = tl.load(b_ptr + row * COLS + cols)
    tl.store(out_ptr + row * COLS + cols, a * b)


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: [R,1], b: [R,C]  →  [R,C]"""
    R, C = b.shape
    out = torch.empty_like(b)
    _kernel[(R,)](a, b, out, ROWS=R, COLS=C)
    return out


def test():
    if not torch.cuda.is_available() or not HAS_TRITON:
        print("SKIP: no CUDA or Triton")
        return
    torch.manual_seed(42)
    a = torch.randn(ROWS, 1, device="cuda")
    b = torch.randn(ROWS, COLS, device="cuda")
    ref = a * b
    got = run(a, b)
    torch.testing.assert_close(got, ref, atol=2e-5, rtol=2e-5)
    print("PASS: mul_5")


if __name__ == "__main__":
    test()
