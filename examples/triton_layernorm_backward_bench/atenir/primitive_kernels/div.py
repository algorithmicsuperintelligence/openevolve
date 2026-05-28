"""Node: div  |  aten.div.Tensor
Inputs:  rstd_1 [64, 1] f32,  scalar=256
Output:  [64, 1] f32
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
SCALAR = 256


@_jit
def _kernel(a_ptr, out_ptr, SCALAR: tl.constexpr, ROWS: tl.constexpr):
    """[R,1] / scalar  →  [R,1]"""
    row = tl.program_id(0)
    a = tl.load(a_ptr + row)
    tl.store(out_ptr + row, a / SCALAR)


def run(a: torch.Tensor, scalar: int = SCALAR) -> torch.Tensor:
    """a: [R,1]  →  [R,1]"""
    R = a.shape[0]
    out = torch.empty_like(a)
    _kernel[(R,)](a, out, SCALAR=scalar, ROWS=R)
    return out


def test():
    if not torch.cuda.is_available() or not HAS_TRITON:
        print("SKIP: no CUDA or Triton")
        return
    torch.manual_seed(42)
    a = torch.randn(ROWS, 1, device="cuda")
    ref = a / SCALAR
    got = run(a)
    torch.testing.assert_close(got, ref, atol=2e-5, rtol=2e-5)
    print("PASS: div")


if __name__ == "__main__":
    test()
