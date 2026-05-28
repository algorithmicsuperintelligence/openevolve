"""Node: sum_4  |  aten.sum.dim_IntList
Inputs:  grad_out_1 [64, 256] f32,  reduction_dims=[0],  keepdim=False
Output:  [256] f32
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
def _kernel(a_ptr, out_ptr, ROWS: tl.constexpr, COLS: tl.constexpr):
    """sum([R,C], dim=0)  →  [C]"""
    col = tl.program_id(0)
    rows = tl.arange(0, ROWS)
    a = tl.load(a_ptr + rows * COLS + col)
    tl.store(out_ptr + col, tl.sum(a, axis=0))


def run(a: torch.Tensor) -> torch.Tensor:
    """a: [R,C]  →  [C]"""
    R, C = a.shape
    out = torch.empty(C, device=a.device, dtype=a.dtype)
    _kernel[(C,)](a, out, ROWS=R, COLS=C)
    return out


def test():
    if not torch.cuda.is_available() or not HAS_TRITON:
        print("SKIP: no CUDA or Triton")
        return
    torch.manual_seed(42)
    a = torch.randn(ROWS, COLS, device="cuda")
    ref = torch.sum(a, dim=0)
    got = run(a)
    torch.testing.assert_close(got, ref, atol=2e-5, rtol=2e-5)
    print("PASS: sum_4")


if __name__ == "__main__":
    test()
