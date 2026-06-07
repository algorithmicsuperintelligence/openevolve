"""Reference optimized candidate for the LayerNorm backward benchmark."""

import torch


# EVOLVE-BLOCK-START
import triton
import triton.language as tl
from backward_naive_triton import layernorm_backward_naive_triton as _fallback_backward


@triton.jit
def _ln_bwd_fused_kernel(
    dy,
    x,
    w,
    dx,
    pdw,
    pdb,
    ROWS: tl.constexpr,
    H: tl.constexpr,
    EPS: tl.constexpr,
    BN: tl.constexpr,
    BR: tl.constexpr,
):
    pid = tl.program_id(0)
    rs = pid * BR + tl.arange(0, BR)
    cs = tl.arange(0, BN)
    cm = cs < H
    m = (rs[:, None] < ROWS) & cm[None, :]

    xv = tl.load(x + rs[:, None] * H + cs[None, :], mask=m, other=0.0).to(tl.float32)
    dyv = tl.load(dy + rs[:, None] * H + cs[None, :], mask=m, other=0.0).to(tl.float32)
    wv = tl.load(w + cs, mask=cm, other=0.0).to(tl.float32)

    mean = tl.sum(xv, axis=1) / H
    xc = tl.where(cm[None, :], xv - mean[:, None], 0.0)
    rstd = tl.rsqrt(tl.sum(xc * xc, axis=1) / H + EPS)
    xh = xc * rstd[:, None]

    one = dyv * wv[None, :]
    mo = tl.sum(one, axis=1) / H
    mxo = tl.sum(one * xh, axis=1) / H
    dxv = (one - mo[:, None] - xh * mxo[:, None]) * rstd[:, None]
    tl.store(dx + rs[:, None] * H + cs[None, :], dxv, mask=m)

    tl.store(pdw + pid * H + cs, tl.sum(dyv * xh, axis=0), mask=cm)
    tl.store(pdb + pid * H + cs, tl.sum(dyv, axis=0), mask=cm)


@triton.jit
def _ln_bwd_sum_kernel(
    pdw,
    pdb,
    dw,
    db,
    NB: tl.constexpr,
    H: tl.constexpr,
    BB: tl.constexpr,
    BC: tl.constexpr,
):
    pc = tl.program_id(0) * BC + tl.arange(0, BC)
    rb = tl.arange(0, BB)
    m = (rb[:, None] < NB) & (pc[None, :] < H)
    a = tl.load(pdw + rb[:, None] * H + pc[None, :], mask=m, other=0.0).to(tl.float32)
    b = tl.load(pdb + rb[:, None] * H + pc[None, :], mask=m, other=0.0).to(tl.float32)
    tl.store(dw + pc, tl.sum(a, axis=0), mask=pc < H)
    tl.store(db + pc, tl.sum(b, axis=0), mask=pc < H)


def _seed_backward(dy, x, weight, bias, eps):
    rows, h = x.shape
    if rows == 0 or h > 8192:
        return _fallback_backward(dy, x, weight, bias, eps)

    br = 8 if h <= 256 else 4 if h <= 512 else 2 if h <= 1024 else 1
    bn = 1 << (h - 1).bit_length()
    nb = triton.cdiv(rows, br)

    # Keep the second-stage reduction compact; very tall cases use the safe baseline.
    if nb > 2048:
        return _fallback_backward(dy, x, weight, bias, eps)

    dx = torch.empty_like(x)
    dw = torch.empty_like(weight)
    db = torch.empty_like(bias)
    pdw = torch.empty((nb, h), device=x.device, dtype=torch.float32)
    pdb = torch.empty((nb, h), device=x.device, dtype=torch.float32)

    _ln_bwd_fused_kernel[(nb,)](
        dy,
        x,
        weight,
        dx,
        pdw,
        pdb,
        rows,
        h,
        eps,
        bn,
        br,
        num_warps=8 if bn >= 2048 else 4,
        num_stages=3,
    )

    bb = 1 << (nb - 1).bit_length()
    bc = 16 if bb > 1024 else 32 if bb > 512 else 64 if bb > 128 else 128
    sw = 1 if bb <= 32 else 4 if bb <= 128 else 8
    _ln_bwd_sum_kernel[(triton.cdiv(h, bc),)](
        pdw,
        pdb,
        dw,
        db,
        nb,
        h,
        bb,
        bc,
        num_warps=sw,
        num_stages=3,
    )
    return dx, dw, db
# EVOLVE-BLOCK-END


def layernorm_backward_triton(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (dx, dweight, dbias) for row-wise LayerNorm."""
    return _seed_backward(dy, x, weight, bias, eps)
