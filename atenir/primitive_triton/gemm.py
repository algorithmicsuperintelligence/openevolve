"""Primitive Triton GEMM kernels: mm, bmm, addmm.

Derived directly from the Inductor Jinja2 templates in
  atenir/primitive_triton/templates/triton_mm.py.jinja
  atenir/primitive_triton/templates/triton_bmm.py.jinja

What the templates provided and how they map here:
  {{def_kernel(...)}}         → @triton.jit def _mm_kernel(A, B, C, M, N, K, strides…, constexprs…)
  {{size("A", 0)}}            → explicit M/N/K parameters
  {{stride("A", 0)}}          → explicit stride_am/ak/bk/bn/cm/cn parameters
  {{load_input("A", …)}}      → tl.load with computed pointer offset + mask
  {{store_output(…)}}         → tl.store with mask
  INDEX_DTYPE                 → tl.int32 (primitive; no int64 needed)
  ACC_TYPE                    → tl.float32 (always accumulate in fp32)
  EVEN_K = False              → always use K-tail masking (correct for any K)
  USE_FAST_ACCUM = False      → acc += tl.dot(…) form
  GROUP_M = 8                 → kept as constexpr for L2 reordering
  ALLOW_TF32                  → passed as constexpr, default True

Conv templates (triton_conv2d_bwd_input/weight.py.jinja) require per-call
shape metadata (padding, dilation, stride, groups, H/W) that is not present
in the AtenIR graph JSON; those remain PyTorch fallbacks in dispatch.py.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# Default tile sizes. All must be powers-of-2 and ≥ 16 for tl.dot.
_BLOCK_M = 64
_BLOCK_N = 64
_BLOCK_K = 32
_GROUP_M = 8


# ── Triton JIT kernels ────────────────────────────────────────────────────────

if HAS_TRITON:

    @triton.jit
    def _mm_kernel(
        # pointers
        A_ptr, B_ptr, C_ptr,
        # problem size
        M, N, K,
        # strides for A [M, K]
        stride_am, stride_ak,
        # strides for B [K, N]
        stride_bk, stride_bn,
        # strides for C [M, N]
        stride_cm, stride_cn,
        # tile sizes (constexpr)
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
    ):
        """2-D matrix multiply: C = A @ B.

        Translated from triton_mm.py.jinja.
        Grid: (cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N),)
        """
        pid = tl.program_id(0)
        grid_m = tl.cdiv(M, BLOCK_M)
        grid_n = tl.cdiv(N, BLOCK_N)

        # Re-order programs for better L2 reuse (from template: GROUP_M swizzle).
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // group_size

        # Row / column ranges for this tile.
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int32)
        rk = tl.arange(0, BLOCK_K).to(tl.int32)

        # Initial pointers to the first K-slice of A and B.
        A = A_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
        B = B_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
            k_remaining = K - k_idx * BLOCK_K
            # K-tail masking (EVEN_K=False path from template).
            a_mask = (rm[:, None] < M) & (rk[None, :] < k_remaining)
            b_mask = (rk[:, None] < k_remaining) & (rn[None, :] < N)
            a = tl.load(A, mask=a_mask, other=0.0)
            b = tl.load(B, mask=b_mask, other=0.0)
            # USE_FAST_ACCUM=False path from template.
            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A += BLOCK_K * stride_ak
            B += BLOCK_K * stride_bk

        # Rematerialise rm / rn (from template comment: "to save registers").
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int32)
        idx_m = rm[:, None]
        idx_n = rn[None, :]
        mask = (idx_m < M) & (idx_n < N)

        # store_output macro from template.
        C = C_ptr + idx_m * stride_cm + idx_n * stride_cn
        tl.store(C, acc.to(C_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _addmm_kernel(
        # pointers
        A_ptr, B_ptr, Bias_ptr, C_ptr,
        # problem size
        M, N, K,
        # strides for A [M, K]
        stride_am, stride_ak,
        # strides for B [K, N]
        stride_bk, stride_bn,
        # strides for Bias (supports row-vector [N] via stride_bm=0)
        stride_bias_m, stride_bias_n,
        # strides for C [M, N]
        stride_cm, stride_cn,
        # epilogue scalars
        alpha, beta,
        # tile sizes (constexpr)
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
    ):
        """C = beta * Bias + alpha * (A @ B).

        Derived from triton_mm.py.jinja + epilogue (store_output with bias).
        Bias strides allow [M,N], [N] (stride_bm=0), or [M,1] (stride_bn=0).
        Grid: (cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N),)
        """
        pid = tl.program_id(0)
        grid_m = tl.cdiv(M, BLOCK_M)
        grid_n = tl.cdiv(N, BLOCK_N)

        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // group_size

        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int32)
        rk = tl.arange(0, BLOCK_K).to(tl.int32)

        A = A_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
        B = B_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
            k_remaining = K - k_idx * BLOCK_K
            a_mask = (rm[:, None] < M) & (rk[None, :] < k_remaining)
            b_mask = (rk[:, None] < k_remaining) & (rn[None, :] < N)
            a = tl.load(A, mask=a_mask, other=0.0)
            b = tl.load(B, mask=b_mask, other=0.0)
            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A += BLOCK_K * stride_ak
            B += BLOCK_K * stride_bk

        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int32)
        idx_m = rm[:, None]
        idx_n = rn[None, :]
        mask = (idx_m < M) & (idx_n < N)

        # Epilogue: load bias tile and apply beta/alpha.
        bias_ptrs = Bias_ptr + idx_m * stride_bias_m + idx_n * stride_bias_n
        bias = tl.load(bias_ptrs, mask=mask, other=0.0)
        result = beta * bias.to(tl.float32) + alpha * acc

        C = C_ptr + idx_m * stride_cm + idx_n * stride_cn
        tl.store(C, result.to(C_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _bmm_kernel(
        # pointers
        A_ptr, B_ptr, C_ptr,
        # problem size
        BATCH, M, N, K,
        # strides for A [B, M, K]
        stride_aq, stride_am, stride_ak,
        # strides for B [B, K, N]
        stride_bq, stride_bk, stride_bn,
        # strides for C [B, M, N]
        stride_cq, stride_cm, stride_cn,
        # tile sizes (constexpr)
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        EVEN_K: tl.constexpr,
    ):
        """Batched matrix multiply: C[b] = A[b] @ B[b].

        Translated from triton_bmm.py.jinja.
        Grid: (cdiv(M,BM)*cdiv(N,BN), min(BATCH,65535), cdiv(BATCH,65535))
        The 3-axis batch split handles BATCH > 65535 (GPU grid limit on dim 1).
        """
        pid = tl.program_id(0)
        grid_m = tl.cdiv(M, BLOCK_M)
        grid_n = tl.cdiv(N, BLOCK_N)

        # L2 reordering (from template).
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // group_size

        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int32)

        # Clamp indices for safe pointer arithmetic (from template: % M and % N).
        ram = rm % M
        rbn = rn % N
        rk = tl.arange(0, BLOCK_K).to(tl.int32)

        # Batch index from the 2-D batch grid (program_id 1 and 2).
        idx_q = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)).to(tl.int32)
        idx_q_clamped = tl.minimum(idx_q, BATCH - 1)

        # Offset A and B to this batch element and M/N tile (from template).
        A = A_ptr + (
            ram[:, None] * stride_am
            + rk[None, :] * stride_ak
            + idx_q_clamped * stride_aq
        )
        B = B_ptr + (
            rk[:, None] * stride_bk
            + rbn[None, :] * stride_bn
            + idx_q_clamped * stride_bq
        )

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # K loop counting down — matches triton_bmm.py.jinja exactly.
        for k in range(K, 0, -BLOCK_K):
            if EVEN_K:
                a = tl.load(A)
                b = tl.load(B)
            else:
                a = tl.load(A, mask=rk[None, :] < k, other=0.0)
                b = tl.load(B, mask=rk[:, None] < k, other=0.0)
            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A += BLOCK_K * stride_ak
            B += BLOCK_K * stride_bk

        # Rematerialise for store.
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int32)
        idx_q = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)).to(tl.int32)
        idx_m = rm[:, None]
        idx_n = rn[None, :]
        mask = (idx_m < M) & (idx_n < N) & (idx_q < BATCH)

        C = C_ptr + idx_m * stride_cm + idx_n * stride_cn + idx_q * stride_cq
        tl.store(C, acc.to(C_ptr.dtype.element_ty), mask=mask)


# ── Python wrappers ───────────────────────────────────────────────────────────


def _check_cuda(t: torch.Tensor, name: str) -> None:
    if not t.is_cuda:
        raise ValueError(f"gemm: {name} must be a CUDA tensor")


def mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B.  Shapes: [M, K] @ [K, N] → [M, N]."""
    if not HAS_TRITON or not a.is_cuda:
        return torch.mm(a, b)
    assert a.ndim == 2 and b.ndim == 2, "mm: inputs must be 2-D"
    assert a.shape[1] == b.shape[0], f"mm: K mismatch {a.shape[1]} vs {b.shape[0]}"
    M, K = a.shape
    _, N = b.shape

    a = a.contiguous()
    b = b.contiguous()
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)

    grid = (triton.cdiv(M, _BLOCK_M) * triton.cdiv(N, _BLOCK_N),)
    _mm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
        GROUP_M=_GROUP_M,
        ALLOW_TF32=False,
    )
    return c


def addmm(
    bias: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> torch.Tensor:
    """C = beta * bias + alpha * (A @ B).

    bias may be [M, N], [N] (broadcast along rows), or [M, 1] (broadcast along cols).
    """
    if not HAS_TRITON or not a.is_cuda:
        return torch.addmm(bias, a, b, beta=beta, alpha=alpha)
    assert a.ndim == 2 and b.ndim == 2, "addmm: A/B must be 2-D"
    M, K = a.shape
    _, N = b.shape

    a = a.contiguous()
    b = b.contiguous()

    # Expand bias strides to [M, N] semantics via PyTorch's stride mechanism.
    # A 1-D bias [N] gets stride (0, 1) after expand; we compute this manually.
    bias = bias.contiguous()
    if bias.ndim == 1:
        assert bias.shape[0] == N, f"addmm: bias shape {bias.shape} incompatible with N={N}"
        stride_bias_m, stride_bias_n = 0, bias.stride(0)
    elif bias.ndim == 2:
        assert bias.shape[0] in (1, M) and bias.shape[1] in (1, N)
        stride_bias_m = bias.stride(0) if bias.shape[0] == M else 0
        stride_bias_n = bias.stride(1) if bias.shape[1] == N else 0
    else:
        raise ValueError(f"addmm: bias must be 1-D or 2-D, got {bias.ndim}-D")

    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, _BLOCK_M) * triton.cdiv(N, _BLOCK_N),)
    _addmm_kernel[grid](
        a, b, bias, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        stride_bias_m, stride_bias_n,
        c.stride(0), c.stride(1),
        float(alpha), float(beta),
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
        GROUP_M=_GROUP_M,
        ALLOW_TF32=False,
    )
    return c


def bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C[i] = A[i] @ B[i].  Shapes: [B, M, K] @ [B, K, N] → [B, M, N]."""
    if not HAS_TRITON or not a.is_cuda:
        return torch.bmm(a, b)
    assert a.ndim == 3 and b.ndim == 3, "bmm: inputs must be 3-D"
    assert a.shape[0] == b.shape[0], f"bmm: batch mismatch {a.shape[0]} vs {b.shape[0]}"
    assert a.shape[2] == b.shape[1], f"bmm: K mismatch {a.shape[2]} vs {b.shape[1]}"
    BATCH, M, K = a.shape
    _, _, N = b.shape

    a = a.contiguous()
    b = b.contiguous()
    c = torch.empty(BATCH, M, N, device=a.device, dtype=a.dtype)

    tiles_mn = triton.cdiv(M, _BLOCK_M) * triton.cdiv(N, _BLOCK_N)
    # Split BATCH across grid dims 1 and 2 to handle BATCH > 65535.
    batch_y = min(BATCH, 65535)
    batch_z = triton.cdiv(BATCH, 65535)
    grid = (tiles_mn, batch_y, batch_z)

    even_k = K % _BLOCK_K == 0
    _bmm_kernel[grid](
        a, b, c,
        BATCH, M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
        GROUP_M=_GROUP_M,
        ALLOW_TF32=False,
        EVEN_K=even_k,
    )
    return c
