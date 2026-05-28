"""Triton kernels for aten.gather and aten.scatter_add.

Two kernels per op cover the two 2D cases (dim=0 and dim=1).
For non-2D tensors the functions fall back to PyTorch.

gather(input, dim, index)      -- indexed read
scatter_add(self, dim, index, src) -- indexed atomic accumulate into clone of self
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


if HAS_TRITON:

    @triton.jit
    def _gather_1d_kernel(input_ptr, index_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N

        idx = tl.load(index_ptr + offs, mask=mask, other=0).to(tl.int64)
        val = tl.load(input_ptr + idx, mask=mask, other=0.0)

        tl.store(out_ptr + offs, val, mask=mask)

    @triton.jit
    def _scatter_add_1d_kernel(out_ptr, index_ptr, src_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N

        idx = tl.load(index_ptr + offs, mask=mask, other=0).to(tl.int64)
        val = tl.load(src_ptr + offs, mask=mask, other=0.0)

        tl.atomic_add(out_ptr + idx, val, mask=mask)

    @triton.jit
    def _gather_dim1_kernel(input_ptr, index_ptr, out_ptr, C_in, C_idx, BLOCK_C: tl.constexpr):
        """gather([R, C_in], dim=1, index=[R, C_idx]) → [R, C_idx].

        out[r, j] = input[r, index[r, j]]
        One program per row.
        """
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_C)
        mask = offs < C_idx
        idx = tl.load(index_ptr + row * C_idx + offs, mask=mask, other=0).to(tl.int64)
        val = tl.load(input_ptr + row * C_in + idx, mask=mask, other=0.0)
        tl.store(out_ptr + row * C_idx + offs, val, mask=mask)

    @triton.jit
    def _gather_dim0_kernel(input_ptr, index_ptr, out_ptr, C, BLOCK_C: tl.constexpr):
        """gather([R_in, C], dim=0, index=[R_idx, C]) → [R_idx, C].

        out[r, c] = input[index[r, c], c]
        One program per row of index/out.
        """
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_C)
        mask = offs < C
        idx = tl.load(index_ptr + row * C + offs, mask=mask, other=0).to(tl.int64)
        val = tl.load(input_ptr + idx * C + offs, mask=mask, other=0.0)
        tl.store(out_ptr + row * C + offs, val, mask=mask)

    @triton.jit
    def _scatter_add_dim1_kernel(out_ptr, index_ptr, src_ptr, C_out, C_idx, BLOCK_C: tl.constexpr):
        """scatter_add into dim=1.

        out[r, index[r, k]] += src[r, k]
        One program per row.
        """
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_C)
        mask = offs < C_idx
        idx = tl.load(index_ptr + row * C_idx + offs, mask=mask, other=0).to(tl.int64)
        val = tl.load(src_ptr + row * C_idx + offs, mask=mask, other=0.0)
        tl.atomic_add(out_ptr + row * C_out + idx, val, mask=mask)

    @triton.jit
    def _scatter_add_dim0_kernel(out_ptr, index_ptr, src_ptr, C, BLOCK_C: tl.constexpr):
        """scatter_add into dim=0.

        out[index[r, c], c] += src[r, c]
        One program per row of src/index.
        """
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_C)
        mask = offs < C
        idx = tl.load(index_ptr + row * C + offs, mask=mask, other=0).to(tl.int64)
        val = tl.load(src_ptr + row * C + offs, mask=mask, other=0.0)
        tl.atomic_add(out_ptr + idx * C + offs, val, mask=mask)


def gather(input: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    actual_dim = int(dim) % input.ndim

    if HAS_TRITON and input.ndim == 1 and index.ndim == 1 and input.is_cuda:
        input = input.contiguous()
        index = index.contiguous()
        out = torch.empty(index.shape, device=input.device, dtype=input.dtype)

        BLOCK = min(_next_pow2(index.numel()), 1024)
        grid = (triton.cdiv(index.numel(), BLOCK),)
        _gather_1d_kernel[grid](input, index, out, index.numel(), BLOCK=BLOCK)
        return out

    if not HAS_TRITON or input.ndim != 2 or index.ndim != 2 or not input.is_cuda:
        return torch.gather(input, actual_dim, index)

    input = input.contiguous()
    index = index.contiguous()
    out = torch.empty(index.shape, device=input.device, dtype=input.dtype)

    if actual_dim == 1:
        R_idx, C_idx = index.shape
        _, C_in = input.shape
        BLOCK_C = min(_next_pow2(C_idx), 65536)
        _gather_dim1_kernel[(R_idx,)](input, index, out, C_in, C_idx, BLOCK_C=BLOCK_C)
    else:
        R_idx, C = index.shape
        BLOCK_C = min(_next_pow2(C), 65536)
        _gather_dim0_kernel[(R_idx,)](input, index, out, C, BLOCK_C=BLOCK_C)

    return out


def scatter_add(
    self_t: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor
) -> torch.Tensor:
    actual_dim = int(dim) % self_t.ndim

    if HAS_TRITON and self_t.ndim == 1 and index.ndim == 1 and src.ndim == 1 and self_t.is_cuda:
        out = self_t.clone().contiguous()
        index = index.contiguous()
        src = src.contiguous()

        BLOCK = min(_next_pow2(index.numel()), 1024)
        grid = (triton.cdiv(index.numel(), BLOCK),)
        _scatter_add_1d_kernel[grid](out, index, src, index.numel(), BLOCK=BLOCK)
        return out

    if not HAS_TRITON or self_t.ndim != 2 or index.ndim != 2 or not self_t.is_cuda:
        out = self_t.clone()
        out.scatter_add_(actual_dim, index, src)
        return out

    out = self_t.clone().contiguous()
    index = index.contiguous()
    src = src.contiguous()

    if actual_dim == 1:
        R, C_out = out.shape
        _, C_idx = index.shape
        BLOCK_C = min(_next_pow2(C_idx), 65536)
        _scatter_add_dim1_kernel[(R,)](out, index, src, C_out, C_idx, BLOCK_C=BLOCK_C)
    else:
        R_src, C = index.shape
        BLOCK_C = min(_next_pow2(C), 65536)
        _scatter_add_dim0_kernel[(R_src,)](out, index, src, C, BLOCK_C=BLOCK_C)

    return out
