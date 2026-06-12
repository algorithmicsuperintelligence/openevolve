import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START

_MAX_ROW_STATS_BLOCK_C = 65536
_MAX_FINALIZE_BLOCK_RB = 65536


@triton.jit
def _ln_bwd_row_stats_kernel(
    dy_ptr,
    x_ptr,
    weight_ptr,
    mean_ptr,
    rstd_ptr,
    coeff_ptr,
    corr_ptr,
    eps: tl.constexpr,
    C,
    dy_s0,
    dy_s1,
    x_s0,
    x_s1,
    w_s0,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    mask = cols < C

    x = tl.load(x_ptr + row * x_s0 + cols * x_s1, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + row * dy_s0 + cols * dy_s1, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols * w_s0, mask=mask, other=0.0).to(tl.float32)

    inv_c = 1.0 / C

    x_sum = tl.sum(tl.where(mask, x, 0.0), axis=0)
    mean = x_sum * inv_c

    xm = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xm * xm, axis=0) * inv_c
    rstd = tl.rsqrt(var + eps)

    g = dy * w
    sum_g = tl.sum(g, axis=0)
    sum_g_xm = tl.sum(g * xm, axis=0)
    sum_xm = tl.sum(xm, axis=0)

    rstd2 = rstd * rstd
    rstd3 = rstd2 * rstd

    coeff = -sum_g_xm * rstd3 * inv_c
    corr = (-rstd * sum_g - coeff * sum_xm) * inv_c

    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)
    tl.store(coeff_ptr + row, coeff)
    tl.store(corr_ptr + row, corr)


@triton.jit
def _ln_bwd_dx_dparam_partials_kernel(
    dy_ptr,
    x_ptr,
    weight_ptr,
    mean_ptr,
    rstd_ptr,
    coeff_ptr,
    corr_ptr,
    dx_ptr,
    partial_dbias_ptr,
    partial_dweight_ptr,
    R,
    C,
    dy_s0,
    dy_s1,
    x_s0,
    x_s1,
    w_s0,
    dx_s0,
    dx_s1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rb = tl.program_id(0)
    cb = tl.program_id(1)

    rows = rb * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = cb * BLOCK_N + tl.arange(0, BLOCK_N)

    row_mask = rows < R
    col_mask = cols < C
    mask = row_mask[:, None] & col_mask[None, :]

    dy = tl.load(
        dy_ptr + rows[:, None] * dy_s0 + cols[None, :] * dy_s1,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    x = tl.load(
        x_ptr + rows[:, None] * x_s0 + cols[None, :] * x_s1,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    w = tl.load(weight_ptr + cols * w_s0, mask=col_mask, other=0.0).to(tl.float32)

    mean = tl.load(mean_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    coeff = tl.load(coeff_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    corr = tl.load(corr_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

    xm = x - mean[:, None]
    xhat = xm * rstd[:, None]
    g = dy * w[None, :]

    dx = g * rstd[:, None] + coeff[:, None] * xm + corr[:, None]
    tl.store(
        dx_ptr + rows[:, None] * dx_s0 + cols[None, :] * dx_s1,
        dx,
        mask=mask,
    )

    p_dbias = tl.sum(tl.where(mask, dy, 0.0), axis=0)
    p_dweight = tl.sum(tl.where(mask, dy * xhat, 0.0), axis=0)

    tl.store(partial_dbias_ptr + rb * C + cols, p_dbias, mask=col_mask)
    tl.store(partial_dweight_ptr + rb * C + cols, p_dweight, mask=col_mask)


@triton.jit
def _ln_bwd_dparam_finalize_kernel(
    partial_dbias_ptr,
    partial_dweight_ptr,
    dbias_ptr,
    dweight_ptr,
    num_row_blocks,
    C,
    dbias_s0,
    dweight_s0,
    BLOCK_RB: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cb = tl.program_id(0)

    rbs = tl.arange(0, BLOCK_RB)
    cols = cb * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (rbs[:, None] < num_row_blocks) & (cols[None, :] < C)

    pdb = tl.load(
        partial_dbias_ptr + rbs[:, None] * C + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    pdw = tl.load(
        partial_dweight_ptr + rbs[:, None] * C + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    db = tl.sum(pdb, axis=0)
    dw = tl.sum(pdw, axis=0)

    col_mask = cols < C
    tl.store(dbias_ptr + cols * dbias_s0, db, mask=col_mask)
    tl.store(dweight_ptr + cols * dweight_s0, dw, mask=col_mask)


def _pow2_at_least_1(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _num_warps_for_width(block: int) -> int:
    block = int(block)
    if block >= 4096:
        return 8
    if block >= 1024:
        return 4
    return 1


def _choose_block_m(R: int, C: int) -> int:
    if C >= 1024:
        base = 16
    elif C >= 256:
        base = 32
    else:
        base = 64

    if R <= 0:
        return base

    needed = triton.cdiv(int(R), _MAX_FINALIZE_BLOCK_RB)
    return max(base, int(needed))


def _choose_block_n(C: int) -> int:
    if C <= 0:
        return 1
    return min(128, _pow2_at_least_1(int(C)))


def _launch_layernorm_backward(dy, x, weight, bias, eps):
    R = int(x.shape[0])
    C = int(x.shape[1])

    dx = torch.empty_like(x)
    dweight = torch.empty_like(weight)
    dbias = torch.empty_like(bias)

    if C == 0:
        if R == 0:
            return dx, dweight, dbias
        raise ValueError("layernorm_backward_triton does not support nonzero rows with C == 0")

    if C > _MAX_ROW_STATS_BLOCK_C:
        raise NotImplementedError(
            f"normalized dimension C={C} exceeds supported Triton row block size "
            f"{_MAX_ROW_STATS_BLOCK_C}"
        )

    mean = torch.empty((R,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((R,), device=x.device, dtype=torch.float32)
    coeff = torch.empty((R,), device=x.device, dtype=torch.float32)
    corr = torch.empty((R,), device=x.device, dtype=torch.float32)

    BLOCK_M = _choose_block_m(R, C)
    BLOCK_N = _choose_block_n(C)
    num_row_blocks = triton.cdiv(R, BLOCK_M)

    partial_dbias = torch.empty((num_row_blocks, C), device=x.device, dtype=torch.float32)
    partial_dweight = torch.empty((num_row_blocks, C), device=x.device, dtype=torch.float32)

    BLOCK_C_STATS = _pow2_at_least_1(C)

    if R > 0:
        _ln_bwd_row_stats_kernel[(R,)](
            dy,
            x,
            weight,
            mean,
            rstd,
            coeff,
            corr,
            float(eps),
            C,
            dy.stride(0),
            dy.stride(1),
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            BLOCK_C=BLOCK_C_STATS,
            num_warps=_num_warps_for_width(BLOCK_C_STATS),
        )

        grid_dx = (num_row_blocks, triton.cdiv(C, BLOCK_N))
        _ln_bwd_dx_dparam_partials_kernel[grid_dx](
            dy,
            x,
            weight,
            mean,
            rstd,
            coeff,
            corr,
            dx,
            partial_dbias,
            partial_dweight,
            R,
            C,
            dy.stride(0),
            dy.stride(1),
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            dx.stride(0),
            dx.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )

    BLOCK_RB = _pow2_at_least_1(num_row_blocks)
    if BLOCK_RB > _MAX_FINALIZE_BLOCK_RB:
        raise NotImplementedError(
            f"number of row partial blocks={num_row_blocks} exceeds supported finalize block size"
        )

    grid_finalize = (triton.cdiv(C, BLOCK_N),)
    _ln_bwd_dparam_finalize_kernel[grid_finalize](
        partial_dbias,
        partial_dweight,
        dbias,
        dweight,
        num_row_blocks,
        C,
        dbias.stride(0),
        dweight.stride(0),
        BLOCK_RB=BLOCK_RB,
        BLOCK_N=BLOCK_N,
        num_warps=_num_warps_for_width(BLOCK_RB),
    )

    return dx, dweight, dbias


# EVOLVE-BLOCK-END


def layernorm_backward_triton(dy, x, weight, bias, eps=1e-5):
    if not isinstance(dy, torch.Tensor) or not isinstance(x, torch.Tensor):
        raise TypeError("dy and x must be torch.Tensor instances")
    if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise TypeError("weight and bias must be torch.Tensor instances")

    if not dy.is_cuda or not x.is_cuda or not weight.is_cuda or not bias.is_cuda:
        raise ValueError("layernorm_backward_triton supports CUDA tensors only")

    if x.ndim != 2 or dy.ndim != 2:
        raise ValueError("x and dy must be rank-2 tensors with shape [R, C]")
    if weight.ndim != 1 or bias.ndim != 1:
        raise ValueError("weight and bias must be rank-1 tensors with shape [C]")

    if tuple(dy.shape) != tuple(x.shape):
        raise ValueError("dy and x must have the same shape")

    C = int(x.shape[1])
    if int(weight.shape[0]) != C or int(bias.shape[0]) != C:
        raise ValueError("weight and bias must have length equal to x.shape[1]")

    allowed = (torch.float16, torch.float32)
    if (
        dy.dtype not in allowed
        or x.dtype not in allowed
        or weight.dtype not in allowed
        or bias.dtype not in allowed
    ):
        raise TypeError("dy, x, weight, and bias must have dtype torch.float16 or torch.float32")

    if dy.device != x.device or weight.device != x.device or bias.device != x.device:
        raise ValueError("all inputs must be on the same CUDA device")

    return _launch_layernorm_backward(dy, x, weight, bias, eps)
