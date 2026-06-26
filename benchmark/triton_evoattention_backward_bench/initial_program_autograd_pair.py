import torch
import triton
import triton.language as tl


def _next_power_of_2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


def _can_use_triton(q, k, v, res_mask=None, pair_bias=None):
    if not q.is_cuda:
        return False
    if q.dtype is not torch.float16:
        return False
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    if q.ndim != 5:
        return False
    _, _, n_res, _, dim = q.shape
    block_n = max(16, _next_power_of_2(n_res))
    block_d = max(16, _next_power_of_2(dim))
    return block_n <= 64 and block_d <= 128


# EVOLVE-BLOCK-START


@triton.jit
def _evo_attn_fwd_kernel(
    Q,
    K,
    V,
    RES_MASK,
    PAIR_BIAS,
    O,
    P_SAVE,
    B: tl.constexpr,
    S: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    h = pid % H
    s = (pid // H) % S
    b = pid // (S * H)

    rn = tl.arange(0, BLOCK_N)
    rd = tl.arange(0, BLOCK_D)

    x_offs = ((((b * S + s) * N + rn[:, None]) * H + h) * D + rd[None, :])
    x_mask = (rn[:, None] < N) & (rd[None, :] < D)

    q = tl.load(Q + x_offs, mask=x_mask, other=0.0).to(tl.float32)
    k = tl.load(K + x_offs, mask=x_mask, other=0.0)
    v = tl.load(V + x_offs, mask=x_mask, other=0.0)

    q_scaled = (q * SCALE).to(tl.float16)

    scores = tl.dot(q_scaled, tl.trans(k), out_dtype=tl.float32)
    scores = scores.to(tl.float16).to(tl.float32)

    ri = rn
    rj = rn
    pair_mask = (ri[:, None] < N) & (rj[None, :] < N)

    rm_offs = (b * S + s) * N + rj
    rm = tl.load(RES_MASK + rm_offs, mask=rj < N, other=0.0).to(tl.float32)

    pair_offs = (((b * H + h) * N + ri[:, None]) * N + rj[None, :])
    pb = tl.load(PAIR_BIAS + pair_offs, mask=pair_mask, other=0.0).to(tl.float32)

    scores = scores + rm[None, :] + pb
    scores = tl.where(rj[None, :] < N, scores, -3.4028234663852886e38)

    m = tl.max(scores, axis=1)
    p = tl.exp(scores - m[:, None])
    denom = tl.sum(p, axis=1)
    p = p / denom[:, None]

    p_offs = ((((b * S + s) * H + h) * N + ri[:, None]) * N + rj[None, :])
    tl.store(P_SAVE + p_offs, p, mask=pair_mask)

    out = tl.dot(p.to(tl.float16), v, out_dtype=tl.float32)
    tl.store(O + x_offs, out, mask=x_mask)


@triton.jit
def _evo_attn_bwd_kernel(
    DO,
    Q,
    K,
    V,
    P_SAVE,
    DQ,
    DK,
    DV,
    DPAIR_BIAS,
    B: tl.constexpr,
    S: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    h = pid % H
    s = (pid // H) % S
    b = pid // (S * H)

    rn = tl.arange(0, BLOCK_N)
    rd = tl.arange(0, BLOCK_D)
    ri = rn
    rj = rn

    x_offs = ((((b * S + s) * N + rn[:, None]) * H + h) * D + rd[None, :])
    x_mask = (rn[:, None] < N) & (rd[None, :] < D)

    q = tl.load(Q + x_offs, mask=x_mask, other=0.0)
    k = tl.load(K + x_offs, mask=x_mask, other=0.0)
    v = tl.load(V + x_offs, mask=x_mask, other=0.0)
    do = tl.load(DO + x_offs, mask=x_mask, other=0.0)

    p_mask = (ri[:, None] < N) & (rj[None, :] < N)
    p_offs = ((((b * S + s) * H + h) * N + ri[:, None]) * N + rj[None, :])
    p = tl.load(P_SAVE + p_offs, mask=p_mask, other=0.0).to(tl.float32)

    dv = tl.dot(tl.trans(p.to(tl.float16)), do, out_dtype=tl.float32)
    tl.store(DV + x_offs, dv, mask=x_mask)

    dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
    dp = dp.to(tl.float16).to(tl.float32)

    row_dot = tl.sum(dp * p, axis=1)
    ds = p * (dp - row_dot[:, None])
    ds = tl.where(p_mask, ds, 0.0)

    dpair_offs = (((b * H + h) * N + ri[:, None]) * N + rj[None, :])
    tl.atomic_add(DPAIR_BIAS + dpair_offs, ds, sem="relaxed", mask=p_mask)

    ds_h = ds.to(tl.float16)

    dq = tl.dot(ds_h, k, out_dtype=tl.float32)
    dq = dq * SCALE
    tl.store(DQ + x_offs, dq, mask=x_mask)

    q_scaled = (q.to(tl.float32) * SCALE).to(tl.float16)
    dk = tl.dot(tl.trans(ds_h), q_scaled, out_dtype=tl.float32)
    tl.store(DK + x_offs, dk, mask=x_mask)


def _evoattention_forward_triton(q, k, v, res_mask, pair_bias):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    res_mask = res_mask.contiguous()
    pair_bias = pair_bias.contiguous()

    B, S, N, H, D = q.shape
    block_n = max(16, _next_power_of_2(N))
    block_d = max(16, _next_power_of_2(D))
    scale = 1.0 / (D ** 0.5)

    y = torch.empty_like(q)
    p_save = torch.empty((B, S, H, N, N), device=q.device, dtype=torch.float32)

    grid = (B * S * H,)
    _evo_attn_fwd_kernel[grid](
        q,
        k,
        v,
        res_mask,
        pair_bias,
        y,
        p_save,
        B,
        S,
        N,
        H,
        D,
        scale,
        block_n,
        block_d,
        num_warps=1,
    )
    return y, (q, k, v, p_save)


def _evoattention_backward_triton(do, saved_tensors):
    q, k, v, p_save = saved_tensors
    do = do.contiguous()

    B, S, N, H, D = q.shape
    block_n = max(16, _next_power_of_2(N))
    block_d = max(16, _next_power_of_2(D))
    scale = 1.0 / (D ** 0.5)

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    d_pair_bias = torch.zeros((B, 1, H, N, N), device=q.device, dtype=torch.float32)

    grid = (B * S * H,)
    _evo_attn_bwd_kernel[grid](
        do,
        q,
        k,
        v,
        p_save,
        dq,
        dk,
        dv,
        d_pair_bias,
        B,
        S,
        N,
        H,
        D,
        scale,
        block_n,
        block_d,
        num_warps=1,
    )

    return dq, dk, dv, d_pair_bias


# EVOLVE-BLOCK-END


def _evoattention_forward_torch(q, k, v, res_mask, pair_bias):
    B, S, N, H, D = q.shape
    scale = 1.0 / (D ** 0.5)

    qt = q.permute(0, 1, 3, 2, 4).contiguous()
    kt = k.permute(0, 1, 3, 2, 4).contiguous()
    vt = v.permute(0, 1, 3, 2, 4).contiguous()

    q_scaled = (qt.to(torch.float32) * scale).to(q.dtype)
    scores = torch.matmul(q_scaled, kt.transpose(-1, -2)).to(torch.float32)
    scores = scores + res_mask.to(torch.float32) + pair_bias.to(torch.float32)
    p = torch.softmax(scores, dim=-1, dtype=torch.float32)

    yt = torch.matmul(p.to(v.dtype), vt)
    y = yt.permute(0, 1, 3, 2, 4).contiguous()
    return y, (q.contiguous(), k.contiguous(), v.contiguous(), p.contiguous())


def _evoattention_backward_torch(do, saved_tensors):
    q, k, v, p = saved_tensors
    B, S, N, H, D = q.shape
    scale = 1.0 / (D ** 0.5)

    qt = q.permute(0, 1, 3, 2, 4).contiguous()
    kt = k.permute(0, 1, 3, 2, 4).contiguous()
    vt = v.permute(0, 1, 3, 2, 4).contiguous()
    dot = do.permute(0, 1, 3, 2, 4).contiguous()

    dv_t = torch.matmul(p.to(dot.dtype).transpose(-1, -2), dot).to(v.dtype)

    dp = torch.matmul(dot, vt.transpose(-1, -2)).to(torch.float32)
    row_dot = torch.sum(dp * p, dim=-1, keepdim=True)
    ds = p * (dp - row_dot)

    dq_t = (torch.matmul(ds.to(k.dtype), kt).to(torch.float32) * scale).to(q.dtype)

    q_scaled = (qt.to(torch.float32) * scale).to(q.dtype)
    dk_t = torch.matmul(ds.to(q.dtype).transpose(-1, -2), q_scaled).to(k.dtype)

    d_pair_bias = ds.sum(dim=1, keepdim=True).to(torch.float32)

    dq = dq_t.permute(0, 1, 3, 2, 4).contiguous()
    dk = dk_t.permute(0, 1, 3, 2, 4).contiguous()
    dv = dv_t.permute(0, 1, 3, 2, 4).contiguous()

    return dq, dk, dv, d_pair_bias


def evoattention_forward_with_saved(q, k, v, res_mask, pair_bias):
    if _can_use_triton(q, k, v, res_mask, pair_bias):
        return _evoattention_forward_triton(q, k, v, res_mask, pair_bias)
    return _evoattention_forward_torch(q, k, v, res_mask, pair_bias)


def evoattention_backward_from_saved(do, saved_tensors):
    if isinstance(saved_tensors, torch.Tensor):
        raise ValueError("saved_tensors must contain q, k, v, and saved probabilities")
    q, k, v, p_save = saved_tensors

    if _can_use_triton(q, k, v) and do.is_cuda and do.dtype == torch.float16:
        return _evoattention_backward_triton(do, (q, k, v, p_save))

    return _evoattention_backward_torch(do, (q, k, v, p_save))
