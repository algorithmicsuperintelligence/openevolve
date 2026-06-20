"""Task spec for the EvoformerAttention (3D EvoAttention) Triton backward benchmark."""

from dataclasses import dataclass


CANDIDATE_FN_NAME = "evoattention_backward_triton"
OUTPUT_NAMES = ("dq", "dk", "dv", "d_pair_bias")
# OpenEvolve speedup uses torch_oracle (PyTorch autograd over the reference
# attention forward). Override with BASELINE_PROGRAM_PATH only for legacy
# ablations against a fixed seed program.
PERFORMANCE_BASELINE = "pytorch_autograd"


@dataclass(frozen=True)
class TestCase:
    b: int
    n_seq: int
    head: int
    n_res: int
    dim: int
    dtype_name: str
    atol_value: float
    rtol_value: float


# Correctness: small shapes (incl. non-tile-aligned N_res), both half dtypes.
# N_seq is kept small so the d_pair_bias reduction over the MSA axis stays
# well-conditioned. Tolerances are attention-appropriate (many fp16/bf16
# accumulations through two matmuls and a softmax jacobian).
CORRECTNESS_CASES = [
    TestCase(1, 1, 4, 23, 8, "float16", 2e-2, 2e-2),
    TestCase(1, 2, 4, 64, 16, "float16", 2e-2, 2e-2),
    TestCase(1, 1, 8, 80, 32, "bfloat16", 4e-2, 4e-2),
    TestCase(2, 2, 4, 128, 64, "bfloat16", 4e-2, 4e-2),
]

# Benchmark: AF3-relevant shapes. Two regimes from MegaFold:
#   - attention-pair-bias: N_seq=1, many heads, larger residue crop, Dim=64
#   - triangle attention:  large N_seq, few heads, Dim=32
_EVOATTN_BENCHMARK_SHAPES = [
    # (B, N_seq, Head, N_res, Dim)
    (1, 1, 16, 128, 64),
    (1, 1, 16, 256, 64),
    (1, 1, 16, 384, 64),
    (1, 64, 4, 128, 32),
    (1, 64, 4, 256, 32),
    (1, 128, 4, 128, 32),
]


def _make_benchmark_cases() -> list[TestCase]:
    cases: list[TestCase] = []
    for b, n_seq, head, n_res, dim in _EVOATTN_BENCHMARK_SHAPES:
        cases.append(TestCase(b, n_seq, head, n_res, dim, "float16", 2e-2, 2e-2))
        cases.append(TestCase(b, n_seq, head, n_res, dim, "bfloat16", 4e-2, 4e-2))
    return cases


BENCHMARK_CASES = _make_benchmark_cases()


def _dtype(torch_module, dtype_name: str):
    if dtype_name == "float16":
        return torch_module.float16
    if dtype_name == "bfloat16":
        return torch_module.bfloat16
    if dtype_name == "float32":
        return torch_module.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def seed_for_case(case: TestCase) -> int:
    return (((case.b * 131 + case.n_seq) * 131 + case.head) * 131 + case.n_res) * 131 + case.dim


def case_metadata(case: TestCase):
    return {
        "shape": [case.b, case.n_seq, case.head, case.n_res, case.dim],
        "dims": {"B": case.b, "N_seq": case.n_seq, "Head": case.head, "N_res": case.n_res, "Dim": case.dim},
        "dtype": case.dtype_name,
    }


def make_inputs(torch_module, case: TestCase):
    dtype = _dtype(torch_module, case.dtype_name)
    b, s, h, r, d = case.b, case.n_seq, case.head, case.n_res, case.dim

    # std=0.5 keeps fp16/bf16 magnitudes in a sane range (matches MegaFold tests).
    q = (torch_module.randn((b, s, r, h, d), device="cuda", dtype=dtype) * 0.5)
    k = (torch_module.randn((b, s, r, h, d), device="cuda", dtype=dtype) * 0.5)
    v = (torch_module.randn((b, s, r, h, d), device="cuda", dtype=dtype) * 0.5)
    do = (torch_module.randn((b, s, r, h, d), device="cuda", dtype=dtype) * 0.5)

    # Additive per-key residue mask: 0 where kept, large-negative where dropped.
    # Force key 0 kept so no query row is fully masked (which would NaN softmax).
    keep = torch_module.rand((b, s, 1, 1, r), device="cuda") > 0.5
    keep[..., 0] = True
    res_mask = torch_module.where(
        keep,
        torch_module.zeros((), device="cuda", dtype=torch_module.float32),
        torch_module.full((), -1e9, device="cuda", dtype=torch_module.float32),
    )

    # pair_bias is trainable and kept in float32 (as in MegaFold).
    pair_bias = (torch_module.randn((b, 1, h, r, r), device="cuda", dtype=torch_module.float32) * 0.5)
    return do, q, k, v, res_mask, pair_bias


def torch_oracle(torch_module, do, q, k, v, res_mask, pair_bias):
    dtype = q.dtype
    dim = q.shape[-1]
    scale = dim ** -0.5

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    pair_bias_ref = pair_bias.detach().clone().requires_grad_(True)

    q_t = q_ref.transpose(-2, -3)
    k_t = k_ref.transpose(-2, -3)
    v_t = v_ref.transpose(-2, -3)
    scores = torch_module.matmul(q_t * scale, k_t.transpose(-1, -2)) + pair_bias_ref + res_mask
    probs = torch_module.softmax(scores.float(), dim=-1).to(dtype)
    out_t = torch_module.matmul(probs, v_t)
    out = out_t.transpose(-2, -3).contiguous()
    out.backward(do.detach().clone())
    return q_ref.grad, k_ref.grad, v_ref.grad, pair_bias_ref.grad


def atol(case: TestCase, output_name: str) -> float:
    # d_pair_bias accumulates over the N_seq axis, so allow a bit more slack.
    if output_name == "d_pair_bias":
        return case.atol_value * 2.0
    return case.atol_value


def rtol(case: TestCase, output_name: str) -> float:
    return case.rtol_value


def correctness_hint() -> str:
    return (
        "S=scale*Q@K^T+pair_bias+res_mask (over residue axis), P=softmax(S), "
        "dV=P^T@dO, dP=dO@V^T, dS=P*(dP-rowsum(dP*P)), dQ=scale*dS@K, "
        "dK=scale*dS^T@Q, d_pair_bias=sum_over_Nseq(dS)"
    )
