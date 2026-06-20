"""Task spec for the fused LayerNorm -> Linear Triton backward benchmark."""

from dataclasses import dataclass


CANDIDATE_FN_NAME = "layernorm_linear_backward_triton"
OUTPUT_NAMES = ("dx", "dlinear_weight", "dweight", "dbias")
# OpenEvolve speedup uses torch_oracle (PyTorch autograd over the reference
# LayerNorm -> Linear forward). Override with BASELINE_PROGRAM_PATH only for
# legacy ablations against a fixed seed program.
PERFORMANCE_BASELINE = "pytorch_autograd"
EPS = 1e-5


@dataclass(frozen=True)
class TestCase:
    m: int
    k: int
    n: int
    dtype_name: str
    atol_value: float
    rtol_value: float


# Correctness: small shapes (incl. non-tile-aligned), both dtypes. Tolerances are
# matmul-appropriate (the fp32 path runs TF32 tensor-core matmul on both sides),
# and the layernorm weight grads reduce over M, so they get extra slack.
CORRECTNESS_CASES = [
    TestCase(64, 64, 128, "float32", 8e-2, 2e-2),
    TestCase(129, 127, 257, "float32", 8e-2, 2e-2),
    TestCase(128, 128, 256, "float16", 1e-1, 2e-2),
    TestCase(512, 256, 512, "float16", 1e-1, 2e-2),
]

# Benchmark: AF3 Transition-like shapes. K is the model dim, N = expansion * 2 * K
# (the SwiGLU-gated up-projection), M is the flattened token count (large).
_LN_LINEAR_BENCHMARK_SHAPES = [
    # (M, K, N)
    (4096, 128, 512),
    (8192, 128, 1024),
    (16384, 128, 256),
    (4096, 256, 1024),
    (2048, 512, 2048),
    (384, 768, 1536),
]


def _make_benchmark_cases() -> list[TestCase]:
    cases: list[TestCase] = []
    for m, k, n in _LN_LINEAR_BENCHMARK_SHAPES:
        cases.append(TestCase(m, k, n, "float32", 8e-2, 2e-2))
        cases.append(TestCase(m, k, n, "float16", 1e-1, 2e-2))
    return cases


BENCHMARK_CASES = _make_benchmark_cases()


def _dtype(torch_module, dtype_name: str):
    if dtype_name == "float32":
        return torch_module.float32
    if dtype_name == "float16":
        return torch_module.float16
    if dtype_name == "bfloat16":
        return torch_module.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def seed_for_case(case: TestCase) -> int:
    return (case.m * 100003 + case.k) * 100003 + case.n


def case_metadata(case: TestCase):
    return {
        "shape": [case.m, case.k, case.n],
        "dims": {"M": case.m, "K": case.k, "N": case.n},
        "dtype": case.dtype_name,
        "eps": EPS,
    }


def make_inputs(torch_module, case: TestCase):
    dtype = _dtype(torch_module, case.dtype_name)
    m, k, n = case.m, case.k, case.n

    x = torch_module.randn((m, k), device="cuda", dtype=dtype)
    # LayerNorm affine params near their usual init (gamma~1, beta~0) but perturbed
    # so dweight/dbias are exercised.
    weight = (1.0 + 0.5 * torch_module.randn((k,), device="cuda", dtype=dtype)).to(dtype)
    bias = (0.5 * torch_module.randn((k,), device="cuda", dtype=dtype)).to(dtype)
    # Linear weight [K, N] with standard 1/sqrt(K) init scale to keep fp16 sane.
    scale = k ** -0.5
    linear_weight = (torch_module.randn((k, n), device="cuda", dtype=dtype) * scale).to(dtype)
    dout = torch_module.randn((m, n), device="cuda", dtype=dtype)
    return dout, x, weight, bias, linear_weight, EPS


def torch_oracle(torch_module, dout, x, weight, bias, linear_weight, eps):
    import torch.nn.functional as F

    k = x.shape[-1]
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)
    lw_ref = linear_weight.detach().clone().requires_grad_(True)
    y_hat = F.layer_norm(x_ref, (k,), weight_ref, bias_ref, eps)
    out = y_hat @ lw_ref
    out.backward(dout.detach().clone())
    return x_ref.grad, lw_ref.grad, weight_ref.grad, bias_ref.grad


def atol(case: TestCase, output_name: str) -> float:
    # dweight/dbias reduce over M; give them extra slack on the fp16 path.
    if output_name in ("dweight", "dbias"):
        return case.atol_value * 2.0
    return case.atol_value


def rtol(case: TestCase, output_name: str) -> float:
    return case.rtol_value


def correctness_hint() -> str:
    return (
        "x_hat=(x-mean)*rstd, y_hat=x_hat*gamma+beta, out=y_hat@B; "
        "dy_hat=dout@B^T, dB=y_hat^T@dout, dgamma=sum(dy_hat*x_hat,0), "
        "dbeta=sum(dy_hat,0), wdy=dy_hat*gamma, dx=(wdy-x_hat*mean(x_hat*wdy)-mean(wdy))*rstd"
    )
