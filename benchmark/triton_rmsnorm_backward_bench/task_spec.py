"""Task spec for the RMSNorm Triton backward benchmark."""

from dataclasses import dataclass


CANDIDATE_FN_NAME = "rmsnorm_backward_triton"
OUTPUT_NAMES = ("dx", "dweight")
EPS = 1e-5
# OpenEvolve speedup uses torch_oracle (PyTorch autograd). Override with BASELINE_PROGRAM_PATH
# only for legacy ablations against a fixed Triton seed program.
PERFORMANCE_BASELINE = "pytorch_autograd"


@dataclass(frozen=True)
class TestCase:
    rows: int
    cols: int
    dtype_name: str
    atol_value: float
    rtol_value: float


CORRECTNESS_CASES = [
    TestCase(8, 64, "float32", 2e-5, 2e-5),
    TestCase(17, 128, "float32", 2e-5, 2e-5),
    TestCase(32, 256, "float16", 5e-2, 5e-2),
    TestCase(64, 512, "float16", 5e-2, 5e-2),
]

# Aligned with the LayerNorm benchmark shape presets (dynamic + nontile).
_RMSNORM_BENCHMARK_SHAPES = [
    # dynamic
    (1, 768),
    (8, 1024),
    (32, 1536),
    (8, 4096),
    (1, 8192),
    # nontile
    (17, 127),
    (17, 513),
    (17, 1000),
]


def _make_benchmark_cases() -> list[TestCase]:
    cases: list[TestCase] = []
    for rows, cols in _RMSNORM_BENCHMARK_SHAPES:
        cases.append(TestCase(rows, cols, "float32", 2e-5, 2e-5))
        cases.append(TestCase(rows, cols, "float16", 5e-2, 5e-2))
    return cases


BENCHMARK_CASES = _make_benchmark_cases()


def _dtype(torch_module, dtype_name: str):
    if dtype_name == "float32":
        return torch_module.float32
    if dtype_name == "float16":
        return torch_module.float16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def seed_for_case(case: TestCase) -> int:
    return case.rows * 100000 + case.cols


def case_metadata(case: TestCase):
    return {
        "shape": [case.rows, case.cols],
        "dtype": case.dtype_name,
        "eps": EPS,
    }


def make_inputs(torch_module, case: TestCase):
    dtype = _dtype(torch_module, case.dtype_name)
    x = torch_module.randn((case.rows, case.cols), device="cuda", dtype=dtype)
    weight = torch_module.randn((case.cols,), device="cuda", dtype=dtype)
    dy = torch_module.randn((case.rows, case.cols), device="cuda", dtype=dtype)
    return dy, x, weight, EPS


def torch_oracle(torch_module, dy, x, weight, eps):
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    ms = torch_module.mean(x_ref.float() * x_ref.float(), dim=-1, keepdim=True)
    xhat = x_ref.float() * torch_module.rsqrt(ms + eps)
    y_ref = (xhat * weight_ref.float()).to(x.dtype)
    y_ref.backward(dy.detach().clone())
    return x_ref.grad, weight_ref.grad


def atol(case: TestCase, output_name: str) -> float:
    return case.atol_value


def rtol(case: TestCase, output_name: str) -> float:
    return case.rtol_value


def correctness_hint() -> str:
    return (
        "rrms=rsqrt(mean(x^2)+eps), xhat=x*rrms, g=dy*weight, "
        "dx=(g - xhat*mean(g*xhat))*rrms, dweight=sum(dy*xhat, dim=0)"
    )
