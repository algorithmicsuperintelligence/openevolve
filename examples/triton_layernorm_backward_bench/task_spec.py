"""Task spec for the LayerNorm Triton backward benchmark."""

from dataclasses import dataclass


CANDIDATE_FN_NAME = "layernorm_backward_triton"
OUTPUT_NAMES = ("dx", "dweight", "dbias")
EPS = 1e-5


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

BENCHMARK_CASES = [
    TestCase(128, 256, "float16", 5e-2, 5e-2),
    TestCase(256, 512, "float16", 5e-2, 5e-2),
    TestCase(512, 1024, "float16", 5e-2, 5e-2),
]


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
    bias = torch_module.randn((case.cols,), device="cuda", dtype=dtype)
    dy = torch_module.randn((case.rows, case.cols), device="cuda", dtype=dtype)
    return dy, x, weight, bias, EPS


def torch_oracle(torch_module, dy, x, weight, bias, eps):
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)
    mean = torch_module.mean(x_ref.float(), dim=-1, keepdim=True)
    var = torch_module.mean((x_ref.float() - mean) ** 2, dim=-1, keepdim=True)
    xhat = (x_ref.float() - mean) * torch_module.rsqrt(var + eps)
    y_ref = (xhat * weight_ref.float() + bias_ref.float()).to(x.dtype)
    y_ref.backward(dy.detach().clone())
    return x_ref.grad, weight_ref.grad, bias_ref.grad


def atol(case: TestCase, output_name: str) -> float:
    return case.atol_value


def rtol(case: TestCase, output_name: str) -> float:
    return case.rtol_value


def correctness_hint() -> str:
    return (
        "xhat=(x-mean)/sqrt(var+eps), one=dy*weight, "
        "dx=(one-mean(one)-xhat*mean(xhat*one))*rstd, "
        "dweight=sum(dy*xhat, dim=0), dbias=sum(dy, dim=0)"
    )
