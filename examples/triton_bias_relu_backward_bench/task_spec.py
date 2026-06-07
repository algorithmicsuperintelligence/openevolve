"""Task spec for the bias + ReLU Triton backward benchmark."""

from dataclasses import dataclass


CANDIDATE_FN_NAME = "bias_relu_backward_triton"
OUTPUT_NAMES = ("dx", "dbias")
PERFORMANCE_BASELINE = "pytorch_autograd"


@dataclass(frozen=True)
class TestCase:
    rows: int
    cols: int
    dtype_name: str
    atol_value: float
    rtol_value: float


CORRECTNESS_CASES = [
    TestCase(17, 31, "float32", 1e-5, 1e-5),
    TestCase(128, 256, "float32", 1e-5, 1e-5),
    TestCase(129, 257, "float16", 2e-2, 2e-2),
    TestCase(512, 1024, "float16", 2e-2, 2e-2),
]

BENCHMARK_CASES = [
    TestCase(512, 1024, "float16", 2e-2, 2e-2),
    TestCase(1024, 1024, "float16", 2e-2, 2e-2),
    TestCase(1024, 2048, "float16", 2e-2, 2e-2),
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
    }


def make_inputs(torch_module, case: TestCase):
    dtype = _dtype(torch_module, case.dtype_name)
    x = torch_module.randn((case.rows, case.cols), device="cuda", dtype=dtype)
    bias = torch_module.randn((case.cols,), device="cuda", dtype=dtype)
    dy = torch_module.randn((case.rows, case.cols), device="cuda", dtype=dtype)
    return dy, x, bias


def torch_oracle(torch_module, dy, x, bias):
    x_ref = x.detach().clone().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)
    y_ref = torch_module.relu(x_ref + bias_ref)
    y_ref.backward(dy.detach().clone())
    return x_ref.grad, bias_ref.grad


def atol(case: TestCase, output_name: str) -> float:
    return case.atol_value


def rtol(case: TestCase, output_name: str) -> float:
    return case.rtol_value


def correctness_hint() -> str:
    return "dx = dy * ((x + bias) > 0), dbias = sum(dx, dim=0)"
