"""Task spec for the matmul (plain GEMM) Triton backward benchmark."""

from dataclasses import dataclass


CANDIDATE_FN_NAME = "matmul_backward_triton"
OUTPUT_NAMES = ("da", "db")
# OpenEvolve speedup uses torch_oracle (PyTorch autograd over a @ b, i.e. cuBLAS).
PERFORMANCE_BASELINE = "pytorch_autograd"


@dataclass(frozen=True)
class TestCase:
    m: int
    n: int
    k: int
    dtype_name: str
    atol_value: float
    rtol_value: float


# Correctness: small shapes (incl. non-tile-aligned), both dtypes. Tolerances are
# matmul-appropriate (TF32 tensor-core matmul on the fp32 path rounds differently
# from cuBLAS, so near-zero output elements need a looser atol).
CORRECTNESS_CASES = [
    TestCase(64, 64, 64, "float32", 8e-2, 2e-2),
    TestCase(129, 257, 127, "float32", 8e-2, 2e-2),
    TestCase(128, 256, 128, "float16", 1e-1, 2e-2),
    TestCase(512, 512, 256, "float16", 1e-1, 2e-2),
]

# Benchmark: larger compute-bound shapes (M, N, K), both dtypes.
_MATMUL_BENCHMARK_SHAPES = [
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 1024, 1024),
    (1024, 2048, 1024),
    (2048, 2048, 1024),
    (4096, 1024, 1024),
]


def _make_benchmark_cases() -> list[TestCase]:
    cases: list[TestCase] = []
    for m, n, k in _MATMUL_BENCHMARK_SHAPES:
        cases.append(TestCase(m, n, k, "float32", 8e-2, 2e-2))
        cases.append(TestCase(m, n, k, "float16", 1e-1, 2e-2))
    return cases


BENCHMARK_CASES = _make_benchmark_cases()


def _dtype(torch_module, dtype_name: str):
    if dtype_name == "float32":
        return torch_module.float32
    if dtype_name == "float16":
        return torch_module.float16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def seed_for_case(case: TestCase) -> int:
    return (case.m * 100003 + case.n) * 100003 + case.k


def case_metadata(case: TestCase):
    return {
        "shape": [case.m, case.n, case.k],
        "dtype": case.dtype_name,
    }


def make_inputs(torch_module, case: TestCase):
    dtype = _dtype(torch_module, case.dtype_name)
    a = torch_module.randn((case.m, case.k), device="cuda", dtype=dtype)
    b = torch_module.randn((case.k, case.n), device="cuda", dtype=dtype)
    dc = torch_module.randn((case.m, case.n), device="cuda", dtype=dtype)
    return dc, a, b


def torch_oracle(torch_module, dc, a, b):
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    c_ref = a_ref @ b_ref
    c_ref.backward(dc.detach().clone())
    return a_ref.grad, b_ref.grad


def atol(case: TestCase, output_name: str) -> float:
    return case.atol_value


def rtol(case: TestCase, output_name: str) -> float:
    return case.rtol_value


def correctness_hint() -> str:
    return "da = dc @ b.T; db = a.T @ dc"
