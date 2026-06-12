"""Task spec for the Linear (matmul) Triton backward benchmark."""

from dataclasses import dataclass


CANDIDATE_FN_NAME = "linear_backward_triton"
OUTPUT_NAMES = ("dx", "dweight", "dbias")
# OpenEvolve speedup uses torch_oracle (PyTorch autograd over F.linear, i.e. cuBLAS).
# Override with BASELINE_PROGRAM_PATH only for legacy ablations against a fixed seed.
PERFORMANCE_BASELINE = "pytorch_autograd"


@dataclass(frozen=True)
class TestCase:
    m: int
    n: int
    k: int
    dtype_name: str
    atol_value: float
    rtol_value: float


# Correctness: small shapes (incl. non-tile-aligned), both dtypes.
# Tolerances are matmul-appropriate: the fp32 path runs TF32 tensor-core matmul
# on both sides (Triton tl.dot and cuBLAS autograd), which round differently, so
# near-zero output elements of dweight (= dy.T @ x, the larger-magnitude grad)
# need an atol that absorbs TF32 spread rather than the tight 1e-2 used by the
# memory-bound elementwise/norm benches.
CORRECTNESS_CASES = [
    TestCase(64, 64, 64, "float32", 8e-2, 2e-2),
    TestCase(129, 257, 127, "float32", 8e-2, 2e-2),
    TestCase(128, 256, 128, "float16", 1e-1, 2e-2),
    TestCase(512, 512, 256, "float16", 1e-1, 2e-2),
]

# Benchmark: larger compute-bound shapes (M, N, K), both dtypes.
_LINEAR_BENCHMARK_SHAPES = [
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 1024, 1024),
    (1024, 2048, 1024),
    (2048, 2048, 1024),
    (4096, 1024, 1024),
]


def _make_benchmark_cases() -> list[TestCase]:
    cases: list[TestCase] = []
    for m, n, k in _LINEAR_BENCHMARK_SHAPES:
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
    # weight ~ 1/sqrt(K) keeps fp16 magnitudes in a sane range (standard Linear init scale).
    scale = case.k ** -0.5
    x = torch_module.randn((case.m, case.k), device="cuda", dtype=dtype)
    weight = (torch_module.randn((case.n, case.k), device="cuda", dtype=dtype) * scale).to(dtype)
    dy = torch_module.randn((case.m, case.n), device="cuda", dtype=dtype)
    return dy, x, weight


def torch_oracle(torch_module, dy, x, weight):
    import torch.nn.functional as F

    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    bias_ref = torch_module.zeros(weight.shape[0], device=x.device, dtype=x.dtype, requires_grad=True)
    y_ref = F.linear(x_ref, weight_ref, bias_ref)
    y_ref.backward(dy.detach().clone())
    return x_ref.grad, weight_ref.grad, bias_ref.grad


def atol(case: TestCase, output_name: str) -> float:
    return case.atol_value


def rtol(case: TestCase, output_name: str) -> float:
    return case.rtol_value


def correctness_hint() -> str:
    return "dx = dy @ weight; dweight = dy.T @ x; dbias = sum(dy, dim=0)"
