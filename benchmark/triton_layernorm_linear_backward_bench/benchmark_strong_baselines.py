"""Strong-baseline comparison for an evolved LayerNorm -> Linear backward candidate.

Compares a candidate ``layernorm_linear_backward_triton`` against:
  - PyTorch autograd backward over the reference forward (the OpenEvolve baseline),
  - the naive unfused backward (legacy seed reference),
  - MegaFold's fused Triton LayernormLinear backward (the expert reference,
    arXiv:2506.20686), if it (and its deps, e.g. liger-kernel) can be imported.

Run on a CUDA-visible GPU node:

    python benchmark_strong_baselines.py [path/to/candidate.py]
"""

import importlib.util
import os
import statistics
import sys

import torch
import torch.nn.functional as F

from backward_naive import layernorm_linear_backward_naive
from task_spec import BENCHMARK_CASES, CANDIDATE_FN_NAME, TestCase, make_inputs, seed_for_case

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
MEGAFOLD_ROOT = os.path.join(REPO_ROOT, "MegaFold")


def _load_megafold_layernorm_linear():
    """Import MegaFold's LayernormLinear nn.Module (needs liger-kernel + megafold)."""
    if not os.path.isdir(MEGAFOLD_ROOT):
        return None
    if MEGAFOLD_ROOT not in sys.path:
        sys.path.insert(0, MEGAFOLD_ROOT)
    try:
        from megafold.model.FusedLayernormLinear.fused_layernorm_linear import LayernormLinear
        return LayernormLinear
    except Exception as exc:  # pragma: no cover - optional heavy dependency
        print(f"NOTE: MegaFold FusedLayernormLinear unavailable ({exc}); skipping that baseline.")
        return None


def _load_candidate(path: str):
    spec = importlib.util.spec_from_file_location("ln_linear_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, CANDIDATE_FN_NAME)


def _median_ms(fn, warmup: int = 10, reps: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings = []
    for _ in range(reps):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(float(start.elapsed_time(end)))
    return float(statistics.median(timings))


def _autograd_backward_only(dout, x, weight, bias, linear_weight, eps):
    k = x.shape[-1]
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = weight.detach().clone().requires_grad_(True)
    b_ref = bias.detach().clone().requires_grad_(True)
    lw_ref = linear_weight.detach().clone().requires_grad_(True)
    out = F.layer_norm(x_ref, (k,), w_ref, b_ref, eps) @ lw_ref
    return lambda: out.backward(dout, retain_graph=True)


def _megafold_backward_only(LayernormLinear, dout, x, weight, bias, linear_weight, eps):
    m, k = x.shape
    n = linear_weight.shape[1]
    mod = LayernormLinear(k, n, has_layernorm_bias=True, has_linear_bias=False,
                          device=x.device, dtype=x.dtype)
    with torch.no_grad():
        mod.WEIGHT.copy_(weight)
        mod.BIAS.copy_(bias)
        mod.linear_weight.copy_(linear_weight)
    x_ref = x.detach().clone().requires_grad_(True)
    out = mod(x_ref)
    return lambda: out.backward(dout, retain_graph=True)


def run_case(case: TestCase, candidate_fn, LayernormLinear) -> dict:
    torch.manual_seed(seed_for_case(case))
    dout, x, weight, bias, linear_weight, eps = make_inputs(torch, case)

    autograd_ms = _median_ms(_autograd_backward_only(dout, x, weight, bias, linear_weight, eps))
    naive_ms = _median_ms(
        lambda: layernorm_linear_backward_naive(dout, x, weight, bias, linear_weight, eps)
    )
    candidate_ms = _median_ms(
        lambda: candidate_fn(dout, x, weight, bias, linear_weight, eps)
    )

    report = {
        "shape": [case.m, case.k, case.n],
        "dtype": case.dtype_name,
        "pytorch_autograd_backward_ms": autograd_ms,
        "naive_backward_ms": naive_ms,
        "candidate_backward_ms": candidate_ms,
        "candidate_speedup_vs_pytorch_autograd": autograd_ms / max(candidate_ms, 1e-9),
    }
    if LayernormLinear is not None:
        try:
            megafold_ms = _median_ms(
                _megafold_backward_only(LayernormLinear, dout, x, weight, bias, linear_weight, eps)
            )
            report["megafold_fused_backward_ms"] = megafold_ms
            report["candidate_speedup_vs_megafold"] = megafold_ms / max(candidate_ms, 1e-9)
        except Exception as exc:  # pragma: no cover - MegaFold kernel is config-sensitive
            report["megafold_fused_backward_ms"] = f"error: {exc}"
    return report


def main(argv) -> int:
    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available; run this benchmark on a GPU node.")
        return 0

    candidate_path = argv[1] if len(argv) > 1 else os.path.join(HERE, "initial_program.py")
    candidate_fn = _load_candidate(candidate_path)
    LayernormLinear = _load_megafold_layernorm_linear()

    print(f"Candidate: {candidate_path}")
    print(f"MegaFold expert baseline: {'available' if LayernormLinear else 'unavailable'}")
    for case in BENCHMARK_CASES:
        print(run_case(case, candidate_fn, LayernormLinear))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
