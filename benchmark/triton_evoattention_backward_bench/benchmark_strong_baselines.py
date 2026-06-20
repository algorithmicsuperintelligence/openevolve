"""Strong-baseline comparison for an evolved EvoformerAttention backward candidate.

Compares a candidate ``evoattention_backward_triton`` against:
  - PyTorch autograd backward over the reference forward (the OpenEvolve baseline),
  - the naive materialized backward (legacy seed reference),
  - MegaFold's fused Triton EvoformerAttention backward (the expert reference,
    arXiv:2506.20686), if it can be imported.

Run on a CUDA-visible GPU node:

    python benchmark_strong_baselines.py [path/to/candidate.py]
"""

import importlib.util
import os
import statistics
import sys

import torch

from backward_naive import evoattention_backward_naive
from forward_ref import evoattention_forward_ref
from task_spec import BENCHMARK_CASES, CANDIDATE_FN_NAME, TestCase, make_inputs, seed_for_case

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
MEGAFOLD_EVOATTN = os.path.join(
    REPO_ROOT, "MegaFold", "megafold", "model", "FusedEvoAttention", "evoattention.py"
)


def _load_megafold_evoformer():
    """Import MegaFold's TritonEvoformer by file path (avoids package side effects)."""
    if not os.path.exists(MEGAFOLD_EVOATTN):
        return None
    try:
        spec = importlib.util.spec_from_file_location("megafold_evoattention", MEGAFOLD_EVOATTN)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.TritonEvoformer
    except Exception as exc:  # pragma: no cover - optional dependency / GPU-only
        print(f"NOTE: MegaFold FusedEvoAttention unavailable ({exc}); skipping that baseline.")
        return None


def _load_candidate(path: str):
    spec = importlib.util.spec_from_file_location("evoattn_candidate", path)
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


def _autograd_backward_only(do, q, k, v, res_mask, pair_bias):
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    pb_ref = pair_bias.detach().clone().requires_grad_(True)
    out = evoattention_forward_ref(q_ref, k_ref, v_ref, res_mask, pb_ref)
    return lambda: out.backward(do, retain_graph=True)


def _megafold_backward_only(triton_evoformer, do, q, k, v, res_mask, pair_bias):
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    pb_ref = pair_bias.detach().clone().requires_grad_(True)
    out = triton_evoformer(q_ref, k_ref, v_ref, res_mask, pb_ref)
    return lambda: out.backward(do, retain_graph=True)


def run_case(case: TestCase, candidate_fn, triton_evoformer) -> dict:
    torch.manual_seed(seed_for_case(case))
    do, q, k, v, res_mask, pair_bias = make_inputs(torch, case)

    autograd_ms = _median_ms(_autograd_backward_only(do, q, k, v, res_mask, pair_bias))
    naive_ms = _median_ms(lambda: evoattention_backward_naive(do, q, k, v, res_mask, pair_bias))
    candidate_ms = _median_ms(lambda: candidate_fn(do, q, k, v, res_mask, pair_bias))

    report = {
        "shape": [case.b, case.n_seq, case.head, case.n_res, case.dim],
        "dtype": case.dtype_name,
        "pytorch_autograd_backward_ms": autograd_ms,
        "naive_backward_ms": naive_ms,
        "candidate_backward_ms": candidate_ms,
        "candidate_speedup_vs_pytorch_autograd": autograd_ms / max(candidate_ms, 1e-9),
    }
    if triton_evoformer is not None:
        try:
            megafold_ms = _median_ms(
                _megafold_backward_only(triton_evoformer, do, q, k, v, res_mask, pair_bias)
            )
            report["megafold_fused_backward_ms"] = megafold_ms
            report["candidate_speedup_vs_megafold"] = megafold_ms / max(candidate_ms, 1e-9)
        except Exception as exc:  # pragma: no cover - MegaFold kernel is shape-sensitive
            report["megafold_fused_backward_ms"] = f"error: {exc}"
    return report


def main(argv) -> int:
    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available; run this benchmark on a GPU node.")
        return 0

    candidate_path = argv[1] if len(argv) > 1 else os.path.join(HERE, "initial_program.py")
    candidate_fn = _load_candidate(candidate_path)
    triton_evoformer = _load_megafold_evoformer()

    print(f"Candidate: {candidate_path}")
    print(f"MegaFold expert baseline: {'available' if triton_evoformer else 'unavailable'}")
    for case in BENCHMARK_CASES:
        print(run_case(case, candidate_fn, triton_evoformer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
