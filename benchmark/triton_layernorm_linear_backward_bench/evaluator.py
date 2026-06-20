import os
import sys

BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(BENCHMARK_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if BENCHMARK_DIR not in sys.path:
    sys.path.insert(0, BENCHMARK_DIR)

from benchmark.triton_backward_bench_common.evaluator_core import (  # noqa: E402
    evaluate_program,
    evaluate_stage1_program,
    main as core_main,
)
try:
    from benchmark.triton_layernorm_linear_backward_bench import task_spec  # noqa: E402
except ImportError:  # pragma: no cover - supports direct script execution
    import task_spec  # type: ignore  # noqa: E402


def evaluate(program_path: str):
    return evaluate_program(program_path, task_spec)


def evaluate_stage1(program_path: str):
    return evaluate_stage1_program(program_path, task_spec)


if __name__ == "__main__":
    raise SystemExit(core_main(sys.argv, task_spec))
