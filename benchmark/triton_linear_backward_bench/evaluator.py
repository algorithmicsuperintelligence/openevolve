import os
import sys

BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(BENCHMARK_DIR)
# Force REPO_ROOT to the front of sys.path. OpenEvolve prepends this bench's own
# directory, whose benchmark.py would otherwise shadow the top-level `benchmark`
# package (-> "'benchmark' is not a package"). Re-inserting guarantees the
# package resolves before the local module.
for _p in (BENCHMARK_DIR, REPO_ROOT):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from benchmark.triton_backward_bench_common.evaluator_core import (  # noqa: E402
    evaluate_program,
    evaluate_stage1_program,
    main as core_main,
)
try:
    from benchmark.triton_linear_backward_bench import task_spec  # noqa: E402
except ImportError:  # pragma: no cover - supports direct script execution
    import task_spec  # type: ignore  # noqa: E402


def evaluate(program_path: str):
    return evaluate_program(program_path, task_spec)


def evaluate_stage1(program_path: str):
    return evaluate_stage1_program(program_path, task_spec)


if __name__ == "__main__":
    raise SystemExit(core_main(sys.argv, task_spec))
