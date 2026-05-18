import os
import sys

EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(EXAMPLES_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)

from triton_backward_bench_common.evaluator_core import (  # noqa: E402
    evaluate_program,
    evaluate_stage1_program,
    main as core_main,
)
import task_spec  # noqa: E402


def evaluate(program_path: str):
    return evaluate_program(program_path, task_spec)


def evaluate_stage1(program_path: str):
    return evaluate_stage1_program(program_path, task_spec)


if __name__ == "__main__":
    raise SystemExit(core_main(sys.argv, task_spec))
