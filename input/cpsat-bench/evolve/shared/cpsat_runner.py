"""
Thin wrapper: run_cpsat = subprocess_runner.run_solver bound to
_cpsat_solve_worker.py.

Status strings produced by the worker:
  OPTIMAL, FEASIBLE, INFEASIBLE, UNKNOWN, MODEL_INVALID
For cost mode, success dict also includes "objective" = solver.ObjectiveValue().
"""
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_INPUT_DIR = _HERE.parents[2]
_WORKER = _HERE / "_cpsat_solve_worker.py"

if str(_INPUT_DIR) not in sys.path:
    sys.path.insert(0, str(_INPUT_DIR))

from _lib.subprocess_runner import run_solver  # noqa: E402


def run_cpsat(problem_path, params, timeout_s, python_bin=None, cpu_core=None):
    return run_solver(_WORKER, problem_path, params, timeout_s,
                      python_bin=python_bin, cpu_core=cpu_core)
