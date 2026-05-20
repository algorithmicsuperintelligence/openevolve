"""
Thin wrapper: run_z3 = subprocess_runner.run_solver bound to _z3_solve_worker.py.

Worker uses the z3 Python binding (z3.set_param + Optimize/Solver). Python
binding is preferred over CLI because z3 4.13.x CLI rejects globals like
`threads`, `parallel.enable`, `sls.parallel` as positional `key=value`,
which the binding accepts. Matches the original benchmark's
`applied_params_hash 543b29...` setup.

Status strings: "Sat" / "Unsat" / "Unknown".
"""
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_INPUT_DIR = _HERE.parents[2]
_WORKER = _HERE / "_z3_solve_worker.py"

if str(_INPUT_DIR) not in sys.path:
    sys.path.insert(0, str(_INPUT_DIR))

from _lib.subprocess_runner import run_solver  # noqa: E402


def run_z3(smt2_path, params, timeout_s, python_bin=None, cpu_core=None):
    return run_solver(_WORKER, smt2_path, params, timeout_s,
                      python_bin=python_bin, cpu_core=cpu_core)
