"""
Thin wrapper around input/_lib/runtime.py.
Binds config.yaml path; re-exports parallel_solvers / cascade_threshold / core_range.
"""
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_CONFIG_YAML = _HERE.parent / "config.yaml"
_INPUT_DIR = _HERE.parents[2]    # input/

if str(_INPUT_DIR) not in sys.path:
    sys.path.insert(0, str(_INPUT_DIR))

from _lib import runtime as _rt  # noqa: E402

core_range = _rt.core_range


def parallel_solvers(default=1):
    return _rt.parallel_solvers(_CONFIG_YAML, default=default)


def cascade_threshold(index, default):
    return _rt.cascade_threshold(_CONFIG_YAML, index, default)
