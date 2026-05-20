"""
Thin wrapper: calls _lib.extract_best.main with cpsat-bench phase map.
"""
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_SHARED = _HERE / "shared"
_INPUT_DIR = _HERE.parents[1]

if str(_INPUT_DIR) not in sys.path:
    sys.path.insert(0, str(_INPUT_DIR))

from _lib.extract_best import main  # noqa: E402

PHASE_DIRS = {
    1: "phase1_search",
    2: "phase2_presolve",
    3: "phase3_lp_cuts",
}

if __name__ == "__main__":
    main(_HERE, _SHARED, PHASE_DIRS)
