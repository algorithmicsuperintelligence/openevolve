"""
Thin wrapper: calls _lib.prepare_phase.main with cpsat-bench phase config.
Materializes phase4_unified/initial_program.py from phase{1,2,3}_best.json
plus phase{1,2,3}_buckets.json / phase{1,2,3}_stage3.json when present.
"""
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_SHARED = _HERE / "shared"
_INPUT_DIR = _HERE.parents[1]
_UNIFIED_FILE = _HERE / "phase4_unified" / "initial_program.py"

if str(_INPUT_DIR) not in sys.path:
    sys.path.insert(0, str(_INPUT_DIR))

from _lib.prepare_phase import main  # noqa: E402

PRIOR_PHASES = [1, 2, 3]

if __name__ == "__main__":
    # cpsat-bench unified phase exposes GLOBAL_OVERRIDES (not the legacy
    # UNIFIED_OVERRIDES) so prepare_phase writes that name. SIZE_BUCKETS and
    # STAGE3_OVERRIDES are appended when prior phases produced bucket/stage3
    # extracts.
    main(_HERE, _SHARED, PRIOR_PHASES, _UNIFIED_FILE,
         dict_name="GLOBAL_OVERRIDES")
