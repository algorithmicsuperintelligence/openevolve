"""
After phase N completes, extract get_phase_overrides() from its best_program.py
and write shared/phaseN_best.json. Phase N+1 auto-loads it.

Usage: python extract_best.py <phase_num: 1|2|3>
"""
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
SHARED = ROOT / "shared"

PHASE_DIRS = {
    1: "phase1_opt_sls",
    2: "phase2_sat",
    3: "phase3_smt",
}


def main():
    if len(sys.argv) != 2:
        print("usage: python extract_best.py <1|2|3>", file=sys.stderr)
        sys.exit(2)

    n = int(sys.argv[1])
    if n not in PHASE_DIRS:
        print(f"phase must be 1, 2, or 3 (got {n})", file=sys.stderr)
        sys.exit(2)

    phase_dir = ROOT / PHASE_DIRS[n]
    best_py = phase_dir / "openevolve_output" / "best" / "best_program.py"
    if not best_py.exists():
        print(f"best_program.py not found: {best_py}", file=sys.stderr)
        print("run phase first (./run_phase.sh N) before extracting.", file=sys.stderr)
        sys.exit(1)

    # Make shared/ importable for the module under test
    sys.path.insert(0, str(SHARED))

    spec = importlib.util.spec_from_file_location(f"phase{n}_best", best_py)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"failed to load {best_py}: {e}", file=sys.stderr)
        sys.exit(1)

    if not hasattr(module, "get_phase_overrides"):
        print(f"{best_py} missing get_phase_overrides()", file=sys.stderr)
        sys.exit(1)

    overrides = module.get_phase_overrides()
    if not isinstance(overrides, dict):
        print(f"get_phase_overrides() returned {type(overrides).__name__}, expected dict", file=sys.stderr)
        sys.exit(1)

    out = SHARED / f"phase{n}_best.json"
    out.write_text(json.dumps(overrides, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out.relative_to(ROOT)} ({len(overrides)} keys)")


if __name__ == "__main__":
    main()
