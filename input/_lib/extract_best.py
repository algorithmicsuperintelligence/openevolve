"""
Generic per-phase winner extractor. Per-bench `extract_best.py` wrappers
provide PHASE_DIRS and call `main(root, shared, phase_dirs)`.

After phase N completes, load get_phase_overrides() from its best_program.py
(default: openevolve_output/best/best_program.py) and write
shared/phaseN_best.json. The next phase's initial_program.py picks it up.

--from-checkpoints scans openevolve_output/checkpoints/checkpoint_*/ and
picks the program with the highest combined_score (best_program_info.json).
"""
import argparse
import importlib.util
import json
import pathlib
import sys


def _pick_from_checkpoints(phase_dir):
    ckpt_root = phase_dir / "openevolve_output" / "checkpoints"
    ckpts = sorted(
        ckpt_root.glob("checkpoint_*"),
        key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else -1,
    )
    if not ckpts:
        print(f"no checkpoints found under {ckpt_root}", file=sys.stderr)
        sys.exit(1)

    best_py = None
    best_score = float("-inf")
    best_ck = None
    for ck in ckpts:
        info_path = ck / "best_program_info.json"
        prog_path = ck / "best_program.py"
        if not info_path.exists() or not prog_path.exists():
            continue
        try:
            info = json.loads(info_path.read_text())
            sc = float(info.get("metrics", {}).get("combined_score", float("-inf")))
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"warning: failed to read {info_path}: {e}", file=sys.stderr)
            continue
        if sc > best_score:
            best_score = sc
            best_py = prog_path
            best_ck = ck

    if best_py is None:
        print(f"no usable best_program.py in any checkpoint under {ckpt_root}",
              file=sys.stderr)
        sys.exit(1)

    print(f"[extract_best] from-checkpoints: picked {best_ck.name} "
          f"(combined_score={best_score:.4f})")
    return best_py


def main(root, shared, phase_dirs, argv=None):
    """
    root        — bench evolve dir (pathlib.Path)
    shared      — bench evolve/shared dir (pathlib.Path)
    phase_dirs  — {phase_int: "phaseN_<name>"} (excludes the final/unified phase)
    """
    choices = sorted(phase_dirs.keys())
    ap = argparse.ArgumentParser(
        description="Extract get_phase_overrides() from phase's best_program.py.")
    ap.add_argument("phase", type=int, choices=choices, help="phase number")
    ap.add_argument("--from-checkpoints", action="store_true",
                    help="scan checkpoint_*/ dirs and pick highest combined_score")
    args = ap.parse_args(argv)

    n = args.phase
    phase_dir = pathlib.Path(root) / phase_dirs[n]

    if args.from_checkpoints:
        best_py = _pick_from_checkpoints(phase_dir)
    else:
        best_py = phase_dir / "openevolve_output" / "best" / "best_program.py"
        if not best_py.exists():
            print(f"best_program.py not found: {best_py}", file=sys.stderr)
            print("run phase first (./run_phase.sh N) before extracting,",
                  file=sys.stderr)
            print("or pass --from-checkpoints to use the latest checkpoint.",
                  file=sys.stderr)
            sys.exit(1)

    sys.path.insert(0, str(shared))

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
        print(f"get_phase_overrides() returned {type(overrides).__name__}, "
              f"expected dict", file=sys.stderr)
        sys.exit(1)

    shared_dir = pathlib.Path(shared)
    root_path = pathlib.Path(root)

    out = shared_dir / f"phase{n}_best.json"
    out.write_text(json.dumps(overrides, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out.relative_to(root_path)} ({len(overrides)} keys)")

    # Optional extras for size-bucket / stage3 evolution (cpsat-bench).
    # Programs that don't expose these helpers stay backward-compatible.
    if hasattr(module, "get_phase_size_buckets"):
        try:
            buckets = module.get_phase_size_buckets()
        except Exception as e:
            print(f"get_phase_size_buckets() raised: {e}", file=sys.stderr)
            sys.exit(1)
        if not isinstance(buckets, list):
            print(f"get_phase_size_buckets() returned "
                  f"{type(buckets).__name__}, expected list", file=sys.stderr)
            sys.exit(1)
        # JSON can't encode inf — write None as the sentinel.
        serializable = []
        for entry in buckets:
            if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
                print(f"bucket entry malformed: {entry!r}", file=sys.stderr)
                sys.exit(1)
            upper, override = entry
            if upper == float("inf"):
                upper = None
            elif not isinstance(upper, (int, float)):
                print(f"bucket upper bound not numeric: {upper!r}",
                      file=sys.stderr)
                sys.exit(1)
            if not isinstance(override, dict):
                print(f"bucket override not dict: {override!r}",
                      file=sys.stderr)
                sys.exit(1)
            serializable.append([upper, override])
        out_b = shared_dir / f"phase{n}_buckets.json"
        out_b.write_text(json.dumps(serializable, indent=2, sort_keys=False) + "\n")
        nonempty = sum(1 for _, d in serializable if d)
        print(f"wrote {out_b.relative_to(root_path)} "
              f"({len(serializable)} buckets, {nonempty} non-empty)")

    if hasattr(module, "get_phase_stage3_overrides"):
        try:
            stage3 = module.get_phase_stage3_overrides()
        except Exception as e:
            print(f"get_phase_stage3_overrides() raised: {e}", file=sys.stderr)
            sys.exit(1)
        if not isinstance(stage3, dict):
            print(f"get_phase_stage3_overrides() returned "
                  f"{type(stage3).__name__}, expected dict", file=sys.stderr)
            sys.exit(1)
        out_s = shared_dir / f"phase{n}_stage3.json"
        out_s.write_text(json.dumps(stage3, indent=2, sort_keys=True) + "\n")
        print(f"wrote {out_s.relative_to(root_path)} ({len(stage3)} keys)")
