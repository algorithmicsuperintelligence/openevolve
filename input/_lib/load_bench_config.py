"""
Read `bench:` section from a bench's config.yaml and print bash `export`
statements. input/run_phase.sh evals the stdout.

Usage:
    eval "$(python _lib/load_bench_config.py <bench>/evolve/config.yaml)"

Schema:
    bench:
      phases:
        - dir: phase1_x
          iters: 60        # optional; null/omitted => fall back to max_iterations
        - dir: phase2_y
        ...
      unified_prepare_script: prepare_phase_unified.py    # optional
      solver_check_cmd: 'python -c "from ortools.sat.python import cp_model"'
      solver_install_hint: install pip install ortools
      rebaseline_script: rebaseline_local.py              # optional override
      extract_best_script: extract_best.py                # optional override
      build_samples_script: build_samples.py              # optional override

Output (stdout):
    export PHASE_DIRS='phase1_x phase2_y ...'
    export PHASE_ITERS='60  80 40'    # space-separated; missing iters -> empty slot
    export UNIFIED_PREPARE_SCRIPT='prepare_phase_unified.py'
    export SOLVER_CHECK_CMD='python -c "..."'
    ...

All values shell-quoted via shlex.quote. Non-zero exit on missing/invalid
`bench` section.
"""
import shlex
import sys


def _emit(key, value):
    if value is None or value == "":
        return
    print(f"export {key}={shlex.quote(str(value))}")


def main():
    if len(sys.argv) != 2:
        print("usage: load_bench_config.py <config.yaml>", file=sys.stderr)
        sys.exit(2)

    try:
        import yaml
    except ImportError as e:
        print(f"PyYAML not importable: {e}", file=sys.stderr)
        sys.exit(2)

    path = sys.argv[1]
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"config not found: {path}", file=sys.stderr)
        sys.exit(2)
    except yaml.YAMLError as e:
        print(f"YAML parse error in {path}: {e}", file=sys.stderr)
        sys.exit(2)

    bench = cfg.get("bench")
    if not isinstance(bench, dict):
        print(f"missing or non-dict `bench:` section in {path}", file=sys.stderr)
        sys.exit(2)

    phases = bench.get("phases")
    if not isinstance(phases, list) or not phases:
        print(f"`bench.phases` must be a non-empty list in {path}", file=sys.stderr)
        sys.exit(2)

    dirs = []
    iters = []
    any_iter = False
    for i, p in enumerate(phases, start=1):
        if not isinstance(p, dict) or "dir" not in p:
            print(f"bench.phases[{i-1}] must be a dict with `dir` key", file=sys.stderr)
            sys.exit(2)
        dirs.append(str(p["dir"]))
        it = p.get("iters")
        if it is None or it == "":
            iters.append("")
        else:
            iters.append(str(int(it)))
            any_iter = True

    _emit("PHASE_DIRS", " ".join(dirs))
    # If no phase has iters set, omit PHASE_ITERS entirely so shell array is empty
    # (and config.yaml max_iterations applies). Otherwise emit space-joined list
    # with empty slots for phases without explicit iters.
    if any_iter:
        _emit("PHASE_ITERS", " ".join(iters))

    _emit("UNIFIED_PREPARE_SCRIPT", bench.get("unified_prepare_script"))
    _emit("SOLVER_CHECK_CMD", bench.get("solver_check_cmd"))
    _emit("SOLVER_INSTALL_HINT", bench.get("solver_install_hint"))
    _emit("REBASELINE_SCRIPT", bench.get("rebaseline_script"))
    _emit("EXTRACT_BEST_SCRIPT", bench.get("extract_best_script"))
    _emit("BUILD_SAMPLES_SCRIPT", bench.get("build_samples_script"))


if __name__ == "__main__":
    main()
