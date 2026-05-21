# input/ — Benchmark Pipelines + Shared Lib

Each `<bench>/` directory holds one OpenEvolve solver-parameter-tuning
pipeline. Common scaffolding lives under `_lib/` and is wired into each
bench via thin per-bench wrappers (keeps bench code <100 LOC for the
boilerplate parts).

## Layout

```
input/
├── run_phase.sh                # single entry: ./run_phase.sh <bench> [<phase>] [flags]
├── _lib/                       # shared library (importable as `_lib.*`)
│   ├── runtime.py              # parallel_solvers / cascade_threshold / core_range
│   ├── subprocess_runner.py    # run_solver(worker, problem, params, timeout, ...)
│   ├── extract_best.py         # main(root, shared, phase_dirs) — CLI core
│   ├── prepare_phase.py        # main(root, shared, prior_phases, unified_file)
│   └── load_bench_config.py    # parse `bench:` section of config.yaml → bash exports
├── z3-bench/
│   ├── raw-data/               # <sha>.smt2 + <sha>__<hash>__seed0.meta.jsonl
│   └── evolve/
│       ├── config.yaml         # openevolve + `bench:` (phases / scripts / solver check)
│       └── ...                 # baseline_params, evaluator, phases, etc.
└── cpsat-bench/
    ├── raw-data/               # <sha>.cpsat.pb + <sha>__<hash>__seed0.meta.jsonl
    └── evolve/
        ├── config.yaml         # same
        └── ...
```

## _lib/ — what's shared

| module | purpose | per-bench wrapper |
|---|---|---|
| `_lib/runtime.py` | parse `OPENEVOLVE_PARALLEL_SOLVERS`, `OPENEVOLVE_CORE_RANGE`, `cascade_thresholds` from env + config.yaml | `<bench>/evolve/shared/runtime.py` (binds config path, re-exports) |
| `_lib/subprocess_runner.py` | spawn worker subprocess, taskset pin, timeout grace, parse stdout JSON | `<bench>/evolve/shared/<solver>_runner.py` (binds WORKER path, names `run_<solver>`) |
| `_lib/extract_best.py` | argparse + load best_program.py + write `phaseN_best.json` | `<bench>/evolve/extract_best.py` (passes `PHASE_DIRS` dict) |
| `_lib/prepare_phase.py` | merge `phaseN_best.json` files, rewrite unified phase's EVOLVE-BLOCK | `<bench>/evolve/prepare_phase_unified.py` (passes `PRIOR_PHASES` list) |
| `input/run_phase.sh` | single user-facing entry: `./run_phase.sh <bench> [<phase>] [--pin] [--extract-only] [--iterations]`. Phase optional → runs ALL phases sequentially. Reads `bench:` section of `<bench>/evolve/config.yaml` via `_lib/load_bench_config.py`. | `<bench>/evolve/config.yaml` — `bench:` key with `phases[].dir/iters`, `unified_prepare_script`, `solver_check_cmd`, `solver_install_hint` |

## Per-bench files NOT shared

Stay per-bench because each is solver- or dataset-specific:

- `shared/baseline_params.py` — `BASELINE` + `LOCKED` dicts (recorded params)
- `shared/_<solver>_solve_worker.py` — solver Python binding glue
- `shared/score.py` — score mode (`decision` / `cost` / `time-only`)
- `shared/evaluator.py` — cascade scaffold + decisive-result set + key stats
- `build_samples.py` — meta.jsonl field paths
- `rebaseline_local.py` — local baseline measurement
- `phase{N}_<name>/initial_program.py` — phase namespace + initial overrides
- `config.yaml` — bench knobs + LLM system message

## Wiring pattern (per-bench wrapper)

Each wrapper:
1. Computes `_INPUT_DIR` (relative to its own file).
2. Inserts it into `sys.path` so `from _lib import ...` works.
3. Calls / re-exports the shared function with bench-specific args.

Example — `<bench>/evolve/shared/runtime.py`:

```python
import pathlib, sys
_HERE = pathlib.Path(__file__).resolve().parent
_CONFIG_YAML = _HERE.parent / "config.yaml"
_INPUT_DIR = _HERE.parents[2]
if str(_INPUT_DIR) not in sys.path:
    sys.path.insert(0, str(_INPUT_DIR))
from _lib import runtime as _rt

core_range = _rt.core_range
def parallel_solvers(default=1): return _rt.parallel_solvers(_CONFIG_YAML, default=default)
def cascade_threshold(i, d):     return _rt.cascade_threshold(_CONFIG_YAML, i, d)
```

Same pattern for `<solver>_runner.py`, `extract_best.py`, `prepare_phase_unified.py`.

## Adding a new bench

1. Make `input/<bench>/raw-data/` and populate it (e.g. `load_script.sh`).
2. Scaffold `<bench>/evolve/` using the [openevolve-pipeline skill](../.claude/skills/openevolve-pipeline/SKILL.md).
3. For each shared module, copy a per-bench wrapper from `z3-bench/` or
   `cpsat-bench/` and adjust the bench-specific bits (worker path, phase
   dirs, config path).

## Environment knobs (work in both benches)

| variable | use |
|---|---|
| `OPENEVOLVE_PARALLEL_SOLVERS` | concurrent solver subprocesses per stage (cores 1..N) |
| `OPENEVOLVE_CORE_RANGE` | explicit taskset core range `N-M` (overrides PARALLEL_SOLVERS for pinning; concurrency = range size) |
| `OPENEVOLVE_MAX_PROBLEMS` | cap stage problem count (testing) |
| `OPENEVOLVE_STATS_WEIGHT` | exponent on efficiency factor in score.py (0 disables) |
| `OPENEVOLVE_COST_WEIGHT` | (cost-mode only) exponent on cost_ratio |
| `OPENEVOLVE_PYTHON_BIN` | python used by solver worker subprocess |
| `SKIP_REBASELINE` | skip per-host baseline remeasurement (reuse `local_baseline.json`) |

## Quick start (any bench)

```bash
# (cpsat-bench example — swap to z3-bench by name)
bash input/cpsat-bench/raw-data/load_script.sh              # populate raw-data
python input/cpsat-bench/evolve/build_samples.py            # stage{1..4}_sample.json
python input/cpsat-bench/evolve/shared/baseline_params.py   # sanity self-test

# Run all phases (1..N) with one command:
./input/run_phase.sh cpsat-bench --pin 2-7

# Or step phase-by-phase:
./input/run_phase.sh cpsat-bench 1 --pin 2-7
./input/run_phase.sh cpsat-bench 2 --pin 2-7
./input/run_phase.sh cpsat-bench 3 --pin 2-7
./input/run_phase.sh cpsat-bench 4 --pin 2-7

# Final verify on the unified winner (fresh LOCAL baseline per problem):
python input/cpsat-bench/evolve/final_verify.py \
    input/cpsat-bench/evolve/phase4_unified/openevolve_output/best/best_program.py
# (writes final_verify.json next to best_program.py)
```
