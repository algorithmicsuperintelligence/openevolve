"""
Final verification for cpsat-bench: on a final-test sample, measure LOCAL
baseline elapsed_ms (+ objective_value) and then run the optimized program.
Report per-problem cost_ratio + time speedup using the fresh local baseline
(not the raw-data baseline recorded on a different machine).

Usage:
    python final_verify.py <program_path>

Example:
    python input/cpsat-bench/evolve/final_verify.py \\
        input/cpsat-bench/evolve/phase4_unified/openevolve_output/best/best_program.py

Sample selection (in priority order):
    1. shared/final_sample.json — JSON file with {"sha256": [<sha>, ...]}.
       Hand-edit or generate this to pin a specific subset for verification.
    2. Fall back to ALL problems in problems.jsonl.

Order of operations:
    for each problem p in final sample:
        run BASELINE on p             → record base_ms_local, base_obj_local
        run <program_path> params on p → record variant_ms, variant_obj
        cost_ratio = base_obj_local / variant_obj   (cost mode, minimize)
        speedup    = base_ms_local / variant_ms     (when status decisive)

Baseline + variant are run back-to-back per problem so they share the same
warm cache / system noise. Concurrency = config parallel_solvers (taskset
pinned via OPENEVOLVE_CORE_RANGE or 1..N).
"""
import importlib.util
import json
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "shared"))

from baseline_params import BASELINE, LOCKED  # noqa: E402
from score import score  # noqa: E402
from cpsat_runner import run_cpsat  # noqa: E402
from runtime import parallel_solvers, core_range, alloc_core_blocks  # noqa: E402

_BENCH_DIR = _HERE.parent
_RAW_DIR = _BENCH_DIR / "raw-data"
_PROBLEMS_JSONL = _BENCH_DIR / "problems.jsonl"
_FINAL_SAMPLE = _HERE / "shared" / "final_sample.json"

TIMEOUT_S = 300
_DECISIVE = ("OPTIMAL", "FEASIBLE")


def _load_program(program_path):
    spec = importlib.util.spec_from_file_location("program", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_params"):
        print(f"ERROR: {program_path} missing get_params()", file=sys.stderr)
        sys.exit(2)
    return module


def _load_problem_index():
    idx = {}
    with open(_PROBLEMS_JSONL) as f:
        for line in f:
            d = json.loads(line)
            idx[d["problem_sha256"]] = {
                "sha": d["problem_sha256"],
                "problem_filename": d["problem_filename"],
                "raw_ms": (d.get("cpsat_status") or {}).get("elapsed_ms", 0),
                "raw_result": (d.get("cpsat_status") or {}).get("result"),
            }
    return idx


def _resolve_sample(idx):
    if _FINAL_SAMPLE.exists():
        shas = list(json.loads(_FINAL_SAMPLE.read_text())["sha256"])
        source = f"shared/final_sample.json ({len(shas)} SHAs)"
    else:
        shas = list(idx.keys())
        source = f"problems.jsonl (full {len(shas)})"
    metas = []
    for sha in shas:
        meta = idx.get(sha)
        if meta is None:
            print(f"ERROR: {sha[:12]} from sample not in problems.jsonl", file=sys.stderr)
            sys.exit(2)
        pb = _RAW_DIR / meta["problem_filename"]
        if not pb.exists():
            print(f"ERROR: missing {pb}", file=sys.stderr)
            sys.exit(2)
        metas.append((meta, pb))
    return metas, source


def main():
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2

    program_path = pathlib.Path(sys.argv[1]).resolve()
    if not program_path.exists():
        print(f"ERROR: {program_path} not found", file=sys.stderr)
        return 2

    program = _load_program(program_path)
    variant_params = program.get_params()
    # Use the program's PHASE_LOCKED if it exposes one (later phases typically
    # lock num_search_workers higher than baseline); fall back to global LOCKED.
    _phase_locked = getattr(program, "PHASE_LOCKED", None)
    _lock = _phase_locked if isinstance(_phase_locked, dict) else LOCKED
    violations = {k: variant_params.get(k) for k in _lock
                  if variant_params.get(k) != _lock[k]}
    if violations:
        print(f"ERROR: locked params violated: {violations}", file=sys.stderr)
        return 2

    idx = _load_problem_index()
    metas, source = _resolve_sample(idx)

    cores = core_range()
    if cores is None:
        cores = list(range(1, parallel_solvers(default=1) + 1))
    workers_per_solve = int(variant_params.get("num_search_workers", 1) or 1)
    blocks = alloc_core_blocks(cores, workers_per_solve)
    if not blocks:
        blocks = [list(cores)] if cores else [None]
    n_parallel = min(len(blocks), len(metas))
    blocks = blocks[:n_parallel]
    # Baseline run still uses workers=1 (single core). Give it the first core
    # of each block so it's pinned within the same NUMA neighborhood as the
    # variant for fair comparison.
    baseline_cores = [(b[0] if isinstance(b, (list, tuple)) and b else b) for b in blocks]

    print(f"final verify: {program_path}")
    print(f"  sample : {source}")
    print(f"  params : {len(variant_params)} keys, "
          f"{sum(1 for k, v in variant_params.items() if BASELINE.get(k) != v)} differ from BASELINE")
    print(f"  parallel solvers : {n_parallel} variant_workers={workers_per_solve} blocks={blocks}")
    print(f"  per-problem timeout : {TIMEOUT_S}s × 2 (baseline + variant)")
    print()

    def _measure(idx_meta):
        i, meta, pb = idx_meta
        slot = i % n_parallel if n_parallel > 0 else 0
        v_core = blocks[slot] if n_parallel > 0 else None
        b_core = baseline_cores[slot] if n_parallel > 0 else None
        b = run_cpsat(pb, BASELINE, TIMEOUT_S, cpu_core=b_core)
        v = run_cpsat(pb, variant_params, TIMEOUT_S, cpu_core=v_core)
        return i, meta, b, v

    tasks = [(i, meta, pb) for i, (meta, pb) in enumerate(metas)]
    t_start = time.monotonic()
    completed = []
    if n_parallel == 1:
        for t in tasks:
            completed.append(_measure(t))
    else:
        with ThreadPoolExecutor(max_workers=n_parallel) as ex:
            futures = [ex.submit(_measure, t) for t in tasks]
            for fut in as_completed(futures):
                completed.append(fut.result())
    completed.sort(key=lambda x: x[0])
    elapsed = time.monotonic() - t_start

    results = []
    for i, meta, b, v in completed:
        base_ms_local = int(b.get("elapsed_ms", 0))
        base_result = b.get("result", "Unknown")
        base_obj = b.get("objective")
        var_ms = int(v.get("elapsed_ms", 0))
        var_result = v.get("result", "Unknown")
        var_obj = v.get("objective")
        var_invalid = v.get("invalid_param")

        speedup = base_ms_local / max(var_ms, 1)
        if var_invalid:
            flag = f"  INVALID_PARAM={var_invalid}"
            speedup = 0.0
            cost_ratio = 0.0
        elif base_result not in _DECISIVE and var_result not in _DECISIVE:
            # Both failed to reach decisive status — uncomparable, NOT a regression.
            # score.py skips this from geomean.
            flag = f"  BOTH_NONDECISIVE ({base_result}=base, {var_result}=variant) — skipped from score"
            cost_ratio = 0.0
        elif base_result in _DECISIVE and var_result not in _DECISIVE:
            # Baseline solved, variant didn't → real regression.
            flag = f"  REGRESSION (base={base_result} variant={var_result})"
            speedup = 0.0
            cost_ratio = 0.0
        elif base_result not in _DECISIVE and var_result in _DECISIVE:
            # Variant solved a problem baseline couldn't — bonus, but cost mode
            # can't ratio it. Show but don't score.
            flag = f"  VARIANT_WIN_UNCOMPARABLE (base={base_result})"
            cost_ratio = 0.0
        else:
            if base_obj is not None and var_obj is not None:
                cost_ratio = (float(base_obj) + 1e-9) / (float(var_obj) + 1e-9)
            else:
                cost_ratio = 1.0
            flag = ""

        print(
            f"  [{i+1:>2}/{len(metas)}] {meta['sha'][:10]}  "
            f"base_local={base_result:<10}/{base_ms_local:>7}ms"
            f"{('/obj=' + format(base_obj, '.3g')) if base_obj is not None else ''}  "
            f"variant={var_result:<10}/{var_ms:>7}ms"
            f"{('/obj=' + format(var_obj, '.3g')) if var_obj is not None else ''}  "
            f"speedup={speedup:.2f}x  cost_ratio={cost_ratio:.3f}{flag}",
            flush=True,
        )
        results.append({
            "sha": meta["sha"],
            "input_file": meta["problem_filename"],
            "baseline_ms": base_ms_local,
            "baseline_result": base_result,
            "baseline_objective": base_obj,
            "result": var_result,
            "elapsed_ms": var_ms,
            "objective": var_obj,
            "timeout": bool(v.get("timeout")),
            "raw_baseline_ms": meta["raw_ms"],
            "stats": v.get("stats") or {},
            "baseline_stats": b.get("stats") or {},
        })

    metrics = score(results)
    print()
    print("== summary (cost-mode, vs fresh LOCAL baseline) ==")
    print(f"  total problems  : {metrics['total']}")
    print(f"  comparable      : {metrics['comparable']}  "
          f"(baseline reached decisive status; geomean computed over this subset)")
    print(f"  uncomparable    : {metrics['uncomparable']}  "
          f"(baseline non-decisive; skipped from score)")
    print(f"  solved          : {metrics['solved']}/{metrics['comparable']}")
    print(f"  regressions     : {metrics['regressions']}  "
          f"(baseline OK, variant failed)")
    print(f"  geomean (cost×time) : {metrics['geomean_speedup']:.3f}")
    print(f"  solved_rate     : {metrics['solved_rate']:.3f}  (over comparable)")
    print(f"  efficiency      : {metrics.get('efficiency', 1.0):.3f}")
    print(f"  combined_score  : {metrics['combined_score']:.3f}")
    print(f"  wall-clock      : {elapsed:.1f}s")

    out_path = program_path.parent / "final_verify.json"
    out_path.write_text(json.dumps({
        "program": str(program_path),
        "sample_source": source,
        "metrics": metrics,
        "per_problem": [
            {
                "sha": r["sha"][:12],
                "base_result": r["baseline_result"],
                "got_result": r["result"],
                "base_local_ms": r["baseline_ms"],
                "variant_ms": r["elapsed_ms"],
                "base_obj": r["baseline_objective"],
                "variant_obj": r["objective"],
                "raw_baseline_ms": r["raw_baseline_ms"],
                "timeout": r["timeout"],
            }
            for r in results
        ],
    }, indent=2) + "\n")
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
