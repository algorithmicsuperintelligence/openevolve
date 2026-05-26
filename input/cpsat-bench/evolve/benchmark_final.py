"""
Repeated-trial benchmark of BASELINE vs final-best on the main test sample.

For each problem in the sample, run BASELINE N times AND the optimized
program's params N times (default 20 each, back-to-back on the same core
to share warm-cache / system-noise envelope). Aggregate mean / median /
stddev / min / max per case and report speedup of variant vs baseline.

Parallelism comes from config.yaml `parallel_solvers` (env
OPENEVOLVE_PARALLEL_SOLVERS overrides) or OPENEVOLVE_CORE_RANGE. Each
problem is pinned to one core; the 40 runs for that problem stay serial
on that core. Different problems run concurrently across cores.

Usage:
    python benchmark_final.py <program_path> [--iters N] [--timeout S]

Outputs (next to <program_path>):
    benchmark_final.json                summary metrics
    benchmark_final_iterations.csv      long-form: per-run row
    benchmark_final_summary.csv         per-problem aggregates

Sample selection (same as final_verify.py):
    1. shared/final_sample.json — JSON file with {"sha256": [<sha>, ...]}.
    2. Fall back to ALL problems in problems.jsonl.
"""
import argparse
import csv
import importlib.util
import inspect
import json
import pathlib
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "shared"))

from baseline_params import BASELINE, LOCKED  # noqa: E402
from cpsat_runner import run_cpsat  # noqa: E402
from runtime import parallel_solvers, core_range, alloc_core_blocks  # noqa: E402

_BENCH_DIR = _HERE.parent
_RAW_DIR = _BENCH_DIR / "raw-data"
_PROBLEMS_JSONL = _BENCH_DIR / "problems.jsonl"
_FINAL_SAMPLE = _HERE / "shared" / "final_sample.json"
_OUTLIERS_JSON = _HERE / "shared" / "outliers.json"

_DEFAULT_ITERS = 20
_DEFAULT_TIMEOUT_S = 300
_DECISIVE = ("OPTIMAL", "FEASIBLE")


def _load_outlier_shas():
    if not _OUTLIERS_JSON.exists():
        return set()
    try:
        d = json.loads(_OUTLIERS_JSON.read_text())
    except (json.JSONDecodeError, OSError):
        return set()
    return set(d.get("outliers") or {})


def _supports_kwargs(fn, *kwargs):
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params_ = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params_.values()):
        return True
    return any(name in params_ for name in kwargs)


def _resolve_params(fn, problem, stage_name):
    kwargs = {}
    if _supports_kwargs(fn, "problem"):
        kwargs["problem"] = problem
    if _supports_kwargs(fn, "stage"):
        kwargs["stage"] = stage_name
    return fn(**kwargs) if kwargs else fn()


def _load_program(program_path):
    spec = importlib.util.spec_from_file_location("program", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_params"):
        print(f"ERROR: {program_path} missing get_params()", file=sys.stderr)
        sys.exit(2)
    return module


def _load_problem_index():
    outliers = _load_outlier_shas()
    idx = {}
    with open(_PROBLEMS_JSONL) as f:
        for line in f:
            d = json.loads(line)
            sha = d["problem_sha256"]
            features = d.get("features") or {}
            idx[sha] = {
                "sha": sha,
                "problem_filename": d["problem_filename"],
                "raw_ms": (d.get("cpsat_status") or {}).get("elapsed_ms", 0),
                "raw_result": (d.get("cpsat_status") or {}).get("result"),
                "num_variables": int(features.get("num_variables") or 0),
                "num_constraints": int(features.get("num_constraints") or 0),
                "num_bool": int(features.get("num_bool") or 0),
                "num_int": int(features.get("num_int") or 0),
                "is_outlier": sha in outliers,
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


def _aggregate(runs):
    """runs = list of dicts {elapsed_ms, result, objective?, timeout, stats?}.
    Returns aggregate stats over only the decisive runs (others as 'failed').
    Tracks both wall (elapsed_ms) and deterministic_time aggregates."""
    decisive = [r for r in runs if r["result"] in _DECISIVE]
    ms_list = [r["elapsed_ms"] for r in decisive]
    dt_list = [(r.get("stats") or {}).get("deterministic_time")
               for r in decisive]
    dt_list = [d for d in dt_list if d is not None and d > 0]
    obj_list = [r["objective"] for r in decisive
                if r.get("objective") is not None]
    agg = {
        "n_total": len(runs),
        "n_decisive": len(decisive),
        "n_failed": len(runs) - len(decisive),
    }
    if ms_list:
        agg["mean_ms"] = statistics.mean(ms_list)
        agg["median_ms"] = statistics.median(ms_list)
        agg["min_ms"] = min(ms_list)
        agg["max_ms"] = max(ms_list)
        agg["stddev_ms"] = statistics.stdev(ms_list) if len(ms_list) > 1 else 0.0
    else:
        agg.update({"mean_ms": 0, "median_ms": 0, "min_ms": 0, "max_ms": 0, "stddev_ms": 0.0})
    if dt_list:
        agg["mean_dt"] = statistics.mean(dt_list)
        agg["median_dt"] = statistics.median(dt_list)
        agg["stddev_dt"] = statistics.stdev(dt_list) if len(dt_list) > 1 else 0.0
        agg["n_dt"] = len(dt_list)
    else:
        agg.update({"mean_dt": 0, "median_dt": 0, "stddev_dt": 0.0, "n_dt": 0})
    if obj_list:
        agg["mean_obj"] = statistics.mean(obj_list)
        agg["min_obj"] = min(obj_list)
        agg["max_obj"] = max(obj_list)
    return agg


def _run_repeat(problem_path, params, n, timeout_s, core, tag):
    runs = []
    for i in range(n):
        r = run_cpsat(problem_path, params, timeout_s, cpu_core=core)
        runs.append({
            "iter": i + 1,
            "elapsed_ms": int(r.get("elapsed_ms", 0)),
            "result": r.get("result", "Unknown"),
            "objective": r.get("objective"),
            "timeout": bool(r.get("timeout")),
            "invalid_param": r.get("invalid_param"),
            "error": r.get("error"),
            "stats": r.get("stats") or {},
        })
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("program", help="path to best_program.py (must expose get_params())")
    ap.add_argument("--iters", type=int, default=_DEFAULT_ITERS,
                    help=f"runs per (problem, variant) pair (default {_DEFAULT_ITERS})")
    ap.add_argument("--timeout", type=int, default=_DEFAULT_TIMEOUT_S,
                    help=f"per-run timeout in seconds (default {_DEFAULT_TIMEOUT_S})")
    args = ap.parse_args()

    program_path = pathlib.Path(args.program).resolve()
    if not program_path.exists():
        print(f"ERROR: {program_path} not found", file=sys.stderr)
        return 2

    program = _load_program(program_path)
    # No-arg path resolves the "global" params used for worker/block sizing,
    # locked-key validation, and summary diff stats. Per-problem variants
    # (SIZE_BUCKETS + STAGE3_OVERRIDES) are resolved inside _worker below.
    variant_params = _resolve_params(program.get_params, None, "final")
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
    baseline_cores = [(b[0] if isinstance(b, (list, tuple)) and b else b) for b in blocks]

    n_iters = args.iters
    timeout_s = args.timeout

    n_total_runs = len(metas) * n_iters * 2
    print(f"benchmark final: {program_path}")
    print(f"  sample          : {source}")
    print(f"  iters per case  : {n_iters} baseline + {n_iters} variant = {n_iters * 2}")
    print(f"  total runs      : {len(metas)} problems × {n_iters * 2} = {n_total_runs}")
    print(f"  per-run timeout : {timeout_s}s")
    print(f"  parallel solvers: {n_parallel} variant_workers={workers_per_solve} blocks={blocks}")
    print(f"  params diff     : {sum(1 for k, v in variant_params.items() if BASELINE.get(k) != v)} keys differ from BASELINE")
    print()

    def _worker(idx_meta):
        i, meta, pb = idx_meta
        slot = i % n_parallel
        v_core = blocks[slot]
        b_core = baseline_cores[slot]
        # Per-problem param resolution. Locked + worker count re-pinned to
        # match `variant_params` so core blocks stay consistent.
        per_params = _resolve_params(program.get_params, meta, "final")
        for k, v in _lock.items():
            per_params[k] = v
        if "num_search_workers" in variant_params:
            per_params["num_search_workers"] = variant_params["num_search_workers"]
        t0 = time.monotonic()
        baseline_runs = _run_repeat(pb, BASELINE, n_iters, timeout_s, b_core, "baseline")
        variant_runs = _run_repeat(pb, per_params, n_iters, timeout_s, v_core, "variant")
        dt = time.monotonic() - t0
        return i, meta, baseline_runs, variant_runs, v_core, dt

    tasks = [(i, meta, pb) for i, (meta, pb) in enumerate(metas)]
    t_start = time.monotonic()
    completed = []
    if n_parallel == 1:
        for t in tasks:
            completed.append(_worker(t))
            i, meta, b_runs, v_runs, core, dt = completed[-1]
            b_agg = _aggregate(b_runs)
            v_agg = _aggregate(v_runs)
            sp_w = (b_agg["mean_ms"] / max(v_agg["mean_ms"], 1)) if v_agg["mean_ms"] else 0
            sp_d = (b_agg["mean_dt"] / max(v_agg["mean_dt"], 1)) if v_agg["mean_dt"] else 0
            sp_main = sp_d if sp_d else sp_w
            print(f"  [{i+1:>2}/{len(metas)}] {meta['sha'][:10]} core={core} "
                  f"base_mean={b_agg['mean_ms']:>7.0f}ms (n_ok={b_agg['n_decisive']}/{n_iters})  "
                  f"variant_mean={v_agg['mean_ms']:>7.0f}ms (n_ok={v_agg['n_decisive']}/{n_iters})  "
                  f"speedup={sp_main:.2f}x[{'dt' if sp_d else 'wall'}] wall={sp_w:.2f}x  (took {dt:.1f}s)",
                  flush=True)
    else:
        with ThreadPoolExecutor(max_workers=n_parallel) as ex:
            futures = [ex.submit(_worker, t) for t in tasks]
            for fut in as_completed(futures):
                completed.append(fut.result())
                i, meta, b_runs, v_runs, core, dt = completed[-1]
                b_agg = _aggregate(b_runs)
                v_agg = _aggregate(v_runs)
                sp = (b_agg["mean_ms"] / max(v_agg["mean_ms"], 1)) if v_agg["mean_ms"] else 0
                print(f"  [{i+1:>2}/{len(metas)}] {meta['sha'][:10]} core={core} "
                      f"base_mean={b_agg['mean_ms']:>7.0f}ms (n_ok={b_agg['n_decisive']}/{n_iters})  "
                      f"variant_mean={v_agg['mean_ms']:>7.0f}ms (n_ok={v_agg['n_decisive']}/{n_iters})  "
                      f"speedup={sp:.2f}x  (took {dt:.1f}s)",
                      flush=True)
    completed.sort(key=lambda x: x[0])
    wall = time.monotonic() - t_start

    # ----- write CSVs -----
    out_dir = program_path.parent
    iter_csv = out_dir / "benchmark_final_iterations.csv"
    summary_csv = out_dir / "benchmark_final_summary.csv"
    json_path = out_dir / "benchmark_final.json"

    with open(iter_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sha", "variant", "iter", "elapsed_ms", "result",
                    "objective", "timeout", "invalid_param", "error"])
        for _, meta, b_runs, v_runs, _, _ in completed:
            sha = meta["sha"]
            for r in b_runs:
                w.writerow([sha, "baseline", r["iter"], r["elapsed_ms"], r["result"],
                            r["objective"] if r["objective"] is not None else "",
                            int(r["timeout"]),
                            r["invalid_param"] or "",
                            (r["error"] or "")[:200]])
            for r in v_runs:
                w.writerow([sha, "variant", r["iter"], r["elapsed_ms"], r["result"],
                            r["objective"] if r["objective"] is not None else "",
                            int(r["timeout"]),
                            r["invalid_param"] or "",
                            (r["error"] or "")[:200]])

    summary_rows = []
    for _, meta, b_runs, v_runs, _, _ in completed:
        b_agg = _aggregate(b_runs)
        v_agg = _aggregate(v_runs)
        mean_sp_w = (b_agg["mean_ms"] / max(v_agg["mean_ms"], 1)) if v_agg["mean_ms"] else 0
        med_sp_w = (b_agg["median_ms"] / max(v_agg["median_ms"], 1)) if v_agg["median_ms"] else 0
        mean_sp_d = (b_agg["mean_dt"] / max(v_agg["mean_dt"], 1)) if v_agg["mean_dt"] else 0
        med_sp_d = (b_agg["median_dt"] / max(v_agg["median_dt"], 1)) if v_agg["median_dt"] else 0
        summary_rows.append({
            "sha": meta["sha"],
            "n_iters": n_iters,
            "baseline_n_decisive": b_agg["n_decisive"],
            "baseline_mean_ms": round(b_agg["mean_ms"], 1),
            "baseline_median_ms": round(b_agg["median_ms"], 1),
            "baseline_min_ms": round(b_agg["min_ms"], 1),
            "baseline_max_ms": round(b_agg["max_ms"], 1),
            "baseline_stddev_ms": round(b_agg["stddev_ms"], 1),
            "baseline_mean_dt": round(b_agg["mean_dt"], 3),
            "baseline_median_dt": round(b_agg["median_dt"], 3),
            "baseline_n_dt": b_agg["n_dt"],
            "variant_n_decisive": v_agg["n_decisive"],
            "variant_mean_ms": round(v_agg["mean_ms"], 1),
            "variant_median_ms": round(v_agg["median_ms"], 1),
            "variant_min_ms": round(v_agg["min_ms"], 1),
            "variant_max_ms": round(v_agg["max_ms"], 1),
            "variant_stddev_ms": round(v_agg["stddev_ms"], 1),
            "variant_mean_dt": round(v_agg["mean_dt"], 3),
            "variant_median_dt": round(v_agg["median_dt"], 3),
            "variant_n_dt": v_agg["n_dt"],
            "mean_speedup": round(mean_sp_w, 3),       # wall (legacy)
            "median_speedup": round(med_sp_w, 3),      # wall (legacy)
            "mean_dt_speedup": round(mean_sp_d, 3),    # deterministic (primary)
            "median_dt_speedup": round(med_sp_d, 3),
        })

    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # ----- overall aggregate -----
    # speedup geomean over problems with both means valid. Compute both wall
    # and deterministic-time geomeans; dt is preferred when populated for both
    # baseline + variant on every (problem, iter) pair.
    import math
    valid = [r for r in summary_rows
             if r["baseline_mean_ms"] > 0 and r["variant_mean_ms"] > 0
             and r["baseline_n_decisive"] > 0 and r["variant_n_decisive"] > 0]
    valid_dt = [r for r in valid
                if r["mean_dt_speedup"] > 0 and r["median_dt_speedup"] > 0]
    if valid:
        geo_mean = math.exp(sum(math.log(r["mean_speedup"]) for r in valid) / len(valid))
        geo_med = math.exp(sum(math.log(r["median_speedup"]) for r in valid) / len(valid))
        arith_mean = sum(r["mean_speedup"] for r in valid) / len(valid)
    else:
        geo_mean = geo_med = arith_mean = 0.0
    if valid_dt:
        geo_mean_dt = math.exp(sum(math.log(r["mean_dt_speedup"])
                                   for r in valid_dt) / len(valid_dt))
        geo_med_dt = math.exp(sum(math.log(r["median_dt_speedup"])
                                  for r in valid_dt) / len(valid_dt))
    else:
        geo_mean_dt = geo_med_dt = 0.0

    n_baseline_failed = sum(r["baseline_n_decisive"] < n_iters for r in summary_rows)
    n_variant_failed = sum(r["variant_n_decisive"] < n_iters for r in summary_rows)
    n_baseline_all_failed = sum(r["baseline_n_decisive"] == 0 for r in summary_rows)
    n_variant_all_failed = sum(r["variant_n_decisive"] == 0 for r in summary_rows)

    json_path.write_text(json.dumps({
        "program": str(program_path),
        "sample_source": source,
        "iters_per_case": n_iters,
        "timeout_s": timeout_s,
        "n_parallel": n_parallel,
        "cores": cores,
        "wall_clock_s": round(wall, 1),
        "n_problems": len(metas),
        "summary": {
            "geomean_dt_speedup_mean": round(geo_mean_dt, 3),    # primary
            "geomean_dt_speedup_median": round(geo_med_dt, 3),
            "geomean_wall_speedup_mean": round(geo_mean, 3),     # diagnostic
            "geomean_wall_speedup_median": round(geo_med, 3),
            "arith_mean_wall_speedup": round(arith_mean, 3),
            "n_problems_with_valid_speedup": len(valid),
            "n_problems_with_valid_dt_speedup": len(valid_dt),
            "n_baseline_partial_fail": n_baseline_failed,
            "n_baseline_total_fail": n_baseline_all_failed,
            "n_variant_partial_fail": n_variant_failed,
            "n_variant_total_fail": n_variant_all_failed,
        },
        "per_problem": summary_rows,
    }, indent=2) + "\n")

    print()
    print("== overall (mean-of-means across problems) ==")
    print(f"  problems with valid speedup : wall={len(valid)}/{len(metas)} dt={len(valid_dt)}/{len(metas)}")
    print(f"  geomean dt-speedup (mean)   : {geo_mean_dt:.3f}x  (primary, hardware-independent)")
    print(f"  geomean dt-speedup (median) : {geo_med_dt:.3f}x")
    print(f"  geomean wall-speedup (mean) : {geo_mean:.3f}x  (diagnostic)")
    print(f"  geomean wall-speedup (med)  : {geo_med:.3f}x")
    print(f"  arith mean wall-speedup     : {arith_mean:.3f}x")
    print(f"  baseline partial-fail cases : {n_baseline_failed}  (≥1 of {n_iters} runs non-decisive)")
    print(f"  baseline total-fail cases   : {n_baseline_all_failed}  (all {n_iters} runs non-decisive)")
    print(f"  variant partial-fail cases  : {n_variant_failed}")
    print(f"  variant total-fail cases    : {n_variant_all_failed}")
    print(f"  wall-clock                  : {wall:.1f}s")
    print()
    print(f"  wrote {iter_csv}")
    print(f"  wrote {summary_csv}")
    print(f"  wrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
