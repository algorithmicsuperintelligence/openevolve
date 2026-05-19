"""
Final verification: on a final-test sample, measure LOCAL baseline elapsed_ms
and then run the optimized program. Report per-problem speedup using the
fresh local baseline (not the raw-data baseline which was recorded on a
different machine).

Usage:
    python final_verify.py <program_path>

Sample selection (in priority order):
    1. shared/final_sample.json — JSON file with {"sha256": [<sha>, ...]}.
       Hand-edit or generate this to pin a specific subset for verification.
    2. Fall back to ALL problems in problems.jsonl (50).

Order of operations:
    for each problem p in final sample:
        run BASELINE on p             → record base_ms_local
        run <program_path> params on p → record variant_ms
        speedup = base_ms_local / variant_ms (when result matches)

Baseline + variant are run back-to-back per problem so they share the same
warm cache / system noise. Concurrency = config parallel_solvers (one z3
process pair per problem; each problem runs baseline then variant serially
inside its slot to keep timing apples-to-apples).
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
from z3_runner import run_z3  # noqa: E402
from runtime import parallel_solvers  # noqa: E402

_BENCH_DIR = _HERE.parent
_RAW_DIR = _BENCH_DIR / "raw-data"
_PROBLEMS_JSONL = _BENCH_DIR / "problems.jsonl"
_FINAL_SAMPLE = _HERE / "shared" / "final_sample.json"

TIMEOUT_S = 120


def _load_get_params(program_path):
    spec = importlib.util.spec_from_file_location("program", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_params"):
        print(f"ERROR: {program_path} missing get_params()", file=sys.stderr)
        sys.exit(2)
    return module.get_params()


def _load_problem_index():
    idx = {}
    with open(_PROBLEMS_JSONL) as f:
        for line in f:
            d = json.loads(line)
            idx[d["problem_sha256"]] = {
                "sha": d["problem_sha256"],
                "smt2": d["smt2_filename"],
                "raw_ms": d["z3_status"]["elapsed_ms"],
                "raw_result": d["z3_status"]["result"],
            }
    return idx


def _resolve_sample(idx):
    """Pick the SHA list for final verification."""
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
        smt2 = _RAW_DIR / meta["smt2"]
        if not smt2.exists():
            print(f"ERROR: missing {smt2}", file=sys.stderr)
            sys.exit(2)
        metas.append((meta, smt2))
    return metas, source


def main():
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2

    program_path = pathlib.Path(sys.argv[1]).resolve()
    if not program_path.exists():
        print(f"ERROR: {program_path} not found", file=sys.stderr)
        return 2

    variant_params = _load_get_params(program_path)
    violations = {k: variant_params.get(k) for k in LOCKED
                  if variant_params.get(k) != LOCKED[k]}
    if violations:
        print(f"ERROR: locked params violated: {violations}", file=sys.stderr)
        return 2

    idx = _load_problem_index()
    metas, source = _resolve_sample(idx)
    n_parallel = min(parallel_solvers(default=1), len(metas))

    print(f"final verify: {program_path}")
    print(f"  sample : {source}")
    print(f"  params : {len(variant_params)} keys, "
          f"{sum(1 for k, v in variant_params.items() if BASELINE.get(k) != v)} differ from BASELINE")
    print(f"  parallel solvers : {n_parallel} (taskset core pin)")
    print(f"  per-problem timeout : {TIMEOUT_S}s × 2 (baseline + variant)")
    print()

    def _measure(idx_meta):
        i, meta, smt2 = idx_meta
        # Skip core 0 — pin baseline + variant to cores 1..n_parallel.
        core = ((i % n_parallel) + 1) if n_parallel > 1 else None
        # Baseline first, then variant — back-to-back so system noise affects
        # both equally and speedup ratio cancels it out.
        b = run_z3(smt2, BASELINE, TIMEOUT_S, cpu_core=core)
        v = run_z3(smt2, variant_params, TIMEOUT_S, cpu_core=core)
        return i, meta, b, v

    tasks = [(i, meta, smt2) for i, (meta, smt2) in enumerate(metas)]
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
        var_ms = int(v.get("elapsed_ms", 0))
        var_result = v.get("result", "Unknown")
        var_invalid = v.get("invalid_param")
        # Speedup uses LOCAL baseline (this run), not raw_ms.
        if var_invalid:
            speedup = 0.0
            flag = f"  invalid={var_invalid}"
        elif var_result != base_result:
            speedup = 0.0
            flag = f"  MISMATCH (base={base_result} variant={var_result})"
        else:
            speedup = base_ms_local / max(var_ms, 1)
            flag = ""
        print(
            f"  [{i+1:>2}/{len(metas)}] {meta['sha'][:10]}  "
            f"base_local={base_result:<7}/{base_ms_local:>7}ms  "
            f"variant={var_result:<7}/{var_ms:>7}ms  "
            f"speedup={speedup:.2f}x{flag}",
            flush=True,
        )
        results.append({
            "sha": meta["sha"],
            "smt2": meta["smt2"],
            "baseline_ms": base_ms_local,
            "baseline_result": base_result,
            "result": var_result,
            "elapsed_ms": var_ms,
            "timeout": bool(v.get("timeout")),
            "raw_baseline_ms": meta["raw_ms"],
        })

    metrics = score(results)
    print()
    print(f"== summary (speedup vs fresh LOCAL baseline) ==")
    print(f"  solved          : {metrics['solved']}/{metrics['total']}")
    print(f"  regressions     : {metrics['regressions']}")
    print(f"  geomean_speedup : {metrics['geomean_speedup']:.3f}")
    print(f"  solved_rate     : {metrics['solved_rate']:.3f}")
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
