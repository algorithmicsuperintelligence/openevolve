"""
Init-phase rebaseline: measure BASELINE elapsed_ms on the union of
stage1 + stage2 + stage3 + stage4 sample files and write
shared/local_baseline.json.

Wall-clock varies by hardware / z3 version. Raw-data baseline_ms was recorded
on a different machine, so comparing against it gives misleading speedup.
evaluator._load_problems overlays this local file onto raw data so that
speedup = local_baseline_ms / variant_elapsed_ms and per-problem timeout
= baseline_ms * 1.3 is calibrated for the local box.

All three stages feed the evolve loop (cascade_evaluation: true). Without
rebaselining stage3 (50 SHAs), 45/50 problems would fall back to raw_ms
recorded elsewhere and timeout/speedup calculations would drift.
final_verify.py re-measures baseline on the fly per problem anyway, but
the evolve-loop stage3 invocation needs trustworthy local baselines up front.

Per-problem: 1 run, timeout = REBASELINE_TIMEOUT_S (1 hr safety floor — a
truncated baseline measurement is worse than a slow one). MISMATCH-by-timeout
that the old multiplier produced would poison local_baseline. Big problems
under parallel contention may run ~2x raw; let them finish.
Concurrency = config parallel_solvers (env OPENEVOLVE_PARALLEL_SOLVERS override).
"""
import json
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "shared"))

from baseline_params import BASELINE  # noqa: E402
from z3_runner import run_z3  # noqa: E402
from runtime import parallel_solvers, core_range  # noqa: E402

_BENCH_DIR = _HERE.parent
_RAW_DIR = _BENCH_DIR / "raw-data"
_PROBLEMS_JSONL = _BENCH_DIR / "problems.jsonl"
_STAGE1_SAMPLE = _HERE / "shared" / "stage1_sample.json"
_STAGE2_SAMPLE = _HERE / "shared" / "stage2_sample.json"
_STAGE3_SAMPLE = _HERE / "shared" / "stage3_sample.json"
_STAGE4_SAMPLE = _HERE / "shared" / "stage4_sample.json"
_OUT = _HERE / "shared" / "local_baseline.json"

# Baseline measurement must never be truncated — let z3 finish naturally.
# 1 hour cap is just a safety against true infinite loops; not expected to trigger.
REBASELINE_TIMEOUT_S = 3600


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


def _load_target_shas():
    # Rebaseline the union of stage1 + stage2 + stage3 samples. With cascade
    # evaluation enabled all three stages run in the evolve loop, so accurate
    # local baseline is required for per-problem timeout = baseline_ms * 1.3
    # to be machine-correct. Without rebaselining stage3, 45/50 problems fall
    # back to raw_ms recorded on a different machine and timeout calibration
    # drifts.
    if not _STAGE1_SAMPLE.exists():
        print(f"ERROR: {_STAGE1_SAMPLE} missing — run build_samples.py first",
              file=sys.stderr)
        sys.exit(2)
    shas = []
    seen = set()
    for sample_path, label in (
        (_STAGE1_SAMPLE, "stage1"),
        (_STAGE2_SAMPLE, "stage2"),
        (_STAGE3_SAMPLE, "stage3"),
        (_STAGE4_SAMPLE, "stage4"),
    ):
        if not sample_path.exists():
            print(f"WARN: {sample_path.name} missing — skipping {label}",
                  file=sys.stderr)
            continue
        for sha in json.loads(sample_path.read_text())["sha256"]:
            if sha not in seen:
                shas.append(sha)
                seen.add(sha)
    return shas


def main():
    shas = _load_target_shas()
    idx = _load_problem_index()

    tasks = []
    for i, sha in enumerate(shas):
        meta = idx.get(sha)
        if meta is None:
            print(f"ERROR: {sha[:12]} not in problems.jsonl", file=sys.stderr)
            return 2
        smt2_path = _RAW_DIR / meta["smt2"]
        if not smt2_path.exists():
            print(f"ERROR: smt2 not found: {smt2_path}", file=sys.stderr)
            return 2
        tasks.append((i, meta, smt2_path))

    import queue as _queue
    # Cores leased from a queue so each in-flight task holds a unique slot.
    # OPENEVOLVE_CORE_RANGE (e.g. "1-5") overrides; else cores 1..N from
    # parallel_solvers (core 0 reserved for kernel housekeeping).
    cores = core_range()
    if cores is None:
        cores = list(range(1, parallel_solvers(default=1) + 1))
    n_parallel = min(len(cores), len(tasks))
    cores = cores[:n_parallel]
    print(f"rebaselining stage1+stage2+stage3+stage4 samples: {len(tasks)} problems "
          f"(union of stage{{1,2,3,4}}_sample.json)")
    print(f"timeout per problem = {REBASELINE_TIMEOUT_S}s (effectively unbounded "
          f"— never cut a baseline run short), parallel={n_parallel} cores={cores}")
    print()

    _core_pool = _queue.Queue()
    for _c in cores:
        _core_pool.put(_c)

    def _solve(task):
        i, meta, smt2_path = task
        core = _core_pool.get()
        try:
            res = run_z3(smt2_path, BASELINE, REBASELINE_TIMEOUT_S, cpu_core=core)
        finally:
            _core_pool.put(core)
        return i, meta, res, core

    t_start = time.monotonic()
    completed = []
    if n_parallel == 1:
        for task in tasks:
            completed.append(_solve(task))
    else:
        with ThreadPoolExecutor(max_workers=n_parallel) as ex:
            futures = [ex.submit(_solve, t) for t in tasks]
            for fut in as_completed(futures):
                completed.append(fut.result())
    completed.sort(key=lambda x: x[0])

    out = {}
    mismatch = 0
    for i, meta, res, core in completed:
        got_result = res.get("result", "Unknown")
        got_ms = int(res.get("elapsed_ms", 0))
        ok = (got_result == meta["raw_result"])
        if not ok:
            mismatch += 1

        flag = "" if ok else "  MISMATCH"
        ratio = got_ms / max(meta["raw_ms"], 1)
        print(
            f"  [{i+1:>2}/{len(tasks)}] {meta['sha'][:10]}  "
            f"raw={meta['raw_result']:<7}/{meta['raw_ms']:>7}ms  "
            f"local={got_result:<7}/{got_ms:>7}ms  ratio={ratio:.2f}x{flag}  "
            f"core={core}",
            flush=True,
        )

        out[meta["sha"]] = {
            "elapsed_ms": got_ms,
            "result": got_result,
            "matches_raw": ok,
            "raw_elapsed_ms": meta["raw_ms"],
            "stats": res.get("stats") or {},
        }

    elapsed = time.monotonic() - t_start
    _OUT.write_text(json.dumps(out, indent=2) + "\n")
    print()
    print(f"wrote {_OUT.relative_to(_BENCH_DIR.parent)} "
          f"({len(out)} entries, {mismatch} mismatches)")
    print(f"total time: {elapsed:.1f}s")
    if mismatch:
        print(f"WARNING: {mismatch} problems had result mismatch — "
              f"evaluator will keep raw_ms for those")
    return 0 if mismatch == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
