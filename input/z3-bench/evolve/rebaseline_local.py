"""
Init-phase rebaseline: measure BASELINE elapsed_ms on the union of
stage1_sample.json + stage2_sample.json (10 SHAs total) and write
shared/local_baseline.json.

Wall-clock varies by hardware / z3 version. Raw-data baseline_ms was recorded
on a different machine, so comparing against it gives misleading speedup.
evaluator._load_problems overlays this local file onto raw data so that
speedup = local_baseline_ms / variant_elapsed_ms.

Stage1+stage2 are rebaselined — both feed the evolve loop (stage1 fast
triage, stage2 medium-slow regression check). Stage3 (50 SHAs, broad
size-distribution) is NOT rebaselined here because final_verify.py
re-measures baseline on the fly per problem.

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
from runtime import parallel_solvers  # noqa: E402

_BENCH_DIR = _HERE.parent
_RAW_DIR = _BENCH_DIR / "raw-data"
_PROBLEMS_JSONL = _BENCH_DIR / "problems.jsonl"
_STAGE1_SAMPLE = _HERE / "shared" / "stage1_sample.json"
_STAGE2_SAMPLE = _HERE / "shared" / "stage2_sample.json"
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
    # Rebaseline stage1 + stage2 (10 SHAs total). Stage3 (50) is skipped —
    # final_verify.py rebaselines on the fly per-problem and the cost of
    # rebaselining 50 with multi-min runtimes is too high for routine setup.
    if not _STAGE1_SAMPLE.exists():
        print(f"ERROR: {_STAGE1_SAMPLE} missing — run build_samples.py first",
              file=sys.stderr)
        sys.exit(2)
    shas = list(json.loads(_STAGE1_SAMPLE.read_text())["sha256"])
    if _STAGE2_SAMPLE.exists():
        s2 = json.loads(_STAGE2_SAMPLE.read_text())["sha256"]
        seen = set(shas)
        for sha in s2:
            if sha not in seen:
                shas.append(sha)
                seen.add(sha)
    else:
        print(f"WARN: {_STAGE2_SAMPLE} missing — rebaselining stage1 only",
              file=sys.stderr)
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
    n_parallel = min(parallel_solvers(default=1), len(tasks))
    print(f"rebaselining stage1+stage2 evolution samples: {len(tasks)} problems "
          f"(union of stage1_sample.json + stage2_sample.json)")
    print(f"timeout per problem = {REBASELINE_TIMEOUT_S}s (effectively unbounded "
          f"— never cut a baseline run short), parallel={n_parallel} (taskset core pin)")
    print()

    # Cores leased from a queue so each in-flight task holds a unique slot.
    # Cores 1..n_parallel — core 0 reserved for kernel interrupts / housekeeping
    # (avoids tail-latency spikes). Serial mode also leases core 1, symmetric
    # with parallel so baseline measurement matches the pin envelope variants
    # will see during evolve.
    _core_pool = _queue.Queue()
    for _c in range(1, n_parallel + 1):
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
