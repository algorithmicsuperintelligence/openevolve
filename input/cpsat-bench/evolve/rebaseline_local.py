"""
Init-phase rebaseline: measure BASELINE on the union of
stage{1,2,3,4}_sample.json (on the local host, with current CP-SAT version)
and write shared/local_baseline.json. Captures elapsed_ms, stats, AND
objective_value — the last is critical for cost-mode scoring (variant cost
needs a baseline_obj to ratio against).

Wall-clock varies by hardware / ortools version. raw-data timings were
recorded elsewhere; evaluator overlays this local file so per-problem
timeout = baseline_ms * 1.3 and speedup = local_baseline_ms / variant_ms
are calibrated for this box.

Per-problem timeout = REBASELINE_TIMEOUT_S (1 hr safety floor). Never cut a
baseline run short — a truncated baseline poisons every variant comparison.

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
from cpsat_runner import run_cpsat  # noqa: E402
from runtime import parallel_solvers, core_range  # noqa: E402

_BENCH_DIR = _HERE.parent
_RAW_DIR = _BENCH_DIR / "raw-data"
_PROBLEMS_JSONL = _BENCH_DIR / "problems.jsonl"
_STAGE1_SAMPLE = _HERE / "shared" / "stage1_sample.json"
_STAGE2_SAMPLE = _HERE / "shared" / "stage2_sample.json"
_STAGE3_SAMPLE = _HERE / "shared" / "stage3_sample.json"
_STAGE4_SAMPLE = _HERE / "shared" / "stage4_sample.json"
_OUT = _HERE / "shared" / "local_baseline.json"

REBASELINE_TIMEOUT_S = 3600


def _load_problem_index():
    idx = {}
    with open(_PROBLEMS_JSONL) as f:
        for line in f:
            d = json.loads(line)
            sha = d["problem_sha256"]
            idx[sha] = {
                "sha": sha,
                "problem_filename": d["problem_filename"],
                "raw_ms": (d.get("cpsat_status") or {}).get("elapsed_ms", 0),
                "raw_result": (d.get("cpsat_status") or {}).get("result"),
            }
    return idx


def _load_target_shas():
    if not _STAGE1_SAMPLE.exists():
        print(f"ERROR: {_STAGE1_SAMPLE} missing — run build_samples.py first",
              file=sys.stderr)
        sys.exit(2)
    ids = []
    seen = set()
    for sample_path, label in (
        (_STAGE1_SAMPLE, "stage1"),
        (_STAGE2_SAMPLE, "stage2"),
        (_STAGE3_SAMPLE, "stage3"),
        (_STAGE4_SAMPLE, "stage4"),
    ):
        if not sample_path.exists():
            print(f"WARN: {sample_path.name} missing — skipping {label}", file=sys.stderr)
            continue
        for sha in json.loads(sample_path.read_text())["sha256"]:
            if sha not in seen:
                ids.append(sha)
                seen.add(sha)
    return ids


def main():
    shas = _load_target_shas()
    idx = _load_problem_index()

    tasks = []
    for i, sha in enumerate(shas):
        meta = idx.get(sha)
        if meta is None:
            print(f"ERROR: {sha[:12]} not in problems.jsonl", file=sys.stderr)
            return 2
        path = _RAW_DIR / meta["problem_filename"]
        if not path.exists():
            print(f"ERROR: input not found: {path}", file=sys.stderr)
            return 2
        tasks.append((i, meta, path))

    import queue as _queue
    cores = core_range()
    if cores is None:
        cores = list(range(1, parallel_solvers(default=1) + 1))
    n_parallel = min(len(cores), len(tasks))
    cores = cores[:n_parallel]
    print(f"rebaselining union of stage{{1,2,3,4}}_sample.json: {len(tasks)} problems")
    print(f"timeout per problem = {REBASELINE_TIMEOUT_S}s (never cut short), "
          f"parallel={n_parallel} cores={cores}")
    print()

    _core_pool = _queue.Queue()
    for _c in cores:
        _core_pool.put(_c)

    def _solve(task):
        i, meta, path = task
        core = _core_pool.get()
        try:
            res = run_cpsat(path, BASELINE, REBASELINE_TIMEOUT_S, cpu_core=core)
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
        invalid = res.get("invalid_param")
        ok = (got_result == meta["raw_result"]) and not invalid
        if not ok:
            mismatch += 1

        if invalid:
            flag = f"  INVALID_PARAM={invalid}"
        elif ok:
            flag = ""
        else:
            flag = "  MISMATCH"
        ratio = got_ms / max(meta["raw_ms"], 1)
        print(
            f"  [{i+1:>2}/{len(tasks)}] {meta['sha'][:10]}  "
            f"raw={meta['raw_result']:<10}/{int(meta['raw_ms']):>7}ms  "
            f"local={got_result:<10}/{got_ms:>7}ms  ratio={ratio:.2f}x{flag}  "
            f"core={core}",
            flush=True,
        )

        entry = {
            "elapsed_ms": got_ms,
            "result": got_result,
            "matches_raw": ok,
            "raw_elapsed_ms": meta["raw_ms"],
            "stats": res.get("stats") or {},
        }
        if "objective" in res:
            entry["objective"] = res["objective"]
        out[meta["sha"]] = entry

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
