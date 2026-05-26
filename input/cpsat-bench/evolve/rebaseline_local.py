"""
Init-phase rebaseline: measure BASELINE on the union of
stage{1,2,3,4}_sample.json (on the local host, with current CP-SAT version)
and write shared/local_baseline.json. Captures elapsed_ms, stats, AND
objective_value — the last is critical for cost-mode scoring (variant cost
needs a baseline_obj to ratio against).

MULTI-WORKER BASELINES (2026-05 revision):
  Each phase's initial_program.py pins num_search_workers via PHASE_LOCKED.
  Comparing a workers=8 variant against a workers=1 baseline conflates
  param-tuning gain with multi-thread parallelism gain. To keep speedup
  honest, we capture one baseline PER unique worker count discovered across
  the phase files, and store them under `by_workers` in the output.

  Auto-discovery: walks sibling phase*_*/initial_program.py and unions every
  PHASE_LOCKED["num_search_workers"]. Override with --workers 1,8 if needed.

STAGE3 OUTLIER POLICY (2026-05 revision):
  Stage3 holds outlier problems with W=8-equivalent baselines from raw-data,
  some > 1500s. Measuring those at W=1 would either timeout or take ~10×
  longer with no value (evaluator skips stage3 for W=1 phases anyway — see
  evaluate_stage3). To save rebaseline wall-clock, this script EXCLUDES stage3
  shas from the W=1 task list and measures stage3 only at W>=2.

Output schema (shared/local_baseline.json):
  {
    "<sha>": {
      "raw_result": "OPTIMAL",
      "raw_elapsed_ms": 4321,
      "by_workers": {
        "1": {"elapsed_ms": ..., "result": ..., "stats": {...},
              "objective": ..., "matches_raw": true},
        "8": {"elapsed_ms": ..., "result": ..., "stats": {...},
              "objective": ..., "matches_raw": true}
      }
    }
  }

Wall-clock varies by hardware / ortools version. raw-data timings were
recorded elsewhere; evaluator overlays this local file so per-problem
timeout = baseline_ms * 1.3 and speedup = local_baseline_ms / variant_ms
are calibrated for this box AND for the variant's worker count.

Per-problem timeout = REBASELINE_TIMEOUT_S (1 hr safety floor). Never cut a
baseline run short — a truncated baseline poisons every variant comparison.

Concurrency = floor(len(core_pool) / W) per worker count. W=1 fills the
pool fully; W=8 typically runs sequentially on small hosts.
"""
import argparse
import importlib.util
import json
import pathlib
import queue as _queue
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "shared"))

from baseline_params import BASELINE  # noqa: E402
from cpsat_runner import run_cpsat  # noqa: E402
from runtime import parallel_solvers, core_range, alloc_core_blocks  # noqa: E402

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


def _load_target_shas(include_stage3=True):
    """Union of stage sample SHAs (dedup, ordered by first appearance).
    include_stage3=False → drop stage3 sample (used for W=1 task list)."""
    if not _STAGE1_SAMPLE.exists():
        print(f"ERROR: {_STAGE1_SAMPLE} missing — run build_samples.py first",
              file=sys.stderr)
        sys.exit(2)
    samples = [
        (_STAGE1_SAMPLE, "stage1"),
        (_STAGE2_SAMPLE, "stage2"),
        (_STAGE4_SAMPLE, "stage4"),
    ]
    if include_stage3:
        samples.insert(2, (_STAGE3_SAMPLE, "stage3"))
    ids = []
    seen = set()
    for sample_path, label in samples:
        if not sample_path.exists():
            print(f"WARN: {sample_path.name} missing — skipping {label}", file=sys.stderr)
            continue
        for sha in json.loads(sample_path.read_text())["sha256"]:
            if sha not in seen:
                ids.append(sha)
                seen.add(sha)
    return ids


def _discover_phase_workers():
    """Union of PHASE_LOCKED['num_search_workers'] across sibling phase dirs.

    Returns sorted list of unique worker counts (defaults to [1] if nothing
    discovered, so the script still produces a usable baseline)."""
    workers = set()
    for prog in sorted(_HERE.glob("phase*_*/initial_program.py")):
        try:
            spec = importlib.util.spec_from_file_location(prog.parent.name, prog)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:
            print(f"WARN: failed to load {prog.relative_to(_HERE)}: {e}",
                  file=sys.stderr)
            continue
        pl = getattr(mod, "PHASE_LOCKED", None)
        if isinstance(pl, dict) and "num_search_workers" in pl:
            try:
                workers.add(int(pl["num_search_workers"]))
            except (TypeError, ValueError):
                pass
    return sorted(workers) if workers else [1]


def _parse_workers_arg(s):
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            w = int(tok)
        except ValueError:
            raise SystemExit(f"--workers: bad integer {tok!r}")
        if w < 1:
            raise SystemExit(f"--workers: {w} must be >= 1")
        if w not in out:
            out.append(w)
    if not out:
        raise SystemExit("--workers: empty list")
    return sorted(out)


def _measure_at_workers(tasks, w, cores):
    """Run baseline for one worker count W across all tasks. Returns
    list of (i, meta, res, core_block) tuples in submission order."""
    blocks = alloc_core_blocks(cores, w)
    if not blocks:
        blocks = [list(cores)] if cores else [None]
    n_parallel = min(len(blocks), len(tasks))
    blocks = blocks[:n_parallel]

    pool = _queue.Queue()
    for b in blocks:
        pool.put(b)

    params = dict(BASELINE)
    params["num_search_workers"] = w

    def _fmt(b):
        if isinstance(b, (list, tuple)):
            return ",".join(str(x) for x in b)
        return str(b) if b is not None else "-"

    print(f"  workers={w}: parallel={n_parallel} blocks={[_fmt(b) for b in blocks]}",
          flush=True)

    def _solve(task):
        i, meta, path = task
        block = pool.get()
        try:
            res = run_cpsat(path, params, REBASELINE_TIMEOUT_S, cpu_core=block)
        finally:
            pool.put(block)
        return i, meta, res, block

    out = []
    if n_parallel == 1:
        for task in tasks:
            out.append(_solve(task))
    else:
        with ThreadPoolExecutor(max_workers=n_parallel) as ex:
            futures = [ex.submit(_solve, t) for t in tasks]
            for fut in as_completed(futures):
                out.append(fut.result())
    out.sort(key=lambda x: x[0])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--workers",
        type=str,
        default=None,
        help="comma-separated worker counts (e.g. '1,8'). Default: union of "
             "PHASE_LOCKED['num_search_workers'] across sibling phase dirs.",
    )
    args = ap.parse_args()

    if args.workers:
        worker_counts = _parse_workers_arg(args.workers)
        print(f"[rebaseline] worker counts (from --workers): {worker_counts}")
    else:
        worker_counts = _discover_phase_workers()
        print(f"[rebaseline] worker counts (auto-discovered): {worker_counts}")

    # Build BOTH task lists: full (W>=2) and stage3-excluded (W=1).
    idx = _load_problem_index()

    def _build_tasks(shas):
        out = []
        for i, sha in enumerate(shas):
            meta = idx.get(sha)
            if meta is None:
                print(f"ERROR: {sha[:12]} not in problems.jsonl",
                      file=sys.stderr)
                sys.exit(2)
            path = _RAW_DIR / meta["problem_filename"]
            if not path.exists():
                print(f"ERROR: input not found: {path}", file=sys.stderr)
                sys.exit(2)
            out.append((i, meta, path))
        return out

    shas_full = _load_target_shas(include_stage3=True)
    shas_no_stage3 = _load_target_shas(include_stage3=False)
    tasks_full = _build_tasks(shas_full)
    tasks_no_stage3 = _build_tasks(shas_no_stage3)
    n_stage3_only = len(shas_full) - len(shas_no_stage3)

    cores = core_range()
    if cores is None:
        cores = list(range(1, parallel_solvers(default=1) + 1))

    print(f"rebaselining stage{{1,2,3,4}}_sample.json: "
          f"{len(tasks_full)} problems total, {n_stage3_only} stage3-only")
    print(f"  W=1 skips stage3 ({len(tasks_no_stage3)} problems)")
    print(f"  W>=2 measures all   ({len(tasks_full)} problems)")
    print(f"  worker counts: {worker_counts}")
    print(f"  per-problem timeout = {REBASELINE_TIMEOUT_S}s (never cut short), "
          f"core pool = {cores}")
    print()

    # results[sha]["by_workers"][str(w)] = {...}
    results = {}
    for meta in (m for _, m, _ in tasks_full):
        results[meta["sha"]] = {
            "raw_result": meta["raw_result"],
            "raw_elapsed_ms": meta["raw_ms"],
            "by_workers": {},
        }

    t_start = time.monotonic()
    mismatch_total = 0
    for w in worker_counts:
        tasks = tasks_no_stage3 if w == 1 else tasks_full
        print(f"[W={w}] {len(tasks)} problems "
              f"({'stage3 skipped' if w == 1 else 'all stages'})", flush=True)
        completed = _measure_at_workers(tasks, w, cores)
        for i, meta, res, block in completed:
            got_result = res.get("result", "Unknown")
            got_ms = int(res.get("elapsed_ms", 0))
            invalid = res.get("invalid_param")
            ok = (got_result == meta["raw_result"]) and not invalid
            if not ok:
                mismatch_total += 1

            if invalid:
                flag = f"  INVALID_PARAM={invalid}"
            elif ok:
                flag = ""
            else:
                flag = "  MISMATCH"
            ratio = got_ms / max(meta["raw_ms"], 1)
            block_str = (",".join(str(x) for x in block)
                         if isinstance(block, (list, tuple)) else str(block))
            print(
                f"  [W={w} {i+1:>2}/{len(tasks)}] {meta['sha'][:10]}  "
                f"raw={meta['raw_result']:<10}/{int(meta['raw_ms']):>7}ms  "
                f"local={got_result:<10}/{got_ms:>7}ms  ratio={ratio:.2f}x{flag}  "
                f"cores={block_str}",
                flush=True,
            )

            entry = {
                "elapsed_ms": got_ms,
                "result": got_result,
                "matches_raw": ok,
                "stats": res.get("stats") or {},
            }
            if "objective" in res:
                entry["objective"] = res["objective"]
            results[meta["sha"]]["by_workers"][str(w)] = entry

    elapsed = time.monotonic() - t_start
    _OUT.write_text(json.dumps(results, indent=2) + "\n")
    n_runs = sum(len(r["by_workers"]) for r in results.values())
    print()
    print(f"wrote {_OUT.relative_to(_BENCH_DIR.parent)} "
          f"({len(results)} entries, {n_runs} (sha,W) runs, "
          f"{mismatch_total} mismatches)")
    print(f"total time: {elapsed:.1f}s")
    if mismatch_total:
        print(f"WARNING: {mismatch_total} (problem, W) pairs had result mismatch — "
              f"evaluator will fall back to raw_ms for those.")
    return 0 if mismatch_total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
