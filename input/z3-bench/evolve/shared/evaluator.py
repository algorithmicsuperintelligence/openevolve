"""
OpenEvolve evaluator for Z3 parameter tuning.

Sample selection (built by build_samples.py):
  stage1: stage1_sample.json (5 SAT, quintile-spread by problem size)
  stage2: stage2_sample.json (50 SAT+UNSAT, quintile-spread by problem size)

Per-problem timeout = max(MIN_TIMEOUT_S, ceil(baseline_ms * TIMEOUT_FACTOR / 1000)).
Adaptive: small problems get tight cap, huge problems get proportional headroom.
Constants TIMEOUT_FACTOR=1.3, MIN_TIMEOUT_S=5 (z3 startup overhead floor).
Baseline_ms reflects local rebaseline (rebaseline_local.py) when SHA is in
local_baseline.json, else raw_ms from problems.jsonl.

Score: geomean(speedup) * solved_rate^2 * efficiency^OPENEVOLVE_STATS_WEIGHT.
       efficiency = geomean over {decisions, propagations, conflicts, mk clause}
       of (baseline_stat / variant_stat) across solved problems. Default
       OPENEVOLVE_STATS_WEIGHT=0 keeps prior behaviour. Per-problem solver
       stats (z3 Optimize.statistics) are always surfaced via metrics +
       artifacts so the LLM sees solver-internal cost beyond wall-clock.

Locked params (sat.random_seed / smt.random_seed / sls.random_seed / parallel.enable)
must not deviate from baseline_params.LOCKED — violation => combined_score=0.

Environment overrides:
  OPENEVOLVE_MAX_PROBLEMS      int — cap stage2 problem count
  OPENEVOLVE_PARALLEL_SOLVERS  int — SINGLE knob for total z3 concurrency
                               (default 1). Spawns N z3 worker subprocesses
                               concurrently per stage; each pinned to a
                               dedicated core via taskset (Linux). Capped at
                               len(problems). Config.yaml parallel_evaluations
                               is fixed at 1 so total concurrent z3 = this env
                               value (no parallel_evaluations × N blow-up).
                               Total RAM ≈ N × 4 GB worst-case.
  OPENEVOLVE_Z3_BIN            str — z3 binary path (default "z3")
"""
import importlib.util
import json
import math
import os
import pathlib
import sys
import traceback

TIMEOUT_FACTOR = 1.3
MIN_TIMEOUT_S = 5

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from baseline_params import BASELINE, LOCKED  # noqa: E402
from score import score  # noqa: E402
from z3_runner import run_z3  # noqa: E402
from runtime import parallel_solvers  # noqa: E402

from openevolve.evaluation_result import EvaluationResult  # noqa: E402

_BENCH_DIR = _HERE.parent.parent          # input/z3-bench/
_RAW_DIR = _BENCH_DIR / "raw-data"
_PROBLEMS_JSONL = _BENCH_DIR / "problems.jsonl"
_STAGE1_SAMPLE = _HERE / "stage1_sample.json"
_STAGE2_SAMPLE = _HERE / "stage2_sample.json"
_STAGE3_SAMPLE = _HERE / "stage3_sample.json"
_LOCAL_BASELINE = _HERE / "local_baseline.json"

_PYTHON_BIN = os.environ.get("OPENEVOLVE_PYTHON_BIN")  # None -> sys.executable


def _load_program(path):
    spec = importlib.util.spec_from_file_location("program", path)
    module = importlib.util.module_from_spec(spec)
    # Phase initial_programs add shared/ to sys.path themselves.
    spec.loader.exec_module(module)
    return module


def _load_problems():
    # Always returns the full problems.jsonl set (50 problems).
    # local_baseline.json (if present) overlays per-SHA local timing so that
    # speedup uses on-machine wall-clock for rebaselined SHAs; non-rebaselined
    # SHAs fall back to raw_ms recorded in problems.jsonl. Local elapsed_ms is
    # only trusted when local result matches raw — otherwise raw_ms wins to
    # avoid speedup distortion from a bad local run (timeout, mismatch).
    local = {}
    if _LOCAL_BASELINE.exists():
        local = json.loads(_LOCAL_BASELINE.read_text())

    rows = []
    with open(_PROBLEMS_JSONL) as f:
        for line in f:
            d = json.loads(line)
            sha = d["problem_sha256"]
            baseline_ms = d["z3_status"]["elapsed_ms"]
            baseline_result = d["z3_status"]["result"]
            baseline_stats = {}
            lo = local.get(sha)
            if lo and lo.get("matches_raw"):
                baseline_ms = lo["elapsed_ms"]
                baseline_stats = lo.get("stats") or {}
            rows.append(
                {
                    "sha": sha,
                    "smt2": d["smt2_filename"],
                    "baseline_ms": baseline_ms,
                    "baseline_result": baseline_result,
                    "baseline_stats": baseline_stats,
                }
            )
    return rows


def _filter_stage1(problems):
    if not _STAGE1_SAMPLE.exists():
        return problems[:5]
    keep = set(json.loads(_STAGE1_SAMPLE.read_text())["sha256"])
    return [p for p in problems if p["sha"] in keep]


def _filter_stage2(problems):
    # stage2_sample.json: 5 SAT problems from the runtime upper-half
    # (medium-slow). Catches regressions on harder instances.
    if not _STAGE2_SAMPLE.exists():
        return problems
    keep = set(json.loads(_STAGE2_SAMPLE.read_text())["sha256"])
    return [p for p in problems if p["sha"] in keep]


def _filter_stage3(problems):
    # stage3_sample.json: 50 SAT+UNSAT problems, size-stratified broad
    # coverage. raw-data may exceed 50; without this filter, stage3 cost
    # grows unbounded as raw-data accumulates.
    if not _STAGE3_SAMPLE.exists():
        return problems
    keep = set(json.loads(_STAGE3_SAMPLE.read_text())["sha256"])
    return [p for p in problems if p["sha"] in keep]


def _err_result(metrics_extra, artifacts):
    metrics = {
        "combined_score": 0.0,
        "geomean_speedup": 0.0,
        "solved_rate": 0.0,
        "regressions": 0,
        "solved": 0,
        "total": 0,
    }
    metrics.update(metrics_extra)
    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def _evaluate(program_path, problems, stage_name):
    try:
        program = _load_program(program_path)
    except Exception as e:
        return _err_result(
            {"error": f"program load failed: {e}"},
            {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "full_traceback": traceback.format_exc()[-2000:],
                "stage": stage_name,
            },
        )

    if not hasattr(program, "get_params"):
        return _err_result(
            {"error": "missing get_params()"},
            {"suggestion": "initial_program.py must expose get_params() -> dict", "stage": stage_name},
        )

    try:
        params = program.get_params()
        if not isinstance(params, dict):
            raise TypeError(f"get_params() returned {type(params).__name__}, expected dict")
    except Exception as e:
        return _err_result(
            {"error": f"get_params() raised: {e}"},
            {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "full_traceback": traceback.format_exc()[-2000:],
                "stage": stage_name,
            },
        )

    violations = {k: params.get(k) for k in LOCKED if params.get(k) != LOCKED[k]}
    if violations:
        return _err_result(
            {"error": "locked params violated"},
            {
                "locked_violated": violations,
                "locked_expected": LOCKED,
                "stage": stage_name,
                "suggestion": "Do not modify sat.random_seed, smt.random_seed, sls.random_seed, parallel.enable",
            },
        )

    if "OPENEVOLVE_MAX_PROBLEMS" in os.environ:
        problems = problems[: int(os.environ["OPENEVOLVE_MAX_PROBLEMS"])]

    # Verify smt2 files exist up front so parallel dispatch doesn't race.
    for p in problems:
        smt2_path = _RAW_DIR / p["smt2"]
        if not smt2_path.exists():
            return _err_result(
                {"error": f"smt2 not found: {p['smt2']}"},
                {"missing_file": str(smt2_path), "stage": stage_name},
            )

    # Parallel dispatch — `OPENEVOLVE_PARALLEL_SOLVERS` controls how many
    # z3 worker subprocesses run concurrently for the stage's problem list.
    # Worker count is capped at len(problems) (no point spawning idle threads).
    # Cores are leased from a queue.Queue so each in-flight task holds a
    # unique core slot. Correct even when len(problems) > n_parallel
    # (idx % n_parallel would collide across workers).
    # Core pool = cores 1..n_parallel — core 0 reserved for kernel interrupts
    # / housekeeping (avoids tail-latency spikes). Serial mode also leases
    # core 1, symmetric with parallel so baseline / variant share the same
    # pin envelope (no unpinned-vs-pinned bias).
    import queue as _queue
    n_parallel = min(parallel_solvers(default=1), len(problems))
    _core_pool = _queue.Queue()
    for _c in range(1, n_parallel + 1):
        _core_pool.put(_c)

    def _solve(idx_p):
        idx, p = idx_p
        smt2_path = _RAW_DIR / p["smt2"]
        timeout_s = max(MIN_TIMEOUT_S, math.ceil(p["baseline_ms"] * TIMEOUT_FACTOR / 1000))
        core = _core_pool.get()
        try:
            r = run_z3(smt2_path, params, timeout_s,
                       python_bin=_PYTHON_BIN, cpu_core=core)
        finally:
            _core_pool.put(core)
        return idx, p, r, core, timeout_s

    def _is_regression(p, r):
        # Correctness regression: baseline gave a definitive answer (Sat/Unsat)
        # and variant disagrees (Unsat, Unknown, or timeout). Unknown baseline
        # is excluded — variant solving an Unknown is an improvement, not a
        # regression. invalid_param handled separately (param-level error).
        if "invalid_param" in r:
            return False
        return (
            p["baseline_result"] in ("Sat", "Unsat")
            and r.get("result") != p["baseline_result"]
        )

    def _invalid_err(r):
        return _err_result(
            {"error": f"invalid z3 param: {r['invalid_param']}"},
            {
                "invalid_param": r["invalid_param"],
                "stderr": r.get("stderr", "")[:2000],
                "stage": stage_name,
                "suggestion": "Remove or fix this key in get_params().",
            },
        )

    def _regression_err(p, r):
        return _err_result(
            {"error": f"result regression on {p['sha'][:10]}: "
                      f"baseline={p['baseline_result']} got={r.get('result')}"},
            {
                "result_mismatch": {
                    "sha": p["sha"][:12],
                    "baseline_result": p["baseline_result"],
                    "got_result": r.get("result"),
                    "elapsed_ms": r.get("elapsed_ms"),
                    "timeout": bool(r.get("timeout")),
                },
                "stage": stage_name,
                "suggestion": (
                    "Variant lost correctness on a problem baseline solved. "
                    "Revert params that disable preprocessing or relax completeness "
                    "(e.g. simplifier off, aggressive sls early-prune, restart "
                    "tuning that starves Sat search)."
                ),
            },
        )

    by_idx = {}
    abort = None  # ("invalid"|"regression", p, r) — first failure observed
    if n_parallel == 1:
        # Sequential: short-circuit immediately on invalid or regression.
        for pair in enumerate(problems):
            idx, p, r, core, timeout_s = _solve(pair)
            print(f"  [{stage_name}] {idx+1}/{len(problems)} {p['sha'][:10]} "
                  f"{r.get('result')} {r.get('elapsed_ms')}ms / {timeout_s}s "
                  f"(core={core})", flush=True)
            if "invalid_param" in r:
                return _invalid_err(r)
            if _is_regression(p, r):
                print(f"  [{stage_name}] regression — aborting remaining "
                      f"{len(problems) - idx - 1} problems", flush=True)
                return _regression_err(p, r)
            by_idx[idx] = (p, r)
    else:
        # Parallel: LPT (longest-processing-time) submission — sort problems
        # by baseline_ms descending so big jobs dispatch first. ThreadPool's
        # internal FIFO queue then drains small jobs onto whichever worker
        # frees up, minimising tail idle time when n_parallel < len(problems).
        # Cancel pending futures on first failure; in-flight tasks keep
        # running until subprocess timeout (with __exit__ waits for them).
        # Per-problem timeout is baseline_ms * 1.3 (adaptive), so worst-case
        # drain depends on the slowest in-flight problem rather than a fixed cap.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        ordered = sorted(enumerate(problems), key=lambda ip: -ip[1]["baseline_ms"])
        with ThreadPoolExecutor(max_workers=n_parallel) as ex:
            futures = [ex.submit(_solve, pair) for pair in ordered]
            for fut in as_completed(futures):
                if abort is not None:
                    continue
                idx, p, r, core, timeout_s = fut.result()
                print(f"  [{stage_name}] {idx+1}/{len(problems)} {p['sha'][:10]} "
                      f"{r.get('result')} {r.get('elapsed_ms')}ms / {timeout_s}s "
                      f"(core={core})", flush=True)
                if "invalid_param" in r:
                    abort = ("invalid", p, r)
                elif _is_regression(p, r):
                    abort = ("regression", p, r)
                if abort is not None:
                    print(f"  [{stage_name}] {abort[0]} — cancelling pending "
                          f"problems (in-flight workers will drain)", flush=True)
                    for f in futures:
                        f.cancel()
                    continue
                by_idx[idx] = (p, r)
        if abort is not None:
            kind, p, r = abort
            return _invalid_err(r) if kind == "invalid" else _regression_err(p, r)

    # No failure — reassemble in original problem order.
    results = []
    for idx in range(len(problems)):
        p, r = by_idx[idx]
        results.append(
            {
                **p,
                "result": r["result"],
                "elapsed_ms": r["elapsed_ms"],
                "timeout": bool(r.get("timeout")),
                "stats": r.get("stats") or {},
            }
        )

    metrics = score(results)
    metrics["stage"] = stage_name

    # Surface a small fixed set of solver-internal counters that drive Z3's
    # search shape. LLM sees these via metrics + per-problem artifacts and can
    # reason about what a param tweak did beyond wall-clock (e.g. fewer
    # decisions/conflicts at same elapsed_ms = sturdier improvement).
    # `mk clause` chosen over `restarts` since z3 does not emit `restarts`
    # for the optimize / arith-heavy stack used here.
    _KEY_STATS = ("decisions", "propagations", "conflicts", "mk clause")
    for k in _KEY_STATS:
        metrics[f"total_{k}"] = float(sum(r["stats"].get(k, 0) for r in results))

    sample = [
        {
            "sha": r["sha"][:10],
            "base_result": r["baseline_result"],
            "got_result": r["result"],
            "base_ms": r["baseline_ms"],
            "ms": r["elapsed_ms"],
            "speedup": round(r["baseline_ms"] / max(r["elapsed_ms"], 1), 3),
            "timeout": r["timeout"],
            "stats": {k: r["stats"].get(k, 0) for k in _KEY_STATS if k in r["stats"]},
            "base_stats": {k: r["baseline_stats"].get(k, 0) for k in _KEY_STATS if k in r.get("baseline_stats", {})},
        }
        for r in results
    ]
    artifacts = {
        "stage": stage_name,
        "summary": (
            f"solved={metrics['solved']}/{metrics['total']} "
            f"regressions={metrics['regressions']} "
            f"geomean_speedup={metrics['geomean_speedup']:.3f} "
            f"efficiency={metrics.get('efficiency', 1.0):.3f} "
            f"score={metrics['combined_score']:.3f}"
        ),
        "per_problem": sample[:20],
    }
    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def evaluate_stage1(program_path):
    # Per-problem timeout = baseline_ms * TIMEOUT_FACTOR (computed in _solve).
    # Stage1 wall-clock budget ≈ TIMEOUT_FACTOR * sum(baseline_ms) over 5 sample
    # problems (parallel mode divides by n_parallel).
    problems = _filter_stage1(_load_problems())
    return _evaluate(program_path, problems, "stage1")


def evaluate_stage2(program_path):
    problems = _filter_stage2(_load_problems())
    return _evaluate(program_path, problems, "stage2")


def evaluate_stage3(program_path):
    problems = _filter_stage3(_load_problems())
    return _evaluate(program_path, problems, "stage3")


def evaluate(program_path):
    # Evolution uses stage1 (5 sampled problems) only — fast iteration loop.
    # Stage2 (full 50 problems) is reserved for final verification via
    # final_verify.py on the best-program, not the per-variant search loop.
    return evaluate_stage1(program_path)
