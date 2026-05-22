"""
OpenEvolve evaluator for cpsat-bench parameter tuning.

Score mode: cost (minimize CP-SAT objective; see score.py).

Per-problem timeout = max(MIN_TIMEOUT_S, ceil(baseline_ms * TIMEOUT_FACTOR / 1000)).
Locked params (see baseline_params.LOCKED) must not deviate — violation => combined_score=0.

Environment overrides:
  OPENEVOLVE_MAX_PROBLEMS      cap stage problem count
  OPENEVOLVE_PARALLEL_SOLVERS  concurrent solver subprocesses (default 1)
  OPENEVOLVE_PYTHON_BIN        python for worker subprocess
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
from cpsat_runner import run_cpsat  # noqa: E402
from runtime import (  # noqa: E402
    parallel_solvers, cascade_threshold, core_range, alloc_core_blocks,
)

from openevolve.evaluation_result import EvaluationResult  # noqa: E402

_BENCH_DIR = _HERE.parent.parent
_RAW_DIR = _BENCH_DIR / "raw-data"
_PROBLEMS_JSONL = _BENCH_DIR / "problems.jsonl"
_STAGE1_SAMPLE = _HERE / "stage1_sample.json"
_STAGE2_SAMPLE = _HERE / "stage2_sample.json"
_STAGE3_SAMPLE = _HERE / "stage3_sample.json"
_STAGE4_SAMPLE = _HERE / "stage4_sample.json"
_LOCAL_BASELINE = _HERE / "local_baseline.json"

_PYTHON_BIN = os.environ.get("OPENEVOLVE_PYTHON_BIN")

_KEY_STATS = ("num_branches", "num_conflicts", "num_booleans", "wall_time", "user_time")
_DECISIVE = ("OPTIMAL", "FEASIBLE")


def _load_program(path):
    spec = importlib.util.spec_from_file_location("program", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_problems():
    local = {}
    if _LOCAL_BASELINE.exists():
        local = json.loads(_LOCAL_BASELINE.read_text())
    rows = []
    with open(_PROBLEMS_JSONL) as f:
        for line in f:
            d = json.loads(line)
            sha = d["problem_sha256"]
            baseline_ms = (d.get("cpsat_status") or {}).get("elapsed_ms", 0)
            baseline_result = (d.get("cpsat_status") or {}).get("result")
            baseline_stats = d.get("cpsat_response_stats") or {}
            baseline_objective = (d.get("cpsat_status") or {}).get("objective_value")
            lo = local.get(sha)
            if lo and lo.get("matches_raw"):
                baseline_ms = lo["elapsed_ms"]
                baseline_stats = lo.get("stats") or baseline_stats
                if lo.get("objective") is not None:
                    baseline_objective = lo["objective"]
            rows.append({
                "sha": sha,
                "input_file": d["problem_filename"],
                "baseline_ms": baseline_ms,
                "baseline_result": baseline_result,
                "baseline_stats": baseline_stats,
                "baseline_objective": baseline_objective,
            })
    return rows


def _filter_stage1(problems):
    if not _STAGE1_SAMPLE.exists():
        return problems
    keep = set(json.loads(_STAGE1_SAMPLE.read_text())["sha256"])
    return [p for p in problems if p["sha"] in keep]


def _filter_stage2(problems):
    if not _STAGE2_SAMPLE.exists():
        return problems
    keep = set(json.loads(_STAGE2_SAMPLE.read_text())["sha256"])
    return [p for p in problems if p["sha"] in keep]


def _filter_stage3(problems):
    if not _STAGE3_SAMPLE.exists():
        return problems
    keep = set(json.loads(_STAGE3_SAMPLE.read_text())["sha256"])
    return [p for p in problems if p["sha"] in keep]


def _filter_stage4(problems):
    if not _STAGE4_SAMPLE.exists():
        return problems
    keep = set(json.loads(_STAGE4_SAMPLE.read_text())["sha256"])
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
            {"error_type": type(e).__name__, "error_message": str(e),
             "full_traceback": traceback.format_exc()[-2000:], "stage": stage_name},
        )

    if not hasattr(program, "get_params"):
        return _err_result(
            {"error": "missing get_params()"},
            {"suggestion": "initial_program.py must expose get_params() -> dict",
             "stage": stage_name},
        )

    try:
        params = program.get_params()
        if not isinstance(params, dict):
            raise TypeError(f"get_params() returned {type(params).__name__}, expected dict")
    except Exception as e:
        return _err_result(
            {"error": f"get_params() raised: {e}"},
            {"error_type": type(e).__name__, "error_message": str(e),
             "full_traceback": traceback.format_exc()[-2000:], "stage": stage_name},
        )

    # Per-phase lock (worker count varies by phase) overrides the global lock.
    phase_locked = getattr(program, "PHASE_LOCKED", None)
    locked = phase_locked if isinstance(phase_locked, dict) else LOCKED
    violations = {k: params.get(k) for k in locked if params.get(k) != locked[k]}
    if violations:
        return _err_result(
            {"error": "locked params violated"},
            {"locked_violated": violations, "locked_expected": locked,
             "stage": stage_name,
             "suggestion": "Do not modify locked keys (see PHASE_LOCKED in "
                           "this phase's initial_program.py)."},
        )

    if "OPENEVOLVE_MAX_PROBLEMS" in os.environ:
        problems = problems[: int(os.environ["OPENEVOLVE_MAX_PROBLEMS"])]

    for p in problems:
        input_path = _RAW_DIR / p["input_file"]
        if not input_path.exists():
            return _err_result(
                {"error": f"input not found: {p['input_file']}"},
                {"missing_file": str(input_path), "stage": stage_name},
            )

    import queue as _queue
    # Core pool: OPENEVOLVE_CORE_RANGE (e.g. "2-7") overrides; else
    # cores 1..parallel_solvers() (core 0 reserved for kernel housekeeping).
    #
    # When params["num_search_workers"] = W > 1, each variant solve needs W
    # cores. We floor-chunk the core list into W-sized blocks; concurrency =
    # number of blocks. Leftover cores at the tail are dropped to keep every
    # solve's CPU budget identical (comparable timings).
    _cores = core_range()
    if _cores is None:
        _cores = list(range(1, parallel_solvers(default=1) + 1))
    workers_per_solve = int(params.get("num_search_workers", 1) or 1)
    _blocks = alloc_core_blocks(_cores, workers_per_solve)
    if not _blocks:
        # Not enough cores for even one block at workers_per_solve. Fall back
        # to a single block of all available cores (still respect taskset pin).
        _blocks = [list(_cores)] if _cores else [None]
    n_parallel = min(len(_blocks), len(problems))
    _blocks = _blocks[:n_parallel]
    _core_pool = _queue.Queue()
    for _b in _blocks:
        _core_pool.put(_b)

    def _solve(idx_p):
        idx, p = idx_p
        input_path = _RAW_DIR / p["input_file"]
        timeout_s = max(MIN_TIMEOUT_S, math.ceil(p["baseline_ms"] * TIMEOUT_FACTOR / 1000))
        core = _core_pool.get()
        try:
            r = run_cpsat(input_path, params, timeout_s, python_bin=_PYTHON_BIN, cpu_core=core)
        finally:
            _core_pool.put(core)
        return idx, p, r, core, timeout_s

    def _is_regression(p, r):
        # cost mode: baseline OPTIMAL → variant must reach OPTIMAL or FEASIBLE.
        # bailing to UNKNOWN/INFEASIBLE/MODEL_INVALID on an OPTIMAL baseline = regression.
        if "invalid_param" in r:
            return False
        if p["baseline_result"] not in _DECISIVE:
            return False
        return r.get("result") not in _DECISIVE

    def _invalid_err(r):
        return _err_result(
            {"error": f"invalid param: {r['invalid_param']}"},
            {"invalid_param": r["invalid_param"], "stderr": r.get("stderr", "")[:2000],
             "stage": stage_name,
             "suggestion": "Remove or fix this key in get_params()."},
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
                "suggestion": "Variant lost feasibility on a problem baseline reached optimal. "
                              "Revert params that disable a needed subsolver or shorten search.",
            },
        )

    def _fmt_core(c):
        if c is None:
            return "-"
        if isinstance(c, (list, tuple)):
            return ",".join(str(x) for x in c) if c else "-"
        return str(c)

    print(f"  [{stage_name}] workers/solve={workers_per_solve} "
          f"core_blocks={[_fmt_core(b) for b in _blocks]}", flush=True)

    by_idx = {}
    abort = None
    if n_parallel == 1:
        for pair in enumerate(problems):
            idx, p, r, core, timeout_s = _solve(pair)
            print(f"  [{stage_name}] {idx+1}/{len(problems)} {p['sha'][:10]} "
                  f"{r.get('result')} {r.get('elapsed_ms')}ms / {timeout_s}s "
                  f"(core={_fmt_core(core)})", flush=True)
            if "invalid_param" in r:
                return _invalid_err(r)
            if _is_regression(p, r):
                print(f"  [{stage_name}] regression — aborting remaining "
                      f"{len(problems) - idx - 1}", flush=True)
                return _regression_err(p, r)
            by_idx[idx] = (p, r)
    else:
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
                      f"(core={_fmt_core(core)})", flush=True)
                if "invalid_param" in r:
                    abort = ("invalid", p, r)
                elif _is_regression(p, r):
                    abort = ("regression", p, r)
                if abort is not None:
                    print(f"  [{stage_name}] {abort[0]} — cancelling pending "
                          f"(in-flight workers will drain)", flush=True)
                    for f in futures:
                        f.cancel()
                    continue
                by_idx[idx] = (p, r)
        if abort is not None:
            kind, p, r = abort
            return _invalid_err(r) if kind == "invalid" else _regression_err(p, r)

    results = []
    for idx in range(len(problems)):
        p, r = by_idx[idx]
        rec = {
            **p,
            "result": r["result"],
            "elapsed_ms": r["elapsed_ms"],
            "timeout": bool(r.get("timeout")),
            "stats": r.get("stats") or {},
        }
        if "objective" in r:
            rec["objective"] = r["objective"]
        results.append(rec)

    metrics = score(results)
    metrics["stage"] = stage_name

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
            "base_obj": r.get("baseline_objective"),
            "obj": r.get("objective"),
            "stats": {k: r["stats"].get(k, 0) for k in _KEY_STATS if k in r["stats"]},
            "base_stats": {k: r["baseline_stats"].get(k, 0)
                           for k in _KEY_STATS if k in r.get("baseline_stats", {})},
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
    return _evaluate(program_path, _filter_stage1(_load_problems()), "stage1")


def evaluate_stage2(program_path):
    return _evaluate(program_path, _filter_stage2(_load_problems()), "stage2")


def evaluate_stage3(program_path):
    # openevolve cascade hardcodes 3 stages, so user-stage4 (broad runtime
    # sample) is chained inside stage3 via the runtime cascade_threshold gate.
    problems3 = _filter_stage3(_load_problems())
    r3 = _evaluate(program_path, problems3, "stage3")
    if not isinstance(r3, EvaluationResult):
        return r3
    gate = cascade_threshold(2, default=1.03)
    if r3.metrics.get("combined_score", 0.0) < gate:
        return r3
    problems4 = _filter_stage4(_load_problems())
    r4 = _evaluate(program_path, problems4, "stage4")
    if not isinstance(r4, EvaluationResult):
        return r4
    merged_metrics = {**r3.metrics, **r4.metrics}
    merged_artifacts = {**r3.artifacts, **r4.artifacts}
    return EvaluationResult(metrics=merged_metrics, artifacts=merged_artifacts)


def evaluate_stage4(program_path):
    # Standalone entry for manual / final-verify use. Not invoked by cascade.
    problems = _filter_stage4(_load_problems())
    return _evaluate(program_path, problems, "stage4")


def evaluate(program_path):
    # Evolution loop entry: stage1 only. Cascade chains 2/3/4.
    return evaluate_stage1(program_path)
