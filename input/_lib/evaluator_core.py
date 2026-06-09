"""
Unified evaluator. Lifted from cpsat-bench's evaluator.py and parameterized
over the per-bench adapter + catalog + config.

Per-bench `_lib/evaluator_entry.py` is what openevolve loads; it just calls
`build_evaluators(bench_root)` (reading `OPENEVOLVE_BENCH_ROOT` from env)
and re-exports `evaluate_stage1..4` + `evaluate`.

Per-problem param resolution:
  If the program exposes `get_params(problem=..., stage=...)`, the evaluator
  calls it once per problem so phases can ship SIZE_BUCKETS
  (constraint-count conditional overrides) and STAGE3_OVERRIDES (outlier-only
  tuning) inside their EVOLVE-BLOCK. Programs that only expose `get_params()`
  keep working.

10-run averaging is the standard (configurable via
`bench.evaluation.repeats`).
"""

import importlib.util
import inspect
import json
import logging
import math
import os
import pathlib
import queue as _queue
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from _lib import averaging, bench_paths, params_catalog, runtime, scorer, subprocess_runner

try:
    from openevolve.evaluation_result import EvaluationResult
except Exception:

    class EvaluationResult:
        def __init__(self, metrics=None, artifacts=None):
            self.metrics = metrics or {}
            self.artifacts = artifacts or {}


logger = logging.getLogger(__name__)


def _load_program(path):
    spec = importlib.util.spec_from_file_location("program", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _supports_kwargs(fn, *names):
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return any(n in params for n in names)


def _resolve_params(get_params, problem, stage):
    kwargs = {}
    if _supports_kwargs(get_params, "problem"):
        kwargs["problem"] = problem
    if _supports_kwargs(get_params, "stage"):
        kwargs["stage"] = stage
    out = get_params(**kwargs) if kwargs else get_params()
    if not isinstance(out, dict):
        raise TypeError(f"get_params() returned {type(out).__name__}, expected dict")
    return out


def _pick_local_baseline(entry, workers):
    """Return inner per-W dict, or None when missing."""
    if not isinstance(entry, dict):
        return None
    bw = entry.get("by_workers")
    if isinstance(bw, dict) and bw:
        key = str(int(workers))
        return bw.get(key)
    if "elapsed_ms" in entry and int(workers) == 1:
        return entry
    return None


def build_evaluators(bench_root):
    bench_root = pathlib.Path(bench_root).resolve()
    bench_paths.add_input_to_sys_path()

    adapter = bench_paths.load_adapter(bench_root)
    catalog = params_catalog.load_for_bench(bench_root)
    cfg = bench_paths.load_config(bench_root)
    eval_cfg = (cfg.get("bench") or {}).get("evaluation") or {}

    raw_dir = bench_paths.raw_dir(bench_root)
    problems_jsonl = bench_paths.problems_jsonl(bench_root)
    cache = bench_paths.cache_dir(bench_root)
    worker = bench_paths.worker_path(bench_root)
    config_yaml = bench_paths.config_path(bench_root)

    decisive = tuple(adapter.DECISIVE_RESULTS)
    decided = tuple(getattr(adapter, "DECIDED_RESULTS", adapter.DECISIVE_RESULTS))
    key_stats = tuple(adapter.KEY_STATS)
    stats_weights = dict(getattr(adapter, "STATS_WEIGHTS", {}))
    score_mode = eval_cfg.get("score_mode", getattr(adapter, "SCORE_MODE", "speedup"))
    if bench_paths.solver_mode(bench_root) == "optimize" and score_mode != "cost":
        print(
            f"[warn] solver_mode=optimize but score_mode={score_mode} — "
            f"objective guard inactive; score_mode: cost is recommended for "
            f"optimize mode.",
            flush=True,
        )
    time_metric = eval_cfg.get("time_metric")
    cost_weight = eval_cfg.get("cost_weight")
    repeats_default = int(eval_cfg.get("repeats", 10))
    timeout_factor = float(eval_cfg.get("timeout_factor", 1.3))
    min_timeout_s = int(eval_cfg.get("min_timeout_s", 5))
    workers_key = getattr(adapter, "WORKERS_KEY", None)
    catalog_locked = catalog.locked

    py_bin = os.environ.get("OPENEVOLVE_PYTHON_BIN")

    # Outlier set (cpsat opt-in). cache/outliers.json shape:
    # {"outliers": {sha: {...}}, "stage3_sample": [...]}
    enable_outliers = bool(eval_cfg.get("enable_outlier_stage", False))
    outlier_shas = set()
    if enable_outliers:
        path = cache / "outliers.json"
        if path.exists():
            try:
                blob = json.loads(path.read_text())
                outlier_shas = set((blob.get("outliers") or {}).keys())
            except (json.JSONDecodeError, OSError):
                pass

    # Small/large cache for cpsat-style cascade merging (stage1+stage2 → final).
    # Keyed by program_path so multiple cascades within one process don't leak.
    _SMALL_CACHE: dict = {}

    def _solve_repeats():
        env = os.environ.get("OPENEVOLVE_SOLVE_REPEATS")
        if env:
            try:
                return max(1, int(env))
            except ValueError:
                pass
        return repeats_default

    def _load_problems(workers=1):
        local = {}
        path = cache / "local_baseline.json"
        if path.exists():
            try:
                local = json.loads(path.read_text())
            except json.JSONDecodeError:
                local = {}
        rows = []
        with open(problems_jsonl) as f:
            for line in f:
                d = json.loads(line)
                sha = d["problem_sha256"]
                status = d.get(adapter.STATUS_FIELD) or {}
                stats_field = getattr(adapter, "STATS_FIELD", None)
                baseline_stats = (d.get(stats_field) or {}) if stats_field else {}
                features = d.get(getattr(adapter, "FEATURES_FIELD", "features")) or {}
                obj_field = getattr(adapter, "OBJECTIVE_FIELD", None)
                baseline_objective = status.get(obj_field) if obj_field else None
                baseline_ms = status.get("elapsed_ms", 0)
                baseline_result = status.get("result")

                lo = _pick_local_baseline(local.get(sha), workers)
                if lo and lo.get("matches_raw"):
                    baseline_ms = lo["elapsed_ms"]
                    baseline_stats = lo.get("stats") or baseline_stats
                    if lo.get("objective") is not None:
                        baseline_objective = lo["objective"]
                rows.append(
                    {
                        "sha": sha,
                        "input_file": d[adapter.PROBLEM_FILE_FIELD],
                        "baseline_ms": baseline_ms,
                        "baseline_result": baseline_result,
                        "baseline_stats": baseline_stats,
                        "baseline_objective": baseline_objective,
                        "size": adapter.get_problem_size(features),
                        "is_outlier": sha in outlier_shas,
                        "features": features,
                    }
                )
        return rows

    def _filter_by_sample(problems, stage_n):
        path = cache / f"stage{stage_n}_sample.json"
        if not path.exists():
            return problems
        keep = set(json.loads(path.read_text())["sha256"])
        return [p for p in problems if p["sha"] in keep]

    def _err(metrics_extra, artifacts):
        m = {
            "combined_score": 0.0,
            "geomean_speedup": 0.0,
            "solved_rate": 0.0,
            "regressions": 0,
            "solved": 0,
            "total": 0,
        }
        m.update(metrics_extra)
        return EvaluationResult(metrics=m, artifacts=artifacts)

    def _peek_workers(program_path):
        if not workers_key:
            return 1
        try:
            program = _load_program(program_path)
        except Exception:
            return 1
        pl = getattr(program, "PHASE_LOCKED", None)
        if isinstance(pl, dict) and workers_key in pl:
            try:
                return int(pl[workers_key])
            except (TypeError, ValueError):
                pass
        try:
            return int(program.get_params().get(workers_key, 1) or 1)
        except Exception:
            return 1

    def _evaluate(program_path, problems, stage_name):
        try:
            program = _load_program(program_path)
        except Exception as e:
            return _err(
                {"error": f"program load failed: {e}"},
                {
                    "error_type": type(e).__name__,
                    "full_traceback": traceback.format_exc()[-2000:],
                    "stage": stage_name,
                },
            )
        if not hasattr(program, "get_params"):
            return _err({"error": "missing get_params()"}, {"stage": stage_name})

        try:
            global_params = _resolve_params(program.get_params, None, stage_name)
        except Exception as e:
            return _err(
                {"error": f"get_params() raised: {e}"},
                {
                    "error_type": type(e).__name__,
                    "full_traceback": traceback.format_exc()[-2000:],
                    "stage": stage_name,
                },
            )

        phase_locked = getattr(program, "PHASE_LOCKED", None)
        if isinstance(phase_locked, dict):
            locked = dict(catalog_locked)
            locked.update(phase_locked)
        else:
            locked = dict(catalog_locked)

        violations = {k: global_params.get(k) for k in locked if global_params.get(k) != locked[k]}
        if violations:
            return _err(
                {"error": "locked params violated"},
                {"locked_violated": violations, "locked_expected": locked, "stage": stage_name},
            )

        catalog_errors = catalog.validate(global_params)
        if catalog_errors:
            first_key, first_msg = catalog_errors[0]
            return _err(
                {"error": f"invalid param: {first_key}"},
                {
                    "invalid_param": first_key,
                    "catalog_errors": catalog_errors[:10],
                    "stage": stage_name,
                },
            )

        if "OPENEVOLVE_MAX_PROBLEMS" in os.environ:
            problems = problems[: int(os.environ["OPENEVOLVE_MAX_PROBLEMS"])]

        if not problems:
            return EvaluationResult(
                metrics={
                    "combined_score": 100.0,
                    "geomean_speedup": 100.0,
                    "solved_rate": 1.0,
                    "regressions": 0,
                    "solved": 0,
                    "total": 0,
                    "stage": stage_name,
                },
                artifacts={"stage": stage_name, "summary": "empty sample — pass-through"},
            )

        for p in problems:
            ip = raw_dir / p["input_file"]
            if not ip.exists():
                return _err(
                    {"error": f"input not found: {p['input_file']}"},
                    {"missing_file": str(ip), "stage": stage_name},
                )

        cores = runtime.core_range()
        if cores is None:
            cores = list(range(1, runtime.parallel_solvers(config_yaml, default=1) + 1))
        if workers_key:
            workers_per_solve = int(global_params.get(workers_key, 1) or 1)
            blocks = runtime.alloc_core_blocks(cores, workers_per_solve)
            if not blocks:
                blocks = [list(cores)] if cores else [None]
        else:
            workers_per_solve = 1
            blocks = [[c] for c in cores] if cores else [None]

        n_parallel = min(len(blocks), len(problems))
        blocks = blocks[:n_parallel]
        core_pool = _queue.Queue()
        for b in blocks:
            core_pool.put(b)

        n_repeats = _solve_repeats()

        def _solve(idx_p):
            idx, p = idx_p
            ip = raw_dir / p["input_file"]
            timeout_s = max(min_timeout_s, math.ceil(p["baseline_ms"] * timeout_factor / 1000))
            try:
                per_params = _resolve_params(program.get_params, p, stage_name)
            except Exception as e:
                return (
                    idx,
                    p,
                    {
                        "result": "ERROR",
                        "elapsed_ms": 0,
                        "invalid_param": f"get_params(problem,stage) raised: {e}",
                    },
                    None,
                    timeout_s,
                )
            for k, v in locked.items():
                per_params[k] = v
            if workers_key and workers_key in global_params:
                per_params[workers_key] = global_params[workers_key]

            block = core_pool.get()
            try:
                runs = []
                for _ in range(n_repeats):
                    rr = subprocess_runner.run_solver(
                        worker_path=worker,
                        problem_path=ip,
                        params=per_params,
                        timeout_s=timeout_s,
                        python_bin=py_bin,
                        cpu_core=block,
                    )
                    runs.append(rr)
                    if "invalid_param" in rr:
                        break
                avg = averaging.average_runs(runs)
            finally:
                core_pool.put(block)
            return idx, p, avg, block, timeout_s

        def _fmt(c):
            if c is None:
                return "-"
            if isinstance(c, (list, tuple)):
                return ",".join(str(x) for x in c) if c else "-"
            return str(c)

        print(
            f"  [{stage_name}] workers/solve={workers_per_solve} "
            f"repeats={n_repeats} "
            f"core_blocks={[_fmt(b) for b in blocks]}",
            flush=True,
        )

        by_idx = {}
        abort = None
        if n_parallel == 1:
            for pair in enumerate(problems):
                idx, p, r, block, timeout_s = _solve(pair)
                print(
                    f"  [{stage_name}] {idx+1}/{len(problems)} {p['sha'][:10]} "
                    f"{r.get('result')} {r.get('elapsed_ms')}ms / {timeout_s}s "
                    f"(core={_fmt(block)})",
                    flush=True,
                )
                if "invalid_param" in r:
                    return _err(
                        {"error": f"invalid param: {r['invalid_param']}"},
                        {
                            "invalid_param": r["invalid_param"],
                            "stderr": r.get("stderr", "")[:2000],
                            "stage": stage_name,
                        },
                    )
                by_idx[idx] = (p, r)
        else:
            ordered = sorted(enumerate(problems), key=lambda ip: -ip[1]["baseline_ms"])
            with ThreadPoolExecutor(max_workers=n_parallel) as ex:
                futs = [ex.submit(_solve, pair) for pair in ordered]
                for f in as_completed(futs):
                    if abort is not None:
                        continue
                    idx, p, r, block, timeout_s = f.result()
                    print(
                        f"  [{stage_name}] {idx+1}/{len(problems)} {p['sha'][:10]} "
                        f"{r.get('result')} {r.get('elapsed_ms')}ms / {timeout_s}s "
                        f"(core={_fmt(block)})",
                        flush=True,
                    )
                    if "invalid_param" in r:
                        abort = r
                        for ff in futs:
                            ff.cancel()
                        continue
                    by_idx[idx] = (p, r)
            if abort is not None:
                return _err(
                    {"error": f"invalid param: {abort['invalid_param']}"},
                    {
                        "invalid_param": abort["invalid_param"],
                        "stderr": abort.get("stderr", "")[:2000],
                        "stage": stage_name,
                    },
                )

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

        _SMALL_CACHE[(str(program_path), stage_name)] = results

        metrics = scorer.score(
            results,
            mode=score_mode,
            decisive_results=decisive,
            decided_results=decided,
            stats_weights=stats_weights,
            time_metric=time_metric,
            cost_weight=cost_weight,
        )
        metrics["stage"] = stage_name
        for k in key_stats:
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
                "stats": {k: r["stats"].get(k, 0) for k in key_stats if k in r["stats"]},
                "base_stats": {
                    k: r["baseline_stats"].get(k, 0)
                    for k in key_stats
                    if k in (r.get("baseline_stats") or {})
                },
            }
            for r in results
        ]
        artifacts = {
            "stage": stage_name,
            "summary": (
                f"solved={metrics['solved']}/{metrics['total']} "
                f"regressions={metrics['regressions']} "
                f"geomean={metrics['geomean_speedup']:.3f} "
                f"efficiency={metrics.get('efficiency', 1.0):.3f} "
                f"score={metrics['combined_score']:.3f}"
            ),
            "per_problem": sample[:20],
        }
        return EvaluationResult(metrics=metrics, artifacts=artifacts)

    def evaluate_stage(stage_n, program_path):
        w = _peek_workers(program_path)
        problems = _filter_by_sample(_load_problems(w), stage_n)
        return _evaluate(program_path, problems, f"stage{stage_n}")

    def evaluate_stage1(program_path):
        return evaluate_stage(1, program_path)

    def evaluate_stage2(program_path):
        return evaluate_stage(2, program_path)

    def evaluate_stage3(program_path):
        # Cascade chain: stage3 result then stage4 if it passes the gate.
        r3 = evaluate_stage(3, program_path)
        if not isinstance(r3, EvaluationResult):
            return r3
        gate = runtime.cascade_threshold(config_yaml, 2, default=1.03)
        if r3.metrics.get("combined_score", 0.0) < gate:
            return r3
        r4 = evaluate_stage(4, program_path)
        if not isinstance(r4, EvaluationResult):
            return r4
        merged_m = {**r3.metrics, **r4.metrics}
        merged_a = {**r3.artifacts, **r4.artifacts}
        return EvaluationResult(metrics=merged_m, artifacts=merged_a)

    def evaluate_stage4(program_path):
        return evaluate_stage(4, program_path)

    def evaluate(program_path):
        return evaluate_stage1(program_path)

    return {
        "evaluate_stage1": evaluate_stage1,
        "evaluate_stage2": evaluate_stage2,
        "evaluate_stage3": evaluate_stage3,
        "evaluate_stage4": evaluate_stage4,
        "evaluate": evaluate,
    }
