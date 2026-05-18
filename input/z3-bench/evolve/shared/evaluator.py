"""
OpenEvolve evaluator for Z3 parameter tuning.

Cascade:
  stage1: 5 stratified problems (stage1_sample.json), per-problem 15s
  stage2: full 50 problems, per-problem 120s

Score: geomean(speedup) * solved_rate^2.

Locked params (sat.random_seed / smt.random_seed / sls.random_seed / parallel.enable)
must not deviate from baseline_params.LOCKED — violation => combined_score=0.

Environment overrides:
  OPENEVOLVE_MAX_PROBLEMS    int — cap stage2 problem count
  OPENEVOLVE_STAGE1_TIMEOUT  int sec — per-problem timeout in stage1 (default 15)
  OPENEVOLVE_STAGE2_TIMEOUT  int sec — per-problem timeout in stage2 (default 120)
  OPENEVOLVE_Z3_BIN          str — z3 binary path (default "z3")
"""
import importlib.util
import json
import os
import pathlib
import sys
import traceback

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from baseline_params import BASELINE, LOCKED  # noqa: E402
from score import score  # noqa: E402
from z3_runner import run_z3  # noqa: E402

from openevolve.evaluation_result import EvaluationResult  # noqa: E402

_BENCH_DIR = _HERE.parent.parent          # input/z3-bench/
_RAW_DIR = _BENCH_DIR / "raw-data"
_PROBLEMS_JSONL = _BENCH_DIR / "problems.jsonl"
_STAGE1_SAMPLE = _HERE / "stage1_sample.json"

_Z3_BIN = os.environ.get("OPENEVOLVE_Z3_BIN", "z3")


def _load_program(path):
    spec = importlib.util.spec_from_file_location("program", path)
    module = importlib.util.module_from_spec(spec)
    # Phase initial_programs add shared/ to sys.path themselves.
    spec.loader.exec_module(module)
    return module


def _load_problems():
    rows = []
    with open(_PROBLEMS_JSONL) as f:
        for line in f:
            d = json.loads(line)
            rows.append(
                {
                    "sha": d["problem_sha256"],
                    "smt2": d["smt2_filename"],
                    "baseline_ms": d["z3_status"]["elapsed_ms"],
                    "baseline_result": d["z3_status"]["result"],
                }
            )
    return rows


def _filter_stage1(problems):
    if not _STAGE1_SAMPLE.exists():
        return problems[:5]
    keep = set(json.loads(_STAGE1_SAMPLE.read_text())["sha256"])
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


def _evaluate(program_path, problems, per_problem_timeout_s, stage_name):
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

    results = []
    for p in problems:
        smt2_path = _RAW_DIR / p["smt2"]
        if not smt2_path.exists():
            return _err_result(
                {"error": f"smt2 not found: {p['smt2']}"},
                {"missing_file": str(smt2_path), "stage": stage_name},
            )
        r = run_z3(smt2_path, params, per_problem_timeout_s, z3_bin=_Z3_BIN)
        if "invalid_param" in r:
            return _err_result(
                {"error": f"invalid z3 param: {r['invalid_param']}"},
                {
                    "invalid_param": r["invalid_param"],
                    "stderr": r.get("stderr", "")[:2000],
                    "stage": stage_name,
                    "suggestion": "Remove or fix this key in get_params().",
                },
            )
        results.append(
            {
                **p,
                "result": r["result"],
                "elapsed_ms": r["elapsed_ms"],
                "timeout": bool(r.get("timeout")),
            }
        )

    metrics = score(results)
    metrics["stage"] = stage_name

    sample = [
        {
            "sha": r["sha"][:10],
            "base_result": r["baseline_result"],
            "got_result": r["result"],
            "base_ms": r["baseline_ms"],
            "ms": r["elapsed_ms"],
            "speedup": round(r["baseline_ms"] / max(r["elapsed_ms"], 1), 3),
            "timeout": r["timeout"],
        }
        for r in results
    ]
    artifacts = {
        "stage": stage_name,
        "summary": (
            f"solved={metrics['solved']}/{metrics['total']} "
            f"regressions={metrics['regressions']} "
            f"geomean_speedup={metrics['geomean_speedup']:.3f} "
            f"score={metrics['combined_score']:.3f}"
        ),
        "per_problem": sample[:20],
    }
    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def evaluate_stage1(program_path):
    timeout = int(os.environ.get("OPENEVOLVE_STAGE1_TIMEOUT", "15"))
    problems = _filter_stage1(_load_problems())
    return _evaluate(program_path, problems, timeout, "stage1")


def evaluate_stage2(program_path):
    timeout = int(os.environ.get("OPENEVOLVE_STAGE2_TIMEOUT", "120"))
    problems = _load_problems()
    return _evaluate(program_path, problems, timeout, "stage2")


def evaluate(program_path):
    return evaluate_stage2(program_path)
