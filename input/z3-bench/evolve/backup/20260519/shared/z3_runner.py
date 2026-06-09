"""
Run a single SMT2 file through Z3 with given parameters.

Implementation: spawn `_z3_solve_worker.py` as a subprocess and communicate
via stdout JSON. The worker imports the z3 Python binding and applies params
via `z3.set_param()`, matching the original benchmark setup that recorded
`applied_params_hash 543b29...`. This is necessary because the z3 CLI
positional `key=value` form rejects globals (`threads`, `parallel.enable`,
`sls.parallel`) that the Python binding accepts.

Subprocess isolation gives: hard wall-clock timeout, crash containment,
memory reclaim between problems.
"""
import json
import pathlib
import shutil
import subprocess
import sys
import time

_WORKER = str(pathlib.Path(__file__).resolve().parent / "_z3_solve_worker.py")
_TASKSET = shutil.which("taskset")  # Linux only; None on macOS / missing


def run_z3(smt2_path, params, timeout_s, python_bin=None, cpu_core=None):
    """
    Returns dict (one of):
      success: {"result": "Sat"|"Unsat"|"Unknown", "elapsed_ms": int, "stats": dict}
      timeout: {"result": "Unknown", "elapsed_ms": int, "timeout": True, "stats": {}}
      invalid: {"invalid_param": str, "stderr": str, "result": "Unknown", "elapsed_ms": int, "stats": {}}
      crash:   {"result": "Unknown", "elapsed_ms": int, "error": str, "stderr": str, "stats": {}}

    "stats" mirrors z3 Optimize.statistics() numeric entries (decisions,
    propagations, conflicts, restarts, arith/bv counters, ...). Empty when
    z3 never reached check() (param error, parse fail, timeout, crash).

    cpu_core: optional int — if given and `taskset` is on PATH, pin the
    worker subprocess to that core (-c <N>). Used by the parallel dispatch
    in evaluator.py to isolate concurrent z3 runs from cross-core
    interference. Silently ignored if taskset missing (macOS / no util-linux).
    """
    py = python_bin or sys.executable
    args = [py, _WORKER, json.dumps(params), str(smt2_path), str(int(timeout_s))]
    if cpu_core is not None and _TASKSET:
        args = [_TASKSET, "-c", str(int(cpu_core))] + args

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout_s + 15,  # grace for z3 startup + parse
        )
    except subprocess.TimeoutExpired:
        return {
            "result": "Unknown",
            "elapsed_ms": int((time.monotonic() - t0) * 1000),
            "timeout": True,
            "stats": {},
        }

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    if not stdout:
        return {
            "result": "Unknown",
            "elapsed_ms": int((time.monotonic() - t0) * 1000),
            "error": f"worker produced no output (rc={proc.returncode})",
            "stderr": stderr[-2000:],
            "stats": {},
        }

    # Use the last non-empty line as JSON (defensive against stray prints).
    last = stdout.splitlines()[-1]
    try:
        out = json.loads(last)
    except json.JSONDecodeError as e:
        return {
            "result": "Unknown",
            "elapsed_ms": int((time.monotonic() - t0) * 1000),
            "error": f"worker json decode: {e}",
            "stderr": (stderr + "\n--stdout--\n" + stdout)[-2000:],
            "stats": {},
        }

    if "invalid_param" in out:
        return {
            "invalid_param": out["invalid_param"],
            "stderr": (out.get("error") or stderr)[-2000:],
            "result": "Unknown",
            "elapsed_ms": out.get("elapsed_ms", 0),
            "stats": {},
        }
    out.setdefault("stats", {})
    return out
