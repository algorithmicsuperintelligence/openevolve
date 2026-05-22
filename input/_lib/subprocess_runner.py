"""
Generic subprocess solver runner. Spawn a worker script as a subprocess and
parse a JSON dict from its stdout. Per-bench `<solver>_runner.py` wrappers
call `run_solver(WORKER_PATH, ...)` and re-export under solver-specific names.

Contract for the worker (any language, must print JSON):
  argv[1]  json.dumps(params)
  argv[2]  str(problem_path)
  argv[3]  str(int(timeout_s))

  stdout (last non-empty line, JSON):
    success: {"result": <str>, "elapsed_ms": int, "stats": dict, ...}
    timeout: {"result": "Unknown", "elapsed_ms": int, "timeout": True, "stats": {}}
    invalid: {"invalid_param": str, "error": str, "result": "Unknown", "elapsed_ms": int}
    crash:   {"result": "Unknown", "elapsed_ms": int, "error": str}

This parent layer adds:
  - hard wall-clock timeout (timeout_s + 15s grace)
  - optional taskset core pin (Linux only; silently ignored if missing)
  - defensive last-line parsing (defensive against stray worker prints)

Return shape — always a dict, one of:
  {"result", "elapsed_ms", "stats", ...}                (success, may include "objective")
  {"result": "Unknown", "elapsed_ms", "timeout": True, "stats": {}}
  {"invalid_param", "stderr", "result": "Unknown", "elapsed_ms", "stats": {}}
  {"result": "Unknown", "elapsed_ms", "error", "stderr", "stats": {}}
"""
import json
import shutil
import subprocess
import sys
import time

_TASKSET = shutil.which("taskset")


def run_solver(worker_path, problem_path, params, timeout_s,
               python_bin=None, cpu_core=None, grace_s=15):
    """
    See module docstring for the contract.

    cpu_core: optional Linux taskset pin. Accepts:
      - int                  → pin to one core (e.g. 3 → "taskset -c 3")
      - Sequence[int] / str  → pin to a set of cores (e.g. [2,3,4,5] or
                               "2-5,7" → "taskset -c 2,3,4,5" / "2-5,7").
      Ignored on macOS / no util-linux.
    grace_s: subprocess timeout = timeout_s + grace_s.
    """
    py = python_bin or sys.executable
    args = [py, str(worker_path), json.dumps(params),
            str(problem_path), str(int(timeout_s))]
    if cpu_core is not None and _TASKSET:
        if isinstance(cpu_core, str):
            cpu_arg = cpu_core
        elif isinstance(cpu_core, (list, tuple)):
            if not cpu_core:
                cpu_arg = None
            else:
                cpu_arg = ",".join(str(int(c)) for c in cpu_core)
        else:
            cpu_arg = str(int(cpu_core))
        if cpu_arg:
            args = [_TASKSET, "-c", cpu_arg] + args

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout_s + grace_s,
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

    # Last non-empty line as JSON (defensive against stray worker prints).
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
