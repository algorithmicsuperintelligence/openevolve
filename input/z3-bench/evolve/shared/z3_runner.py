"""
Run Z3 CLI on a single SMT2 file with parameter dict.
Subprocess isolation: each run a fresh process with -T:<sec> wall-clock cap.
"""
import re
import shutil
import subprocess
import time

_RESULT_RE = re.compile(r"^(sat|unsat|unknown)\b", re.MULTILINE)
_INVALID_PARAM_RES = [
    re.compile(r"unknown\s+parameter\s+'?([\w.\-]+)'?", re.IGNORECASE),
    re.compile(r"invalid\s+parameter\s+'?([\w.\-]+)'?", re.IGNORECASE),
    re.compile(r"unknown\s+option\s+'?([\w.\-]+)'?", re.IGNORECASE),
    re.compile(r"error\s+setting\s+'?([\w.\-]+)'?", re.IGNORECASE),
]


def _format_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def detect_invalid_param(stderr):
    for rx in _INVALID_PARAM_RES:
        m = rx.search(stderr)
        if m:
            return m.group(1)
    if any(tok in stderr.lower() for tok in ("unknown parameter", "invalid parameter", "unknown option")):
        return "<unparsed>"
    return None


def run_z3(smt2_path, params, timeout_s, z3_bin="z3"):
    """
    Returns dict:
      success: {"result": "Sat"|"Unsat"|"Unknown", "elapsed_ms": int}
      timeout: {"result": "Unknown", "elapsed_ms": int, "timeout": True}
      invalid: {"invalid_param": str, "stderr": str, "result": "Unknown", "elapsed_ms": int}
      crash:   {"result": "Unknown", "elapsed_ms": int, "error": str, "stderr": str}
    """
    if shutil.which(z3_bin) is None:
        return {"result": "Unknown", "elapsed_ms": 0, "error": f"z3 binary not found: {z3_bin}"}

    args = [z3_bin, f"-T:{int(timeout_s)}", "-smt2"]
    for k, v in params.items():
        args.append(f"{k}={_format_value(v)}")
    args.append(str(smt2_path))

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout_s + 10,
        )
    except subprocess.TimeoutExpired:
        return {
            "result": "Unknown",
            "elapsed_ms": int(timeout_s * 1000),
            "timeout": True,
        }

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    bad = detect_invalid_param(stderr)
    if bad:
        return {
            "invalid_param": bad,
            "stderr": stderr[-2000:],
            "result": "Unknown",
            "elapsed_ms": elapsed_ms,
        }

    m = _RESULT_RE.search(stdout)
    if not m:
        return {
            "result": "Unknown",
            "elapsed_ms": elapsed_ms,
            "error": f"no result token (rc={proc.returncode})",
            "stderr": stderr[-1000:],
        }

    return {
        "result": {"sat": "Sat", "unsat": "Unsat", "unknown": "Unknown"}[m.group(1)],
        "elapsed_ms": elapsed_ms,
    }
