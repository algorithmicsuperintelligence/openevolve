"""
Solve one SMT2 file using the z3 Python binding (z3.set_param + z3.Optimize).
Matches the original benchmark setup (applied_params_hash 543b29...): params
are applied via z3.set_param so globals like 'threads' / 'parallel.enable' /
'sls.parallel' work, unlike CLI positional `key=value`.

Invoked as a subprocess by z3_runner.py for process isolation + hard timeout.

argv:
    sys.argv[1]  JSON dict of {key: value}    (params)
    sys.argv[2]  smt2 file path
    sys.argv[3]  per-problem timeout in seconds

stdout: a single JSON line, one of:
    {"result": "Sat"|"Unsat"|"Unknown", "elapsed_ms": int, "stats": {<k>: <v>, ...}}
    {"result": "Unknown", "elapsed_ms": int, "timeout": true, "stats": {...}?}
    {"invalid_param": "<key>", "error": "<msg>", "result": "Unknown", "elapsed_ms": 0}
    {"result": "Unknown", "elapsed_ms": 0, "error": "<msg>"}

"stats" mirrors z3 Optimize.statistics() (decisions, propagations, conflicts,
restarts, plus tactic-specific counters like arith/bv overflow, mk-clause, ...).
Numeric values only; non-numeric keys dropped to keep JSON small.
"""
import json
import os
import sys
import time


def emit(d):
    print(json.dumps(d))
    sys.stdout.flush()


def main():
    if len(sys.argv) != 4:
        emit({"result": "Unknown", "elapsed_ms": 0, "error": "bad argv"})
        return

    try:
        params = json.loads(sys.argv[1])
    except Exception as e:
        emit({"result": "Unknown", "elapsed_ms": 0, "error": f"params json: {e}"})
        return

    smt2_path = sys.argv[2]
    timeout_s = int(sys.argv[3])

    try:
        import z3
    except ImportError as e:
        emit({"result": "Unknown", "elapsed_ms": 0, "error": f"z3 binding import: {e}"})
        return

    # Split params per route_solver.cpp set_z3_param_optimize_option():
    #   opt.* keys → per-Optimize via opt.set(Params) (strip "opt." prefix)
    #   all others → global via z3.set_param
    # Matches cpp ground truth: priority/maxsat_engine/enable_*/maxres.*/rc2.*
    # are set on the Optimize instance (opt.set), not globally.
    # Mirror cpp route_solver.cpp:613 — suppress unknown-param warnings
    # BEFORE any other set_param. Without this, z3 4.15.x emits a warning +
    # dumps the legal-param list to stderr for keys like "threads" (no-op in
    # this version), which aborts the subprocess in stderr-piped mode.
    try:
        z3.set_param("warning", False)
    except Exception:
        pass

    opt_local = {}
    for k, v in params.items():
        if k.startswith("opt."):
            opt_local[k[len("opt."):]] = v
            continue
        try:
            z3.set_param(k, v)
        except z3.Z3Exception as e:
            emit({"invalid_param": k, "error": str(e), "result": "Unknown", "elapsed_ms": 0})
            return
        except Exception as e:
            emit({"invalid_param": k, "error": f"{type(e).__name__}: {e}", "result": "Unknown", "elapsed_ms": 0})
            return

    # Soft timeout (z3 polls at safe points) — outer subprocess.run() also
    # enforces a hard wall-clock cap.
    try:
        z3.set_param("timeout", int(timeout_s * 1000))
    except Exception:
        pass

    o = z3.Optimize()

    for k, v in opt_local.items():
        try:
            o.set(k, v)
        except z3.Z3Exception as e:
            emit({"invalid_param": "opt." + k, "error": str(e), "result": "Unknown", "elapsed_ms": 0})
            return
        except Exception as e:
            emit({"invalid_param": "opt." + k, "error": f"{type(e).__name__}: {e}", "result": "Unknown", "elapsed_ms": 0})
            return

    try:
        o.from_file(smt2_path)
    except Exception as e:
        emit({"result": "Unknown", "elapsed_ms": 0, "error": f"smt2 parse: {e}"})
        return

    t0 = time.monotonic()
    try:
        res = o.check()
    except Exception as e:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        emit(
            {
                "result": "Unknown",
                "elapsed_ms": elapsed_ms,
                "error": f"check() raised: {e}",
            }
        )
        return
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    if res == z3.sat:
        label = "Sat"
    elif res == z3.unsat:
        label = "Unsat"
    else:
        label = "Unknown"

    stats = {}
    try:
        st = o.statistics()
        for k in st.keys():
            try:
                v = st.get_key_value(k)
            except Exception:
                continue
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                stats[k] = v
    except Exception:
        stats = {}

    emit({"result": label, "elapsed_ms": elapsed_ms, "stats": stats})


if __name__ == "__main__":
    main()
    # Bypass z3 atexit/teardown that can abort the subprocess after a clean
    # emit() — would mask the result as "worker produced no output".
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
