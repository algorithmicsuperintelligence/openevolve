"""
Solve one SMT2 file using the z3 Python binding (z3.set_param + z3.Optimize).
Matches the original benchmark setup (applied_params_hash 543b29...): params
are applied via z3.set_param so globals like 'threads' / 'parallel.enable' /
'sls.parallel' work, unlike CLI positional `key=value`.

Solver mode is read from the sibling `config.yaml` `bench.solver_mode`
(default "optimize"):
  - "optimize": z3.Optimize, full soft-constraint optimization (+ objective).
  - "sat":      z3.Solver over z3.parse_smt2_file (which drops `assert-soft`),
                i.e. hard-constraint feasibility only — faster, no objective.
                `opt.*` params are Optimize-only and silently dropped here.

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
import pathlib
import sys
import time


def emit(d):
    print(json.dumps(d))
    sys.stdout.flush()


def _solver_mode():
    """Read `bench.solver_mode` from the sibling config.yaml. Returns
    "optimize" (default) or "sat". Config-driven so no env var is needed;
    one parse per worker process is negligible next to the z3 import."""
    try:
        import yaml

        cfg_path = pathlib.Path(__file__).resolve().parent / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        return ((cfg.get("bench") or {}).get("solver_mode")) or "optimize"
    except Exception:
        return "optimize"


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

    mode = _solver_mode()

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
            # opt.* is Optimize-only. In sat mode (z3.Solver) it has no target,
            # so drop it silently rather than flagging it as an invalid_param.
            if mode != "sat":
                opt_local[k[len("opt.") :]] = v
            continue
        try:
            z3.set_param(k, v)
        except z3.Z3Exception as e:
            emit({"invalid_param": k, "error": str(e), "result": "Unknown", "elapsed_ms": 0})
            return
        except Exception as e:
            emit(
                {
                    "invalid_param": k,
                    "error": f"{type(e).__name__}: {e}",
                    "result": "Unknown",
                    "elapsed_ms": 0,
                }
            )
            return

    # Soft timeout (z3 polls at safe points) — outer subprocess.run() also
    # enforces a hard wall-clock cap.
    try:
        z3.set_param("timeout", int(timeout_s * 1000))
    except Exception:
        pass

    if mode == "sat":
        # SAT feasibility only: parse_smt2_file drops `assert-soft` (Optimize
        # extension) and returns just the hard assertions. Solver stops at the
        # first feasible model — no objective search.
        o = z3.Solver()
        try:
            o.add(z3.parse_smt2_file(smt2_path))
        except Exception as e:
            emit({"result": "Unknown", "elapsed_ms": 0, "error": f"smt2 parse: {e}"})
            return
    else:
        o = z3.Optimize()

        for k, v in opt_local.items():
            try:
                o.set(k, v)
            except z3.Z3Exception as e:
                emit(
                    {
                        "invalid_param": "opt." + k,
                        "error": str(e),
                        "result": "Unknown",
                        "elapsed_ms": 0,
                    }
                )
                return
            except Exception as e:
                emit(
                    {
                        "invalid_param": "opt." + k,
                        "error": f"{type(e).__name__}: {e}",
                        "result": "Unknown",
                        "elapsed_ms": 0,
                    }
                )
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

    # Cost mode (optimize): score on z3's deterministic work measure so the
    # signal is machine-independent and immune to the sub-second wall-clock
    # noise this workload exhibits (median solve ~0.65s). "rlimit count" is
    # z3's deterministic step counter — _lib.scorer._time_ratio reads
    # stats["deterministic_time"].
    if "rlimit count" in stats:
        stats["deterministic_time"] = stats["rlimit count"]

    # Optimize objective value (sum of violated assert-soft weights). These
    # instances carry exactly one implicit objective; evaluate it in the model.
    # Used by cost mode as a CORRECTNESS GUARD: a good variant must still reach
    # the baseline optimum (0 on this workload); a worse / non-zero objective is
    # penalized via cost_ratio < 1.
    # sat mode is a plain Solver — no objective() to read (and speedup scoring
    # ignores it anyway). Only Optimize exposes objectives().
    objective = None
    if mode != "sat" and label == "Sat":
        try:
            objs = o.objectives()
            if objs:
                val = o.model().eval(objs[0], model_completion=True)
                try:
                    objective = val.as_long()
                except Exception:
                    try:
                        objective = float(val.as_fraction())
                    except Exception:
                        objective = float(str(val))
        except Exception:
            objective = None

    out = {"result": label, "elapsed_ms": elapsed_ms, "stats": stats}
    if objective is not None:
        out["objective"] = objective
    emit(out)


if __name__ == "__main__":
    main()
    # Bypass z3 atexit/teardown that can abort the subprocess after a clean
    # emit() — would mask the result as "worker produced no output".
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
