"""
Solve one CP-SAT problem via ortools.sat.python.cp_model.
Subprocess worker invoked by cpsat_runner.py.

argv:
    sys.argv[1]  JSON dict of {key: value}  (params for solver.parameters)
    sys.argv[2]  path to problem.cpsat.pb (serialized CpModelProto, binary)
    sys.argv[3]  per-problem timeout in seconds

stdout: a single JSON line.
"""
import json
import os
import pickle
import sys
import time


def emit(d):
    print(json.dumps(d))
    sys.stdout.flush()


def load_problem(path):
    """raw-data layout: each dir contains problem.cpsat.pb (binary CpModelProto)."""
    from ortools.sat.python import cp_model
    p = str(path)
    if p.endswith(".pb") or p.endswith(".bin") or p.endswith(".cpsat.pb"):
        with open(p, "rb") as f:
            data = f.read()
        model = cp_model.CpModel()
        model.Proto().ParseFromString(data)
        return model
    if p.endswith(".pbtxt"):
        from google.protobuf import text_format
        with open(p, "r") as f:
            data = f.read()
        model = cp_model.CpModel()
        text_format.Parse(data, model.Proto())
        return model
    if p.endswith(".pkl"):
        with open(p, "rb") as f:
            return pickle.load(f)
    raise ValueError(f"unknown problem format: {path}")


def main():
    if len(sys.argv) != 4:
        emit({"result": "Unknown", "elapsed_ms": 0, "error": "bad argv"})
        return

    try:
        params = json.loads(sys.argv[1])
    except Exception as e:
        emit({"result": "Unknown", "elapsed_ms": 0, "error": f"params json: {e}"})
        return

    problem_path = sys.argv[2]
    timeout_s = int(sys.argv[3])

    try:
        from ortools.sat.python import cp_model
    except ImportError as e:
        emit({"result": "Unknown", "elapsed_ms": 0,
              "error": f"ortools.sat.python.cp_model import: {e}"})
        return

    solver = cp_model.CpSolver()

    # Hard wall-clock cap on the solver side too. Parent subprocess.run adds
    # an outer +15s grace.
    try:
        solver.parameters.max_time_in_seconds = float(timeout_s)
    except Exception:
        pass

    # Filter out keys that aren't real CpSolverParameters proto fields.
    # `timeout_sec` and `tuned` appear in raw-data applied_params but are
    # wrapper-level scheduler keys, not proto fields — silently drop instead
    # of treating as invalid_param.
    _DROP = {"timeout_sec", "tuned"}

    # CP-SAT params are protobuf fields. Repeated fields (lists like
    # extra_subsolvers) must be assigned via .extend() rather than `=`.
    for k, v in params.items():
        if k in _DROP:
            continue
        try:
            field = getattr(solver.parameters, k)
        except AttributeError as e:
            emit({"invalid_param": k, "error": str(e),
                  "result": "Unknown", "elapsed_ms": 0})
            return
        if isinstance(v, list):
            try:
                del field[:]
                field.extend(v)
            except (AttributeError, TypeError) as e:
                emit({"invalid_param": k,
                      "error": f"list assign: {type(e).__name__}: {e}",
                      "result": "Unknown", "elapsed_ms": 0})
                return
        else:
            try:
                setattr(solver.parameters, k, v)
            except AttributeError as e:
                emit({"invalid_param": k, "error": str(e),
                      "result": "Unknown", "elapsed_ms": 0})
                return
            except (TypeError, ValueError) as e:
                emit({"invalid_param": k,
                      "error": f"{type(e).__name__}: {e}",
                      "result": "Unknown", "elapsed_ms": 0})
                return

    try:
        model = load_problem(problem_path)
    except Exception as e:
        emit({"result": "Unknown", "elapsed_ms": 0, "error": f"problem load: {e}"})
        return

    t0 = time.monotonic()
    try:
        status = solver.Solve(model)
    except Exception as e:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        emit({"result": "Unknown", "elapsed_ms": elapsed_ms,
              "error": f"Solve() raised: {e}"})
        return
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    label_map = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.UNKNOWN: "UNKNOWN",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
    }
    label = label_map.get(status, "UNKNOWN")

    stats = {}
    for k, fn in (
        ("num_branches", "NumBranches"),
        ("num_conflicts", "NumConflicts"),
        ("num_booleans", "NumBooleans"),
        ("wall_time", "WallTime"),
        ("user_time", "UserTime"),
    ):
        try:
            v = getattr(solver, fn)()
            if isinstance(v, (int, float)):
                stats[k] = v
        except Exception:
            pass

    obj = None
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        try:
            obj = float(solver.ObjectiveValue())
        except Exception:
            obj = None

    out = {"result": label, "elapsed_ms": elapsed_ms, "stats": stats}
    if obj is not None:
        out["objective"] = obj
    emit(out)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
