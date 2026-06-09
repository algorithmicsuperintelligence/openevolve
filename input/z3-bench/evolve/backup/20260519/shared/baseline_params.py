"""
Baseline Z3 parameters (applied_params_hash 543b29...) from problems.jsonl.
DO NOT MODIFY. Imported by all phases.
"""

BASELINE = {
    "opt.enable_core_rotate": True,
    "opt.enable_sat": True,
    "opt.enable_sls": True,
    "opt.maxres.hill_climb": True,
    "opt.maxsat_engine": "wmax",
    "opt.priority": "pareto",
    "opt.rc2.totalizer": True,
    "parallel.enable": False,
    "sat.branching.heuristic": "vsids",
    "sat.pb.solver": "totalizer",
    "sat.phase": "caching",
    "sat.random_seed": 0,
    "sat.restart": "geometric",
    "sat.threads": 1,
    "sls.random_seed": 0,
    "smt.phase_selection": 3,
    "smt.random_seed": 0,
    "smt.threads": 1,
    "threads": 1,
    # NOTE: global "threads" — Z3 4.13.x CLI rejects as positional `key=value`.
    # z3_runner.py omits it from CLI args. Default is 1 so behavior matches.
    # Kept in BASELINE for fidelity with applied_params_hash 543b29...
}

LOCKED = {
    "sat.random_seed": 0,
    "smt.random_seed": 0,
    "sls.random_seed": 0,
    "parallel.enable": False,
    "threads": 1,
}


def _self_test():
    """
    Self-test: run stage1 5 problems with BASELINE in parallel and report
    per-problem ratio vs raw baseline_ms. Use to sanity-check that the worker
    + z3 binding reproduce the recorded baseline within tolerance, and that
    parallel dispatch (taskset core pinning) doesn't perturb timing.

    Defaults: OPENEVOLVE_PARALLEL_SOLVERS=5 (matches config.yaml
    parallel_evaluations), capped at problem count.

    Status:
      OK     — result matches, ratio in [0.5, 2.0]
      WARN   — result matches, ratio out of band (noise / hw drift)
      FAIL   — result mismatch or invalid_param
    """
    import json
    import math
    import pathlib
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    here = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    from z3_runner import run_z3  # noqa: E402
    from runtime import parallel_solvers  # noqa: E402

    bench = here.parent.parent           # input/z3-bench/
    raw_dir = bench / "raw-data"
    problems_jsonl = bench / "problems.jsonl"
    stage1_sample = here / "stage1_sample.json"

    if not stage1_sample.exists():
        print(f"ERROR: {stage1_sample} missing. run build_samples.py first.",
              file=sys.stderr)
        return 2

    shas = list(json.loads(stage1_sample.read_text())["sha256"])
    idx = {}
    with open(problems_jsonl) as f:
        for line in f:
            d = json.loads(line)
            idx[d["problem_sha256"]] = {
                "smt2": d["smt2_filename"],
                "baseline_ms": d["z3_status"]["elapsed_ms"],
                "baseline_result": d["z3_status"]["result"],
            }

    tasks = []
    for i, sha in enumerate(shas):
        meta = idx.get(sha)
        if meta is None:
            print(f"ERROR: {sha[:12]} not in problems.jsonl", file=sys.stderr)
            return 2
        smt2 = raw_dir / meta["smt2"]
        if not smt2.exists():
            print(f"ERROR: smt2 not found: {smt2}", file=sys.stderr)
            return 2
        tasks.append((i, sha, meta, smt2))

    n_parallel = min(parallel_solvers(default=5), len(tasks))

    print(f"BASELINE self-test: {len(tasks)} stage1 problems, parallel={n_parallel}, "
          f"taskset core pin")
    print()

    def solve(t):
        i, sha, meta, smt2 = t
        # Generous timeout: 2x baseline_ms (matches the [0.5, 2.0] tolerance band).
        timeout_s = max(30, math.ceil(meta["baseline_ms"] * 2 / 1000))
        # Skip core 0 — pin to cores 1..n_parallel.
        core = (i % n_parallel) + 1
        r = run_z3(smt2, BASELINE, timeout_s, cpu_core=core)
        return i, sha, meta, r

    t_start = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=n_parallel) as ex:
        futures = [ex.submit(solve, t) for t in tasks]
        for fut in as_completed(futures):
            results.append(fut.result())
    elapsed = time.monotonic() - t_start
    results.sort(key=lambda x: x[0])

    print(f"{'sha':<14}{'base_res':<10}{'got_res':<10}"
          f"{'base_ms':>10}{'got_ms':>10}{'ratio':>8}  core  status")
    print("-" * 84)
    fail = 0
    warn = 0
    sum_got_ms = 0
    for i, sha, meta, r in results:
        got_result = r.get("result", "Unknown")
        got_ms = int(r.get("elapsed_ms", 0))
        sum_got_ms += got_ms
        ratio = got_ms / max(meta["baseline_ms"], 1)
        result_ok = (got_result == meta["baseline_result"])
        invalid = r.get("invalid_param")
        if invalid:
            status = f"FAIL(invalid={invalid})"
            fail += 1
        elif not result_ok:
            status = "FAIL"
            fail += 1
        elif not (0.5 <= ratio <= 2.0):
            status = "WARN"
            warn += 1
        else:
            status = "OK"
        print(f"{sha[:12]:<14}{meta['baseline_result']:<10}{got_result:<10}"
              f"{meta['baseline_ms']:>10}{got_ms:>10}{ratio:>7.2f}x  "
              f"{(i % n_parallel) + 1:>4}  {status}")

    seq_estimate = sum_got_ms / 1000
    print()
    print(f"wall-clock: {elapsed:.1f}s  (sequential would be ~{seq_estimate:.0f}s)")
    print(f"summary: {len(results) - fail - warn} ok, {warn} warn, {fail} fail")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
