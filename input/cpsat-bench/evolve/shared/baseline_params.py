"""
Baseline CP-SAT parameters captured from raw-data/<sha>__<hash>__seed0.meta.jsonl
(uniform `cpsat_applied_params` across all problems in package 71 instances).

DO NOT MODIFY. Imported by all phases.

Pkg 71 applied_params has 5 keys: timeout_sec, num_search_workers, random_seed,
interleave_search, tuned. `timeout_sec` and `tuned` are wrapper-level
(scheduler) — NOT real CpSolverParameters proto fields. Including them
would make every variant fail with invalid_param. Dropped from BASELINE
and from LOCKED; the worker also has both in its _DROP set as belt+braces.
"""

BASELINE = {
    "num_search_workers": 8,
    "random_seed": 0,
    "interleave_search": True,
}

LOCKED = {
    "random_seed": 0,
    "num_search_workers": 8,
}


def _self_test():
    """
    Run stage1 problems with BASELINE in parallel; report per-problem ratio vs
    recorded baseline_ms.
      OK   ratio in [0.5, 2.0]
      WARN ratio out of band
      FAIL result mismatch / invalid_param
    """
    import json
    import math
    import pathlib
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    here = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    from cpsat_runner import run_cpsat  # noqa: E402
    from runtime import parallel_solvers, core_range  # noqa: E402

    bench = here.parent.parent
    raw_dir = bench / "raw-data"
    problems_jsonl = bench / "problems.jsonl"
    stage1_sample = here / "stage1_sample.json"

    if not stage1_sample.exists():
        print(f"ERROR: {stage1_sample} missing. run build_samples.py first.",
              file=sys.stderr)
        return 2

    ids = list(json.loads(stage1_sample.read_text())["sha256"])
    idx = {}
    with open(problems_jsonl) as f:
        for line in f:
            d = json.loads(line)
            idx[d["problem_sha256"]] = {
                "problem_filename": d["problem_filename"],
                "baseline_ms": (d.get("cpsat_status") or {}).get("elapsed_ms", 0),
                "baseline_result": (d.get("cpsat_status") or {}).get("result"),
            }

    tasks = []
    for i, pid in enumerate(ids):
        meta = idx.get(pid)
        if meta is None:
            print(f"ERROR: {pid[:12]} not in problems.jsonl", file=sys.stderr)
            return 2
        problem_path = raw_dir / meta["problem_filename"]
        if not problem_path.exists():
            print(f"ERROR: input not found: {problem_path}", file=sys.stderr)
            return 2
        tasks.append((i, pid, meta, problem_path))

    cores = core_range()
    if cores is None:
        cores = list(range(1, parallel_solvers(default=5) + 1))
    n_parallel = min(len(cores), len(tasks))
    cores = cores[:n_parallel]

    print(f"BASELINE self-test: {len(tasks)} stage1 problems, parallel={n_parallel} "
          f"cores={cores}")
    print()

    def solve(t):
        i, pid, meta, problem_path = t
        timeout_s = max(30, math.ceil(meta["baseline_ms"] * 2 / 1000))
        core = cores[i % n_parallel]
        r = run_cpsat(problem_path, BASELINE, timeout_s, cpu_core=core)
        return i, pid, meta, r

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
    for i, pid, meta, r in results:
        got_result = r.get("result", "Unknown")
        got_ms = int(r.get("elapsed_ms", 0))
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
        print(f"{pid[:12]:<14}{str(meta['baseline_result']):<10}{str(got_result):<10}"
              f"{int(meta['baseline_ms']):>10}{got_ms:>10}{ratio:>7.2f}x  "
              f"{cores[i % n_parallel]:>4}  {status}")

    print()
    print(f"wall-clock: {elapsed:.1f}s")
    print(f"summary: {len(results) - fail - warn} ok, {warn} warn, {fail} fail")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
