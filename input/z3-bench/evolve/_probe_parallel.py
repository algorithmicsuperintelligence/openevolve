"""Probe: serial vs parallel z3 measurement noise (run inside docker)."""
import json
import pathlib
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "shared"))
from baseline_params import BASELINE  # noqa
from z3_runner import run_z3  # noqa


def main():
    shas = json.loads((_HERE / "shared/stage1_sample.json").read_text())["sha256"]
    prob_idx = {}
    with open(_HERE.parent / "problems.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d["problem_sha256"] in shas:
                prob_idx[d["problem_sha256"]] = d["smt2_filename"]

    raw_dir = _HERE.parent / "raw-data"
    tasks = [(i, sha, raw_dir / prob_idx[sha]) for i, sha in enumerate(shas)]

    def solve(i, sha, smt2, pin):
        r = run_z3(smt2, BASELINE, 60, cpu_core=pin)
        return i, sha, r

    # --- A: serial, no pin (3 reps each) ---
    print("=== A: serial no-pin (3 reps) ===")
    serial = {sha: [] for _, sha, _ in tasks}
    for rep in range(3):
        for i, sha, smt2 in tasks:
            t0 = time.monotonic()
            _, _, r = solve(i, sha, smt2, None)
            wall = int((time.monotonic() - t0) * 1000)
            serial[sha].append((r.get("result"), r["elapsed_ms"], wall))
            print(f"  rep{rep} [{i+1}/5] {sha[:10]} {r.get('result')} "
                  f"z3={r['elapsed_ms']}ms wall={wall}ms")

    # --- B: parallel=5, pin (3 reps batch) ---
    print("\n=== B: parallel=5 pinned (3 reps) ===")
    par = {sha: [] for _, sha, _ in tasks}
    for rep in range(3):
        with ThreadPoolExecutor(max_workers=5) as ex:
            futs = [ex.submit(solve, i, sha, smt2, (i % 5) + 1) for i, sha, smt2 in tasks]
            for fut in futs:
                i, sha, r = fut.result()
                par[sha].append((r.get("result"), r["elapsed_ms"]))
                print(f"  rep{rep} [{i+1}/5] {sha[:10]} {r.get('result')} "
                      f"z3={r['elapsed_ms']}ms core={(i % 5) + 1}")

    # --- summary ---
    print("\n=== SUMMARY (z3 elapsed_ms) ===")
    print(f"{'sha':<12}  {'serial(med/stdev)':<24}  {'parallel(med/stdev)':<24}  par/ser")
    for _, sha, _ in tasks:
        s = [t[1] for t in serial[sha]]
        p = [t[1] for t in par[sha]]
        sm, ss = statistics.median(s), statistics.stdev(s) if len(s) > 1 else 0
        pm, ps = statistics.median(p), statistics.stdev(p) if len(p) > 1 else 0
        ratio = pm / sm if sm else 0
        print(f"{sha[:12]}  {sm:>7.0f} / {ss:>6.0f}        {pm:>7.0f} / {ps:>6.0f}        "
              f"{ratio:.2f}x")
        # result drift
        s_res = set(t[0] for t in serial[sha])
        p_res = set(t[0] for t in par[sha])
        if s_res != p_res:
            print(f"  RESULT DRIFT: serial={s_res} parallel={p_res}")


if __name__ == "__main__":
    main()
