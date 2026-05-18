"""
Generate stage1_sample.json: 5 problems stratified by baseline elapsed_ms quantile.

Fixed seed (42) for reproducibility. Sort by baseline_ms, split into 5 equal-size
buckets, pick one from each bucket. Result written to shared/stage1_sample.json.
"""
import json
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parent
PROBLEMS = ROOT.parent / "problems.jsonl"
OUT = ROOT / "shared" / "stage1_sample.json"

NUM_BUCKETS = 5
SEED = 42


def main():
    rows = []
    with open(PROBLEMS) as f:
        for line in f:
            d = json.loads(line)
            rows.append(
                (
                    d["problem_sha256"],
                    d["z3_status"]["elapsed_ms"],
                    d["z3_status"]["result"],
                )
            )
    rows.sort(key=lambda r: r[1])

    n = len(rows)
    if n < NUM_BUCKETS:
        raise SystemExit(f"need >= {NUM_BUCKETS} problems, got {n}")

    rng = random.Random(SEED)
    picked = []
    for i in range(NUM_BUCKETS):
        lo = i * n // NUM_BUCKETS
        hi = (i + 1) * n // NUM_BUCKETS
        bucket = rows[lo:hi]
        if bucket:
            picked.append(rng.choice(bucket))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "seed": SEED,
                "num_buckets": NUM_BUCKETS,
                "source": str(PROBLEMS.relative_to(ROOT.parent.parent)),
                "sha256": [r[0] for r in picked],
                "summary": [
                    {
                        "sha": r[0][:12],
                        "baseline_ms": r[1],
                        "baseline_result": r[2],
                    }
                    for r in picked
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUT.relative_to(ROOT.parent.parent)} ({len(picked)} problems)")
    for s in picked:
        print(f"  {s[0][:12]}  {s[1]:>8} ms  {s[2]}")


if __name__ == "__main__":
    main()
