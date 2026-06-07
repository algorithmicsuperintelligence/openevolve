"""Run all per-node primitive kernel tests.
Requires CUDA. Run from repo root or the benchmark directory:
  python3 atenir/primitive_kernels/run_all_tests.py
"""
import sys
import importlib
from pathlib import Path

NODES = [
    "sub", "mul", "mul_1", "mul_2",
    "sum_1", "mul_3", "sum_2", "mul_4",
    "sub_1", "sub_2", "div", "mul_5",
    "mul_6", "sum_3", "sum_4",
]

passed = skipped = failed = 0

sys.path.insert(0, str(Path(__file__).parent))

for name in NODES:
    mod = importlib.import_module(name)
    try:
        mod.test()
        passed += 1
    except SystemExit:
        skipped += 1
    except Exception as e:
        print(f"FAIL: {name}  —  {e}")
        failed += 1

print(f"\n{passed} passed  |  {skipped} skipped  |  {failed} failed")
if failed:
    sys.exit(1)
