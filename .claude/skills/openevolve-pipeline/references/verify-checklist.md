# Verify Checklist

Run after scaffolding. Each step gates the next — stop at first failure.

## 0. Solver installed?

```bash
python3 -c "import <solver_pkg>; print(<solver_pkg>.__version__)"
# OR
command -v <solver_binary> && <solver_binary> --version
```

Fail → tell user to install per `bench.solver_install_hint`.

## 1. params.json valid?

```bash
cd input
python3 -c "
import sys; sys.path.insert(0, '.')
from _lib import params_catalog
c = params_catalog.load('<bench>/evolve/params.json')
print('keys:', len(c.known_keys()), 'defaults:', len(c.defaults), 'locked:', len(c.locked))
print('validate ok:', c.validate(c.defaults))
print('validate bogus:', c.validate({'fake_key': 1}))
"
```

Expect: `validate ok: []`, `validate bogus: [('fake_key', 'unknown key (not in catalog)')]`.

Fail modes:
- `KeyError` → typo in `groups.*.params` schema (type/default/values missing).
- `validate ok` returns non-empty list → a default value violates its declared type/range.

## 2. problems.jsonl readable + adapter fields wired?

```bash
python3 -m _lib.sampler <bench>
ls <bench>/evolve/cache/stage{1,2,3,4}_sample.json
```

Expect: 4 stage sample files, with non-trivial cluster spread shown in stdout
(problem SHAs across multiple `elapsed_ms` buckets).

Fail modes:
- "all problems in cluster 0" → `clustering.feature` path wrong, or feature value `None`/`0` for every problem. Check `head -1 problems.jsonl` against the dotted path.
- KeyError on `PROBLEM_FILE_FIELD`/`STATUS_FIELD` → adapter constants don't match problems.jsonl. Fix adapter.py.

## 3. Adapter sane on BASELINE?

```bash
python3 -m _lib.self_test <bench>
```

Expect: result labels match baseline for stage1; ratio in [0.5, 2.0]. WARN tolerated.

Fail modes:
- "result mismatch" on multiple problems → `_solve_worker.py` returns wrong token. Fix result mapping.
- "ratio > 2.0" — solver invocation is significantly slower than baseline. Check params application (some keys default differently in your binding version than the baseline expects).
- `invalid_param` emitted → BASELINE has a key the binding doesn't recognize. Either drop the key from `defaults` or fix worker.

## 4. Local baseline captured? (slow, ~10× problem timeouts)

```bash
python3 -m _lib.rebaseline <bench>
ls <bench>/evolve/cache/local_baseline.json
```

Skip on resource-constrained dev machines with `SKIP_REBASELINE=1` —
later evaluator will use `problems.jsonl` baseline directly.

## 5. One-phase smoke

```bash
./input/run_phase.sh <bench> 1 --pin 2-3 --iterations 2
ls <bench>/evolve/phase1_*/openevolve_output/best/best_program.py
ls <bench>/evolve/cache/phase1_best.json
```

Expect: best_program.py exists, phase1_best.json contains a dict of overrides.

Fail modes:
- "OPENEVOLVE_BENCH_ROOT unset" → phase module's `_resolve_bench_root()`
  fallback failed. Confirm phase dir is two levels under `<bench>` (i.e.
  `<bench>/evolve/phase1_x/initial_program.py`).
- Evaluator returns 0 for all candidates → check phase module's `get_params()`
  signature matches what evaluator calls (simple: `get_params()`;
  cpsat-style: `get_params(problem=None, stage=None)`).
- LLM never mutates EVOLVE-BLOCK → docstring missing or `unified_dict_name`
  mismatch with last phase (only matters for last phase).
