---
name: openevolve-pipeline
description: Scaffold an OpenEvolve solver-parameter tuning pipeline under input/<bench>/evolve/ when a user supplies a new solver + benchmark dataset. Trigger on requests like "add a new solver to openevolve", "make an evolve pipeline for <solver>", "tune <solver> parameters on this bench", "generate the cpsat-bench style scaffold for X", or whenever the user drops raw solver runs into input/<bench>/raw-data and asks for parameter tuning. The skill follows input/ADD_NEW_SOLVER.md: it interviews for solver/scoring/clustering/phase decisions, writes exactly the 4 per-bench files (config.yaml, params.json, adapter.py, _solve_worker.py) + N phase modules, and verifies the result with the _lib CLIs (sampler / self_test / rebaseline).
---

# OpenEvolve Pipeline Generator

Given a new solver and a raw benchmark dataset, produce a complete per-bench tuning pipeline at `input/<bench>/evolve/`. All orchestration lives in `input/_lib/` — **never edit it**. The bench contributes only the per-bench surface from `input/ADD_NEW_SOLVER.md` §2:

| file | purpose |
|---|---|
| `evolve/config.yaml` | bench + LLM + clustering + evaluation knobs |
| `evolve/params.json` | rich solver parameter catalog (defaults / locked / groups) |
| `evolve/adapter.py` | solver hooks (constants + `get_problem_size`) |
| `evolve/_solve_worker.py` | subprocess entry — `argv = (params_json, problem_path, timeout_s)` → JSON line on stdout |
| `evolve/phase{N}_<name>/initial_program.py` | one per phase (last phase is usually unified) |

Authoritative spec: [input/ADD_NEW_SOLVER.md](../../../input/ADD_NEW_SOLVER.md). Reference implementations: [input/z3-bench/evolve/](../../../input/z3-bench/evolve/) (flat-overrides + speedup), [input/cpsat-bench/evolve/](../../../input/cpsat-bench/evolve/) (SIZE_BUCKETS + worker-axis + cost mode).

Reading order before scaffolding: `references/interview-checklist.md` → `references/decision-guide.md` → `references/verify-checklist.md` → `references/gotchas.md`.

## Workflow

### 1. Inspect what user provided

```bash
ls input/<bench>/raw-data/ | head
ls input/<bench>/problems.jsonl 2>/dev/null && head -1 input/<bench>/problems.jsonl | python3 -m json.tool
```

Confirm presence of `raw-data/` and `problems.jsonl`. If `problems.jsonl` missing, stop and tell user — `_lib/sampler.py` only reads it (does not build it). Generation is solver-specific user work (ADD_NEW_SOLVER.md §1.4). Offer to draft a `build_problems_jsonl.py` after they describe raw-data layout.

If <10 problems: warn that quintile clustering collapses and cascade stages become noisy. Recommend 1–2 phases only.

### 2. Interview

Use `AskUserQuestion` (≤4 per call). Must-know set in [references/interview-checklist.md](references/interview-checklist.md). Skip anything already obvious from `problems.jsonl` + raw-data inspection.

Highest-leverage answers:
1. Solver binary / Python binding and how to invoke it.
2. `problems.jsonl` field names → adapter constants (`PROBLEM_FILE_FIELD`, `STATUS_FIELD`, `STATS_FIELD`, `FEATURES_FIELD`, `OBJECTIVE_FIELD`).
3. Decisive vs. decided result tokens.
4. Score mode: `speedup` (z3-style: wall-clock min) vs `cost` (cpsat-style: objective gap + dtime). See [references/decision-guide.md](references/decision-guide.md).
5. Worker-count axis? If yes → `WORKERS_KEY`, phase-level worker lock.
6. SIZE_BUCKETS / STAGE3_OVERRIDES needed? Default off; turn on when problem-size distribution is wide and multi-modal.
7. Phase plan: how many phases, namespace per phase, last phase unified (yes/no).
8. Clustering: `kmeans` (default) | `quintile` | `thresholds`; feature path inside `problems.jsonl`.

### 3. Scaffold

Write 4 files + N phase dirs. Substitute placeholders from interview into [templates/](templates/):

```
input/<bench>/evolve/
├── config.yaml                       ← templates/config.yaml.tmpl
├── params.json                       ← templates/params.json.tmpl (skeleton; expand groups)
├── adapter.py                        ← templates/adapter.py.tmpl
├── _solve_worker.py                  ← templates/_solve_worker_py.tmpl  (or _solve_worker_cli.tmpl for binary)
├── phase1_<name>/initial_program.py  ← templates/initial_program_simple.py.tmpl
├── phase2_<name>/initial_program.py  ← templates/initial_program_simple.py.tmpl
│                                       (or _cpsat.py.tmpl if SIZE_BUCKETS/worker lock)
└── phaseN_unified/initial_program.py ← templates/initial_program_unified.py.tmpl
```

Phase modules must use `params_catalog.load_for_bench(_BENCH).defaults` for BASELINE — never hardcode a parallel default dict. Config single-source rule: see [[feedback_config_single_source]].

`unified_dict_name` in `config.yaml` MUST match the EVOLVE-BLOCK dict name in the last phase (default: `UNIFIED_OVERRIDES`). Mismatch → `_lib.prepare_phase` fails silently.

### 4. Verify (run these — do not skip)

```bash
# 0. Solver binding installed?
python3 -c "import <solver_pkg>; print(<solver_pkg>.__version__)"   # or `command -v <binary>`

# 1. Catalog load + validation
python3 -c "
import sys; sys.path.insert(0, 'input')
from _lib import params_catalog
c = params_catalog.load('input/<bench>/evolve/params.json')
print('keys:', len(c.known_keys()), 'defaults:', len(c.defaults), 'locked:', len(c.locked))
print('validate ok:', c.validate(c.defaults))
print('validate bogus:', c.validate({'fake_key': 1}))
"

# 2. Clustering + stage split
cd input && python3 -m _lib.sampler <bench>
# Expect cache/stage{1..4}_sample.json. Inspect cluster sizes — all-in-one cluster
# = features field path wrong or every problem has 0.

# 3. BASELINE sanity on stage1
python3 -m _lib.self_test <bench>
# Expect result labels match + ratio in [0.5, 2.0] (WARN tolerated).

# 4. Local baseline capture (10-run avg) — slow
python3 -m _lib.rebaseline <bench>
# Expect cache/local_baseline.json.

# 5. Single-phase smoke (low iter)
./input/run_phase.sh <bench> 1 --pin 2-3 --iterations 2
# Expect phase1/openevolve_output/best/best_program.py + cache/phase1_best.json.
```

Each verify step gates the next. Stop and fix at the first failure — do not proceed to the next CLI hoping it will surface a clearer error.

### 5. Hand off

Report to user:
- 4 files + N phase modules created at the paths above.
- Verification results (catalog key count, sampler cluster sizes, self_test ratio).
- Next command: `./input/run_phase.sh <bench> --pin <core-range>` for full chain.
- `final_program.py` will land at `input/<bench>/evolve/final_program.py` after last phase (auto via `_lib.finalize`). If `bench.solver_mode` is set to a non-default variant, output suffixes to `final_program_<mode>.py` and artifacts isolate to `cache-<mode>/`.

## Refuse / push back

- `input/<bench>/raw-data/` absent → ask where dataset lives. Do not invent a layout.
- `problems.jsonl` absent → solver-specific generation is **user** work (ADD_NEW_SOLVER.md §1.4). Optionally help draft `build_problems_jsonl.py` once they describe meta format.
- Solver has no Python binding AND no CLI that accepts a problem file → ask user how they invoke it; cannot write `_solve_worker.py` without this.
- Fewer than ~10 problems → warn about cascade collapse; suggest single phase.
- User asks to edit `input/_lib/*` → refuse. `_lib` is bench-agnostic; per-bench knobs go in the 4 files only.

## Gotchas

See [references/gotchas.md](references/gotchas.md) — verbatim copy of ADD_NEW_SOLVER.md §6 plus failure-mode index from verify steps.
