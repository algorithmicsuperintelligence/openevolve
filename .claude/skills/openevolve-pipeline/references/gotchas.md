# Gotchas

ADD_NEW_SOLVER.md §6 — verbatim. Re-listed here for skill-local lookup.

## 1. problems.jsonl field name mismatch

`adapter.PROBLEM_FILE_FIELD` / `STATUS_FIELD` MUST match the JSON keys exactly.
Typos are the #1 failure mode. `head -1 input/<bench>/problems.jsonl | python3 -m json.tool`
shows the canonical keys.

## 2. `features.<feature>` missing

`clustering.feature: features.num_X`, but problems.jsonl entries lack
`features.num_X` → sampler treats every problem as size=0 → all problems
collapse into one cluster → cascade stages become meaningless.

Verify with the sampler stdout: each cluster should hold a recognizable
spread of problem SHAs. All-in-one cluster = features field path wrong.

## 3. DECISIVE vs DECIDED confusion

- DECISIVE = "solver gave an answer" (e.g. `Sat`, `Unsat`, `OPTIMAL`).
- DECIDED  = "baseline produced a conclusive answer → regression comparable"
  (e.g. cpsat: `INFEASIBLE` decided but only `OPTIMAL`/`FEASIBLE` decisive).

Most solvers: both sets identical. cpsat: they differ.

## 4. `_solve_worker.py` doesn't surface invalid params

If solver silently ignores unknown keys, the catalog alone cannot catch a
mutated illegal key. Worker MUST emit
`{"invalid_param": "<key>", "result": "Unknown", "elapsed_ms": 0}` when
solver rejects a key — otherwise evaluator cannot 0-score the candidate.

Test: pass `{"obviously_fake_key": 1}` → worker should emit invalid_param.

## 5. Phase docstring empty

LLM has no other signal about phase intent. Even one line — "Phase 2: tune
presolve.* knobs" — improves mutation quality dramatically.

## 6. `unified_dict_name` mismatch

`config.yaml` `bench.unified_dict_name` MUST match the EVOLVE-BLOCK dict
name in the last phase's `initial_program.py`. Default convention:
`UNIFIED_OVERRIDES`.

Mismatch → `_lib.prepare_phase` cannot materialize the union → last phase
starts empty and loses prior-phase wins.

## 7. `worker_path` is relative to `<bench>/evolve/`

`config.yaml` `bench.worker_path: _solve_worker.py` (no directory prefix
when worker is at evolve/ root).

## Verify-time additional gotchas

### v1. `OPENEVOLVE_BENCH_ROOT unset` from phase module

Phase module `_resolve_bench_root()` fallback walks parents looking for
adapter+params.json. Fails if phase dir is not exactly two levels under
`<bench>`. Correct layout:

```
input/<bench>/evolve/params.json
input/<bench>/evolve/adapter.py
input/<bench>/evolve/phase1_x/initial_program.py    ← two levels under <bench>
```

### v2. Cascade thresholds too tight

`evaluator.cascade_thresholds: [1.03, 1.03, 1.03]` means each stage demands
≥3% improvement. Solver with high variance + few problems may never cross.
Lower to `[1.01, 1.01, 1.01]` for noisy benches.

### v3. `parallel_solvers > 1` with single-threaded baseline

If baseline was captured single-threaded, running candidates with
`parallel_solvers: N` co-locates them on shared cores → timings inflate
vs baseline → false regression. Either pin core ranges via `--pin` or
recapture baseline at the same parallelism.

### v4. Catalog `defaults` ≠ binding's real defaults

If `params.json` `defaults` includes a key the binding's real default is
different, the BASELINE phase modules send may diverge from what `_lib.rebaseline`
captured. Symptom: `self_test` ratio drifts outside [0.5, 2.0].

Fix: ensure `defaults` is what the original problems.jsonl baseline run used.
