# Interview Checklist

Ask **only what's not already inferable** from `problems.jsonl` + `raw-data/`.
Batch with `AskUserQuestion` (≤4 per call). Group by topic.

## Must-know (block scaffolding without answers)

### Solver
- Solver name + version (→ `params.json` `solver`/`version`; `adapter.SOLVER_NAME`).
- Invocation path: Python binding (`import <pkg>`) OR CLI binary (`/path/to/solver`).
  - If neither — STOP. Cannot write `_solve_worker.py`.
- Install hint command (→ `config.yaml` `bench.solver_install_hint`).

### problems.jsonl field mapping
Read one record (`head -1 input/<bench>/problems.jsonl | python3 -m json.tool`)
and confirm with user:
- Problem file field name (e.g. `smt2_filename`, `problem_filename`) → `PROBLEM_FILE_FIELD`.
- Status block key (e.g. `z3_status`, `cpsat_status`) → `STATUS_FIELD`.
  - Inside it: `result`, `elapsed_ms`, optional `objective_value`.
- Stats block key (or `None` if baseline has no stats) → `STATS_FIELD`.
- Features block key (default `features`) → `FEATURES_FIELD`.
- Objective path inside status (or `None` for SAT/UNSAT) → `OBJECTIVE_FIELD`.

### Result tokens
- DECISIVE_RESULTS: which result strings mean "solver gave an answer"?
  - z3: `("Sat", "Unsat")`. cpsat: `("OPTIMAL", "FEASIBLE")`.
- DECIDED_RESULTS: which baseline results allow regression comparison?
  - Often == DECISIVE, but cpsat splits: `INFEASIBLE` decided but not feasible-decisive.

### Score mode
- `speedup` (z3-style): minimize wall-clock; objective unused.
- `cost` (cpsat-style): minimize `(objective_value, dtime)` against baseline; needs a determinism counter (`time_metric`).
- See [decision-guide.md](decision-guide.md).

### Worker axis
- Does the solver have a `num_workers`-like knob that meaningfully changes runtime?
  - YES → `WORKERS_KEY = "<key>"`; phase modules lock it via `PHASE_LOCKED`.
  - NO → `WORKERS_KEY = None`.

### Phase plan
- How many phases? (Typical: 3–5.)
- What namespace per phase? (e.g. phase1 = search, phase2 = presolve.)
- Last phase unified (merges priors)? (Default yes.)

## Should-know (sensible defaults if user skips)

### Clustering
- `clustering.method`: kmeans (default) | quintile | thresholds.
- `clustering.feature`: which numeric field to cluster on. Common: `features.num_constraints`, `<solver>_status.elapsed_ms`.
- `clustering.n_clusters`: 5 (default).
- `clustering.max_baseline_ms`: drop outliers above this from sample pool. 300000 (5min) default.
- `clustering.stage_sizes`: how many problems per cascade stage. Default `{stage1:10, stage2:10, stage3:5, stage4:20}`.

### Evaluation
- `evaluation.repeats`: 10 (standard; lowering speeds up but adds noise).
- `evaluation.timeout_factor`: 1.3 (per-problem budget = baseline_ms × this).
- `evaluation.enable_size_buckets`: false default. true → use `initial_program_cpsat.py.tmpl`.
- `evaluation.enable_outlier_stage`: false default.

### KEY_STATS / STATS_WEIGHTS
- Which counters from `solver.statistics()` to surface as metrics?
- Weight per counter for efficiency factor (defaults: conflicts 2.0, decisions 1.5, propagations 0.5).

### LLM
- Which model? Default `claude-sonnet-4-6` via `claude_code` provider.

## Sensible defaults — do NOT ask

- `parallel_solvers: 1`, `max_iterations: 40`, `checkpoint_interval: 10`,
  `random_seed: 42`, `num_islands: 3`, `cascade_thresholds: [1.03, 1.03, 1.03]`,
  `evaluator.timeout: 1800`.
