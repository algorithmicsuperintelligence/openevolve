# Decision Guide

ADD_NEW_SOLVER.md §5 — verbatim guidance for ambiguous knobs.

## score_mode

| Solver characteristic | Recommended mode |
|---|---|
| Baseline records `objective_value`; optimization problem | `cost` |
| Sat/Unsat satisfaction; minimize wall-clock | `speedup` |
| Has determinism counter (e.g. cpsat `deterministic_time`) | `cost` + `time_metric: dtime` |

`speedup` = geomean(baseline_ms / candidate_ms). Higher better.
`cost` = combination of objective-gap + dtime ratio. Lower better (inverted to speedup-like by scorer).

## Worker axis

If solver has a `num_workers`-style knob that strongly affects runtime:
- `adapter.WORKERS_KEY = "<key>"`.
- Each phase pins it via `PHASE_LOCKED["<key>"] = N`.
- `_lib.evaluator_core` uses core-block allocation.
- `_lib.rebaseline` produces `by_workers` baseline schema.

Else:
- `WORKERS_KEY = None`.
- One core per solve; flat baseline.

cpsat-bench uses W=1 (default profile) / W=8 (`OPENEVOLVE_PROFILE=large`)
across phases. z3-bench is single-threaded.

## SIZE_BUCKETS / STAGE3_OVERRIDES

| Situation | Toggle |
|---|---|
| Problem size spans wide range (e.g. 7k–250k constraints), multi-modal score distribution | `enable_size_buckets: true` |
| A few outlier problems dominate aggregate score | `enable_outlier_stage: true` + populate `cache/outliers.json` |
| Pool small (<30) or uniform | Both `false` |

`enable_size_buckets: true` → phase modules must use `initial_program_cpsat.py.tmpl`
(SIZE_BUCKETS + `get_phase_size_buckets()`).
`enable_outlier_stage: true` → add `STAGE3_OVERRIDES` + `get_phase_stage3_overrides()`.

## clustering.method

| Method | When |
|---|---|
| `kmeans` | 1D Lloyd's. Lets cluster boundaries emerge from data shape. Default. |
| `quintile` | Rank-based equal-count splits. Use when boundary consistency across runs matters more than natural breaks. |
| `thresholds` | User-specified cut-offs (e.g. `[50000, 150000]` → 3 buckets). Use when you have domain knowledge of regimes. |

## Existing solver reference

| Solver | score_mode | Worker axis | Size buckets | Phases |
|---|---|---|---|---|
| z3 (`z3-bench`) | speedup | NO | NO | 4 (opt_sls + sat + smt + unified) |
| CP-SAT (`cpsat-bench`) | cost (dtime + cost_ratio) | YES (W=1, W=8) | YES | 5 (search + presolve + lp_cuts + unified + custom_subsolvers) |

Use whichever matches the new solver's profile as the structural template.
