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

## clustering.mode (optional sample-profile override)

Generic `_lib.sampler` feature. `clustering.mode: <name>` selects a
`clustering.modes.<name>` block that is **shallow-merged over the base clustering
block**. Lets one config carry several sample profiles and switch by one field.
Unset → base block only.

- ORTHOGONAL to `solver_mode` — a sample profile (e.g. `large` = focus on
  constraint-heavy instances) applies in any solver mode. Do NOT key the override
  off solver_mode; keep the two knobs independent.
- Only the keys present in the override are replaced (e.g. just `method` +
  `thresholds` + `stage_sizes`); everything else falls through to base.
- z3-bench uses `modes.large` (threshold bucketing, top bucket only) to focus on
  the biggest instances when the speedup signal is dominated by them.

## solver_mode (optional variant suffix)

Generic `_lib.bench_paths` feature. `bench.solver_mode` (default unset ==
`optimize`) does two things:

1. **Artifact suffixing** so multiple modes coexist on disk:
   `cache/`+`final_program.py` (optimize) vs `cache-<X>/`+`final_program_<X>.py`.
   Every `_lib` CLI (sampler/rebaseline/extract_best/prepare_phase/final_verify/
   finalize) routes through `bench_paths.cache_dir` / `variant_suffix`, so the two
   modes' baselines and outputs never collide.
2. **Worker branching** — `_solve_worker.py` reads the SAME sibling `config.yaml`
   field and changes solver behavior (z3: `sat` = `z3.Solver` over
   `parse_smt2_file`, drops `assert-soft`, no objective, `opt.*` params silently
   dropped). `_lib.evaluator_core` warns if `optimize` + `score_mode != cost`.

Use when one workload has two ways to be solved (full optimize vs feasibility-only)
and you want both tunable without copying the bench dir. Switching = edit
`solver_mode` + `score_mode`, then re-run sampler/rebaseline/phases (per-mode
`cache-<X>/` keeps a dedicated baseline — **rebaseline is mandatory after switch**).

## Existing solver reference

| Solver | score_mode | Worker axis | Size buckets | Phases |
|---|---|---|---|---|
| z3 (`z3-bench`) | cost (optimize) / speedup (sat) | NO | NO | 4 (opt_sls + sat + smt + unified) |
| CP-SAT (`cpsat-bench`) | cost (dtime + cost_ratio) | YES (W=1, W=8) | YES | 5 (search + presolve + lp_cuts + unified + custom_subsolvers) |

z3-bench also demonstrates the optional `solver_mode` (optimize/sat) +
`clustering.mode` (base/large) knobs above — both default-off, both config-only.

Use whichever matches the new solver's profile as the structural template.
