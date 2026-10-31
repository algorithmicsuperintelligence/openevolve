# Competitive resource allocation

This example alternates two OpenEvolve populations: an attacker chooses an entry
channel, and a defender allocates a fixed inspection budget across three channels.
It illustrates the competition requested in [issue #311](https://github.com/algorithmicsuperintelligence/openevolve/issues/311).

Training against one fixed attack can concentrate all inspections on that channel.
An evolving attacker can choose an unprotected channel instead. Retaining previous
opposing champions gives the defender a reason to cover earlier attacks as well as
the latest one.

## Run without an API key or model

From the repository root, with OpenEvolve installed:

```bash
python examples/competitive_resource_allocation/compare.py \
  --output /tmp/openevolve-competition --seeds 0 1 2
```

Use a new output directory for each comparison. The script starts a local HTTP
endpoint, calls the actual `run_evolution` and `run_coevolution` APIs, and uses their
ordinary process workers. The endpoint proposes Python programs from a shuffled
grid of allocations. It is a scripted generator, not an LLM; there are no external
model calls, downloads or GPU requirements.

The candidate grid contains the 28 allocations with weights in multiples of 1/6.
The default attacker chooses one of the three individual channels. Each method
uses the same initial policies and the same candidate sequence for each population
and seed. Competition runs three rounds, with 28 proposals per population per
round. Fitness is the minimum payoff against the phase's frozen opponent cohort:
the defender maximizes detection, and the attacker maximizes evasion.

The report includes two unmodified single-population baselines:

- `static_generation_matched`: each population gets the same 84 proposals, but
  always evaluates against the initial opposing policy.
- `static_larger_budget`: each population gets enough extra proposals to exceed
  competition's measured number of individual matches, including all population
  refreshes and cross-matches. The script checks that the baseline actually spent
  at least that many matches; unchanged proposals can be skipped by the engine.

Both generation requests and individual matches are reported, rather than treating
one evaluation against four opponents as equal to one evaluation against one.
Final defender policies are assessed against all three pure attacks, independently
of which attacks appeared during training. The primary metric is the minimum
detection probability across those channels. Mean detection is also reported; its
value is always 1/3 in this fixed-budget game, so this example cannot demonstrate
an increase in mean detection.

The mixed-attacker variant is a separate control:

```bash
python examples/competitive_resource_allocation/compare.py \
  --output /tmp/openevolve-mixed-competition --seeds 0 1 2 --mixed-attacker
```

It gives the attacker the same allocation grid as the defender. A mixture can
become hard to evade without exposing each individual blind spot, so improvements
in worst-channel detection are not guaranteed. This example demonstrates selection
under changing opponents in a small, exhaustively proposed policy space. It does
not measure LLM capability, deployed AI-control effectiveness, convergence to a
Nash equilibrium or training performance.

## Observed comparison

The checked-in [results](results.json) were produced on macOS arm64, Python 3.12.3,
with the source hashes recorded in that file. For seeds 0, 1 and 2, both attacker
variants reached a defender allocation of `(1/3, 1/3, 1/3)`. Both static baselines
retained `(1, 0, 0)` in all six comparisons. Thus worst-channel detection was 1/3
versus 0, while mean detection remained 1/3 for every method.

| Attacker | Seed | Competition matches | Larger static baseline matches |
| --- | ---: | ---: | ---: |
| Single channel | 0 | 561 | 940 |
| Single channel | 1 | 567 | 955 |
| Single channel | 2 | 579 | 969 |
| Mixed | 0 | 626 | 1210 |
| Mixed | 1 | 633 | 1218 |
| Mixed | 2 | 625 | 1210 |

Counts include both populations. Competition and the generation-matched baseline
each requested 168 proposals in total. Match counts can vary with engine selection
and skipped unchanged proposals. An earlier exploratory mixed-attacker run of the
development implementation produced no worst-channel improvement (0 versus 0);
it predates the stable archive-rebuild ordering and is not included in the final
table. The finite example is not a convergence guarantee or an unseen-data test.

## Use other programs and evaluators

`run_coevolution` accepts two named populations in their move order. Each has its
own initial program file and `Config`, including its own models, prompts, islands
and population limits. A round runs one phase for each population.

```python
from pathlib import Path

from openevolve import Population, run_coevolution

# blue_config and red_config are ordinary, independently configured Config objects.
# evaluate_match is supplied by your task; OpenEvolve does not define the game.
def evaluate(program_path, population_name, opponents):
    scores = [evaluate_match(program_path, population_name, opponent.code)
              for opponent in opponents]
    return {"combined_score": min(scores)}

result = run_coevolution(
    {
        "blue": Population(Path("blue.py"), blue_config),
        "red": Population(Path("red.py"), red_config),
    },
    evaluate,
    output_dir="competition_output",
    evaluation_id="control-task-v1-dataset-v1",
    rounds=10,
    iterations_per_phase=20,
    opponent_count=4,
)
```

The callback must be serializable by cloudpickle and return a finite
`combined_score`, with larger values better for the population being evaluated.
Each `Opponent` contains an ID, source code and language. The task owns execution,
aggregation and any evaluator randomness. Ordinary engine evaluation timeout and
retry settings still apply; cascade stage functions are not part of this callback.
An `evaluation_id` must identify the callback's scoring definition and task data,
including changes to dependencies or external state that affect fitness.

## Phase and resume semantics

1. Freeze the other population's most recent distinct champions, up to
   `opponent_count`. Initially its seed is the only opponent. A repeated champion
   refreshes recency without increasing its weight; older entries eventually expire.
2. Re-evaluate every program retained by the active population against that cohort.
   Rebuild its MAP-Elites grids, elite archive, feature statistics and best-program
   references from the new scores. Old evaluation artifacts and prompts are dropped.
3. Run the existing engine for up to `iterations_per_phase` proposals. All workers
   receive the same frozen opponents. Normal admission and eviction rules apply;
   refreshing the population can therefore remove previous cell occupants.
4. When the engine returns successfully, commit the resulting population, champion
   history and phase cursor together in `coevolution.json`. Internal island
   generation/migration counters and lineage are retained across phases. Programs
   never migrate between the two opposing populations.

Call again with `resume=True` and the same settings to continue. `rounds` is the
total target, including completed rounds; it can be increased. Seed code, file
suffix, population order, configurations, opponent count, phase length and
evaluation identity are checked before resuming. API keys can rotate. This API
owns population storage, so `Config.database.db_path` must be unset.

If a phase raises an exception, the previous committed state remains valid. A
resume repeats that phase from its previous population and the same opposing
champion cohort. Partially completed worker output is kept in a separate attempt
directory for inspection and is not resumed. External evaluations or generation
calls may repeat. An engine that stops early and returns normally commits its
returned population, just as a normally completed phase does.

Only one caller may write an output directory. This is phase-level recovery, not
an exactly-once protocol for external services or a durability guarantee against
power failure. Evaluation wrappers contain trusted Python pickles, and evolved
programs run under the same trust assumptions as existing OpenEvolve evaluations.

Returned champions' metrics belong to their respective last opponent cohorts.
Their scores are not directly comparable across phases or populations. For a
progress curve, re-evaluate saved champions against a separately fixed assessment
set, as the example does for the final defender.
