# Axelrod Tournament Strategy Evolution

> **Built on the [Axelrod Python library](https://github.com/Axelrod-Python/Axelrod)**
> — an open-source framework for research into the iterated Prisoner's Dilemma.
> The tournament infrastructure, strategy implementations, and game mechanics used
> in this example are all provided by that project. The seed strategies in this
> example were adapted from strategies in the Axelrod library. See the `NOTICE`
> file in this directory for license details.

This example uses OpenEvolve to evolve a strategy for the iterated Prisoner's Dilemma, inspired by [Robert Axelrod's famous tournaments](https://en.wikipedia.org/wiki/The_Evolution_of_Cooperation) from the 1980s. The goal is to discover a strategy that maximizes its average score when competing against the full suite of Axelrod library strategies.

## Problem Overview

In the [Prisoner's Dilemma](https://en.wikipedia.org/wiki/Prisoner%27s_dilemma), two players simultaneously choose to either **Cooperate (C)** or **Defect (D)**. The payoff matrix rewards mutual cooperation, but each player is individually tempted to defect.

In the *iterated* version, players play many rounds against each other. Strategies that build cooperation tend to do well over time, but must also protect against exploitation by purely selfish strategies.

The evolved strategy plays a one-vs-all tournament against every strategy in the `axelrod.short_run_time_strategies` collection (over 100 strategies). It competes against each one in a separate match and is scored on its **median score** across all matchups.

## Evaluation

The evaluator (`evaluator.py`) works as follows:

1. Loads the evolved strategy as a class named `Evolvo`
2. Constructs a tournament where `Evolvo` plays every strategy in `axl.short_run_time_strategies`
3. Plays all matches (1 repetition per pair, no round-robin among opponents)
4. Returns the `Median_score` as the primary metric (`combined_score`)

Additional artifacts are tracked for each evaluation:
- **Cooperation_rating**: How often the strategy cooperates overall
- **Wins**: Number of outright wins
- **CC_rate / CD_rate / DC_rate / DD_rate**: Frequencies of each outcome pair
- **CC_to_C / CD_to_C / DC_to_C / DD_to_C**: Conditional cooperation rates after each prior outcome

These artifacts are fed back into the LLM prompt (via `include_artifacts: true`) so the LLM can see exactly how the strategy is behaving and make informed improvements.

## Seed Strategies

Rather than starting from a single naive strategy, this example seeds 4 diverse starting strategies—one per island—covering a range of well-known approaches. Each was adapted from the Axelrod library with the evolve block wrapping the class body so OpenEvolve can mutate the strategy.

| File | Original Strategy | Key Behavior |
|------|------------------|--------------|
| `seed1.py` | Tit-for-Tat | Cooperate first; mirror the opponent's last move |
| `seed2.py` | ForgetfulFoolMeOnce(0.05) | Forgive one defection; retaliate forever on a second; occasionally forget with 5% probability |
| `seed3.py` | OriginalGradual | Escalate punishment with each defection; enter a calming state to de-escalate |
| `seed4.py` | SecondByBorufsen | Track opponent patterns; switch to defect mode against random or defective opponents; detect mutual-defect cycles and alternating echoes |

These seeds were chosen to be near-top-level performers that are straightforward to implement, while remaining diverse enough to explore different regions of strategy space.

## Island-Based Seeding

The example uses a custom runner (`run_evolve.py`) rather than the standard `openevolve-run.py`. This allows each seed to be evaluated and placed directly onto a specific island before evolution begins:

```python
for i, seed_path in enumerate(initial_seeds):
    with open(seed_path, 'r') as f:
        code = f.read()
        metrics = await openevolve.evaluator.evaluate_program(code, initial_program_id)
        program = Program(id=..., code=code, metrics=metrics, ...)
        openevolve.database.add(program, target_island=i)
```

This ensures each island starts with a strong, distinct strategy—promoting diversity throughout the evolutionary process and preventing premature convergence to a single approach.

## Running the Example

Install the dependency and run from the repository root:

```bash
pip install axelrod
python examples/axelrod/run_evolve.py
```

## Configuration

The evolution is configured in `config.yaml`:

```yaml
max_iterations: 1000
checkpoint_interval: 10

llm:
  primary_model: "gemini-2.5-flash-lite"   # Fast, cheap model for most mutations
  primary_model_weight: 0.8
  secondary_model: "gemini-2.5-flash"       # Stronger model for refinement
  secondary_model_weight: 0.2
  temperature: 0.7
  max_tokens: 16000

prompt:
  num_top_programs: 3
  num_diverse_programs: 2
  include_artifacts: true   # Feed cooperation stats back to LLM
  system_message: "You're building a program that runs iterated prisoner's dilemma
    in a tournament with many other strategies, similar to Axelrod's famous tournaments.
    Your task is to improve the play strategy for maximum profit."

database:
  population_size: 50
  num_islands: 4
  elite_selection_ratio: 0.2
  exploitation_ratio: 0.7
  similarity_threshold: 0.99

evaluator:
  enable_artifacts: true
  timeout: 60
  parallel_evaluations: 3
```

Key configuration decisions:
- **4 islands** to match the 4 seed strategies, maintaining diversity throughout evolution
- **Artifacts enabled** so the LLM can reason about cooperation patterns and win/loss rates
- **Similarity threshold of 0.99** to discourage near-duplicate strategies from accumulating in the population

## Evolved Strategy: AdaptiveCycleBreakerTFT

The best evolved strategy (`best_program.py`) is named `AdaptiveCycleBreakerTFT`. It is a sophisticated refinement of the SecondByBorufsen seed, with several key changes:

**Mode Switching (every 15 turns instead of 25):**
- Evaluates opponent behavior more frequently, allowing faster adaptation
- Switches to permanent **Defect mode** if the opponent cooperated fewer than 3 times, or if less than 83% of their cooperations were in response to the player's own cooperation (stricter than the original 70% threshold)

**Cycle-Breaking in Normal Mode (triggers at streak ≥ 2 instead of 3):**
- If both players have mutually defected for 2+ consecutive turns, cooperate once to break the cycle
- If an alternating D/C echo pattern persists for 2+ turns, flag the next defect to become a cooperation

**Tit-for-Tat as Default:**
- When none of the special conditions apply, simply mirror the opponent's last move

The strategy is fully deterministic (`stochastic: False`) and has a declared `memory_depth` of 15, matching its re-evaluation period.

```python
class AdaptiveCycleBreakerTFT(Player):
    def strategy(self, opponent: Player) -> Action:
        # Re-evaluate mode every 15 turns
        if turn > 1 and turn % 15 == 0:
            if self.opp_coops < 3 or self.cc_counts / self.opp_coops < 0.83:
                self.mode = "Defect"
            else:
                self.mode = "Normal"

        if self.mode == "Defect":
            return D

        # Break mutual defection cycles
        if self.mutual_defect_streak >= 2:
            return C

        # Flag alternating echo patterns
        if self.echo_streak >= 2:
            self.flip_next_defect = True

        # Tit-for-tat (with optional flip)
        return C if opponent.history[-1] == C else (C if self.flip_next_defect else D)
```

## Validating the Best Program

After evolution completes, you can run a full round-robin validation tournament:

```bash
python examples/axelrod/validation.py
```

This runs `AdaptiveCycleBreakerTFT` against all `short_run_time_strategies` in a proper round-robin tournament (all vs. all) with 5 repetitions, giving a more robust estimate of its ranking.

## Key Observations

1. **Diversity through seeding**: Starting with 4 distinct strategies on separate islands prevented premature convergence and gave the LLM a broader set of patterns to recombine and refine.

2. **Artifact-driven learning**: Feeding cooperation statistics (CC_rate, DC_rate, etc.) back to the LLM allowed it to reason about *why* a strategy was failing—e.g., too cooperative against defectors, or failing to re-establish cooperation after mutual defection.

3. **Convergence toward known principles**: The evolved strategy independently rediscovered key insights from decades of game theory research—the value of proportional retaliation, the need to detect and escape mutual defection traps, and the importance of distinguishing random from deterministically defective opponents.

4. **Parameter refinement**: The evolved strategy tightened the SecondByBorufsen parameters (25→15 turn evaluation period, 70%→83% cooperation threshold, streak threshold 3→2), suggesting the LLM learned to be more responsive and less forgiving than the original.
