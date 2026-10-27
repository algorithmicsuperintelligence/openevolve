# OpenEvolve Configuration Files

This directory contains configuration files for OpenEvolve with examples for different use cases.

## Configuration Files

### `default_config.yaml`
The main configuration file containing all available options with sensible defaults. This file includes:
- Complete documentation for all configuration parameters
- Default values for all settings
- **Island-based evolution parameters** for proper evolutionary diversity

Use this file as a template for your own configurations.

### `island_config_example.yaml`
A practical example configuration demonstrating proper island-based evolution setup. Shows:
- Recommended island settings for most use cases
- Balanced migration parameters
- Complete working configuration

### `island_examples.yaml`
Multiple example configurations for different scenarios:
- **Maximum Diversity**: Many islands, frequent migration
- **Focused Exploration**: Few islands, rare migration  
- **Balanced Approach**: Default recommended settings
- **Quick Exploration**: Small-scale rapid testing
- **Large-Scale Evolution**: Complex optimization runs

Includes guidelines for choosing parameters based on your problem characteristics.

## Island-Based Evolution Parameters

The key new parameters for proper evolutionary diversity are:

```yaml
database:
  num_islands: 5                      # Number of separate populations
  migration_interval: 50              # Migrate every N generations  
  migration_rate: 0.1                 # Fraction of top programs to migrate
  # island_initial_programs: [null, "seeds/greedy.py"]  # Optional per-island starting programs
```

### Parameter Guidelines

- **num_islands**: 3-10 for most problems (more = more diversity)
- **migration_interval**: 25-100 generations (higher = more independence)
- **migration_rate**: 0.05-0.2 (5%-20%, higher = faster knowledge sharing)
- **island_initial_programs**: Start islands from different programs, e.g. distinct
  algorithmic approaches. Entry *i* is the program island *i* starts from; `null` or a
  missing entry keeps the default (the island starts empty and takes its first parent
  from the other islands). Entry 0 must be `null`, because island 0 always starts from
  the initial program given on the command line, and the list may not be longer than
  `num_islands`. Relative paths resolve against the config file's directory. Each seed is
  evaluated once at the start of a fresh run; resuming from a checkpoint does not add
  them again.

### When to Use What

- **Complex problems** → More islands, less frequent migration
- **Simple problems** → Fewer islands, more frequent migration
- **Long runs** → More islands to maintain diversity
- **Short runs** → Fewer islands for faster convergence

## Usage

Copy any of these files as a starting point for your configuration:

```bash
cp configs/default_config.yaml my_config.yaml
# Edit my_config.yaml for your specific needs
```

Then use with OpenEvolve:

```python
from openevolve import OpenEvolve
from openevolve.config import Config

config = Config()
config.from_yaml("my_config.yaml")
evolve = OpenEvolve(
    initial_program_path="program.py",
    evaluation_file="evaluator.py", 
    config=config
)
```
