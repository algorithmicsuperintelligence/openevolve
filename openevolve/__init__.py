"""
OpenEvolve: An open-source implementation of AlphaEvolve
"""

from openevolve._version import __version__
from openevolve.api import (
    EvolutionResult,
    evolve_algorithm,
    evolve_code,
    evolve_function,
    run_evolution,
)
from openevolve.config import Config
from openevolve.controller import OpenEvolve
from openevolve.population import (
    ArchiveDecision,
    MigrationMove,
    PopulationSnapshot,
    PopulationStrategy,
    ProgramState,
)
from openevolve.selection import IslandSelectionContext, IslandSelector, IslandState

__all__ = [
    "Config",
    "OpenEvolve",
    "ArchiveDecision",
    "MigrationMove",
    "PopulationSnapshot",
    "PopulationStrategy",
    "ProgramState",
    "IslandSelectionContext",
    "IslandSelector",
    "IslandState",
    "__version__",
    "run_evolution",
    "evolve_function",
    "evolve_algorithm",
    "evolve_code",
    "EvolutionResult",
]
