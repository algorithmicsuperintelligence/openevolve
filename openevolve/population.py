"""Decision hooks for population management.

Strategies inspect detached snapshots and return decisions. ProgramDatabase owns
all changes to programs, islands, feature maps, and the archive.
"""

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ProgramState:
    id: str
    code: str
    metrics: Mapping[str, Any]
    generation: int
    parent_id: Optional[str]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class PopulationSnapshot:
    programs: Mapping[str, ProgramState]
    islands: Tuple[frozenset[str], ...]
    feature_maps: Tuple[Mapping[str, str], ...]
    archive: frozenset[str]
    best_program_id: Optional[str]
    population_limit: int
    archive_limit: int
    feature_dimensions: Tuple[str, ...]
    generations: Tuple[int, ...]
    last_migration_generation: int
    migration_interval: int
    migration_rate: float
    last_iteration: int


@dataclass(frozen=True)
class ArchiveDecision:
    add: bool
    evict_id: Optional[str] = None


@dataclass(frozen=True)
class MigrationMove:
    program_id: str
    target_island: int


@dataclass(frozen=True)
class PopulationStrategy:
    """Override only the decisions needed; omitted hooks use existing rules.

    ``admit`` may reject a candidate after the initial seed, but cannot bypass
    database novelty checks.
    ``evict`` must return exactly the required number of eligible program IDs.
    ``migrate`` returns moves; the database copies programs and checks duplicates.
    """

    admit: Optional[Callable[[PopulationSnapshot, ProgramState, int], bool]] = None
    replace_cell: Optional[
        Callable[[PopulationSnapshot, ProgramState, ProgramState, int], bool]
    ] = None
    archive: Optional[Callable[[PopulationSnapshot, ProgramState], ArchiveDecision]] = None
    evict: Optional[Callable[[PopulationSnapshot, int, frozenset[str]], Sequence[str]]] = None
    migration_due: Optional[Callable[[PopulationSnapshot], bool]] = None
    migrate: Optional[Callable[[PopulationSnapshot], Sequence[MigrationMove]]] = None
