"""Read-only inputs for user-defined island selection policies."""

from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass(frozen=True)
class IslandState:
    """Current population and fitness summary for one island."""

    population_size: int
    best_score: float
    average_score: float
    diversity: float
    generation: int


@dataclass(frozen=True)
class IslandSelectionContext:
    """Snapshot passed to a selector before an iteration is submitted."""

    iteration: int
    pending_counts: Tuple[int, ...]
    islands: Tuple[IslandState, ...]


IslandSelector = Callable[[IslandSelectionContext], int]
