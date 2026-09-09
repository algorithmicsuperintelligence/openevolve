"""Query-oriented interface for an evolution session's program database.

Implementations own selection and population state. Consumers receive individual
programs, bounded selections, or aggregate state; they never access the backing
collections. The interface has no save/load or checkpoint operations.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

from openevolve.program import Program


@dataclass(frozen=True)
class DatabaseState:
    """Aggregate session state, independent of the storage implementation."""

    program_count: int
    last_iteration: int
    current_island: int
    num_islands: int
    feature_dimensions: Tuple[str, ...]


@runtime_checkable
class ProgramDatabase(Protocol):
    """Search and result operations implemented in memory or by a DBMS.

    Each instance addresses one evolution session. Returned programs are detached
    values: changing one does not update the database. Writes must be explicit.
    Population and migration policy run behind these operations so a DBMS can
    perform them without constructing an authoritative Python population.
    """

    def get_state(self) -> DatabaseState:
        """Read aggregate progress and the session's island/feature definitions."""
        ...

    def record_iteration(self, iteration: int) -> None:
        """Record a processed iteration, including one that produced no program."""
        ...

    def add(
        self, program: Program, iteration: Optional[int] = None, target_island: Optional[int] = None
    ) -> str:
        """Insert a program and apply the session's population/elite policy."""
        ...

    def get(self, program_id: str) -> Optional[Program]:
        """Look up a program by ID."""
        ...

    def sample_from_island(
        self, island_id: int, num_inspirations: Optional[int] = None
    ) -> Tuple[Program, List[Program]]:
        """Select a parent and bounded inspirations using the search policy."""
        ...

    def get_best_program(self, metric: Optional[str] = None) -> Optional[Program]:
        """Select the best program by fitness or a specified metric."""
        ...

    def get_top_programs(
        self, n: int = 10, metric: Optional[str] = None, island_idx: Optional[int] = None
    ) -> List[Program]:
        """Select at most n programs in descending fitness/metric order."""
        ...

    def get_island_stats(self) -> List[Dict[str, Any]]:
        """Return aggregate statistics, not island populations."""
        ...

    def increment_island_generation(self, island_idx: Optional[int] = None) -> None:
        """Advance the generation counter for an island."""
        ...

    def should_migrate(self) -> bool:
        """Query whether the configured migration interval has elapsed."""
        ...

    def migrate_programs(self) -> None:
        """Select and migrate programs, updating population and elite state."""
        ...

    def store_artifacts(self, program_id: str, artifacts: Dict[str, Union[str, bytes]]) -> None:
        """Attach evaluation evidence to a program."""
        ...

    def get_artifacts(self, program_id: str) -> Dict[str, Union[str, bytes]]:
        """Fetch evidence for one program."""
        ...

    def log_prompt(
        self,
        program_id: str,
        template_key: str,
        prompt: Dict[str, str],
        responses: Optional[List[str]] = None,
    ) -> None:
        """Record the prompt and responses for a program."""
        ...

    def get_prompt_history(self, program_id: str) -> Dict[str, Any]:
        """Fetch the recorded prompts and responses for one program."""
        ...
