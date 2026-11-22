"""Logging helpers shared across OpenEvolve.

Provides:
- :data:`current_run_id`: Context variable tracking the currently active OpenEvolve run ID.
- :class:`RunIsolationFilter`: A handler-level filter that isolates log messages to their
  respective run context, preventing cross-writing and duplicate log emissions when multiple
  runs execute in parallel within the same Python process (GitHub issue #295).
- :class:`LoggerPrefixFilter`: A handler-level filter that rewrites ``record.name`` so every
  downstream module logger appears under a common prefix (GitHub issue #290).
"""

import contextvars
import logging
from typing import Optional

# Context variable holding the unique run ID for the currently executing OpenEvolve instance.
# In asynchronous (asyncio) or threaded workflows, context variables are copied per-task,
# allowing multiple concurrent runs in the same process to maintain their own isolated run ID.
current_run_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "openevolve_current_run_id", default=None
)


def get_current_run_id() -> Optional[str]:
    """Retrieve the currently active OpenEvolve run ID, or None if outside a run context."""
    return current_run_id.get()


def set_current_run_id(run_id: Optional[str]) -> contextvars.Token:
    """Set the active OpenEvolve run ID for the current execution context.

    Args:
        run_id: Unique string identifier for the run, or None to clear.

    Returns:
        A contextvars.Token that can be passed to reset_current_run_id.
    """
    return current_run_id.set(run_id)


def reset_current_run_id(token: contextvars.Token) -> None:
    """Reset the active OpenEvolve run ID using the provided context token.

    Args:
        token: The token returned by a prior set_current_run_id call.
    """
    current_run_id.reset(token)


class RunIsolationFilter(logging.Filter):
    """Filter log records so handlers only process records from their matching run.

    When multiple OpenEvolve instances run concurrently in the same process
    (e.g., parallel runs in the same event loop or thread pool), they attach
    handlers to the shared ``openevolve`` package logger. Without filtering,
    each instance's handlers would receive log records from all other runs,
    leading to duplicated console output and cross-written log files (GitHub issue #295).

    This filter ensures that a handler only emits records produced while its
    associated run context is active. If no run context is active (e.g.,
    standalone logging during setup or in test suites outside ``run()``),
    the record is permitted through so information is not lost.
    """

    def __init__(self, run_id: str) -> None:
        """Initialize the filter for a specific run ID.

        Args:
            run_id: The unique identifier of the OpenEvolve run instance.
        """
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log records by matching against the active run context.

        Args:
            record: The log record to inspect.

        Returns:
            True if the record belongs to this run or if no active run context is set;
            False if another run context is currently active.
        """
        active_run = current_run_id.get()
        # If an active run is configured in context, only allow records matching this run ID
        if active_run is not None:
            return active_run == self.run_id
        # If no active run is set (e.g., un-scoped calls or direct utility usage), allow through
        return True


class LoggerPrefixFilter(logging.Filter):
    """Rewrite ``record.name`` to live under a configured prefix.

    When installed on a handler, every record emitted through that handler is
    renamed from e.g. ``openevolve.controller`` to
    ``<prefix>.openevolve.controller``. Formatters that include ``%(name)s``
    then show the prefixed name, which makes concurrent runs distinguishable
    and greppable in shared log output.

    The filter is idempotent and avoids prefix stacking across multiple
    handlers: the original un-prefixed name is saved so that applying filters
    with different prefixes does not create nested prefixes like
    ``run2.run1.openevolve.controller``.
    """

    def __init__(self, prefix: Optional[str] = None) -> None:
        """Initialize the prefix filter.

        Args:
            prefix: Optional prefix string to prepend to record logger names.
        """
        super().__init__()
        self.prefix = prefix or ""

    def filter(self, record: logging.LogRecord) -> bool:
        """Prepend prefix to record name without stacking multiple prefixes.

        Args:
            record: The log record to format.

        Returns:
            Always True (does not suppress records, only alters record.name).
        """
        # Store original logger name so multiple handlers or filters do not stack prefixes
        if not hasattr(record, "_openevolve_orig_name"):
            record._openevolve_orig_name = record.name

        orig_name = record._openevolve_orig_name
        if self.prefix:
            record.name = f"{self.prefix}.{orig_name}"
        else:
            record.name = orig_name
        return True
