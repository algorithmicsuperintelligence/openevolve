"""Logging helpers shared across OpenEvolve.

Provides :class:`LoggerPrefixFilter`, a handler-level filter that rewrites
``record.name`` so every downstream module logger appears under a common
prefix. This lets users tell apart log output coming from multiple concurrent
``OpenEvolve`` runs (GitHub issue #290).
"""

import logging
from typing import Optional


class LoggerPrefixFilter(logging.Filter):
    """Rewrite ``record.name`` to live under a configured prefix.

    When installed on a handler, every record emitted through that handler is
    renamed from e.g. ``openevolve.controller`` to
    ``<prefix>.openevolve.controller``. Formatters that include ``%(name)s``
    then show the prefixed name, which makes concurrent runs distinguishable
    and greppable in shared log output.

    The filter is idempotent: names that already carry the prefix are left
    untouched, so a record flowing through several handlers that share the
    same filter instance is only prefixed once.
    """

    def __init__(self, prefix: Optional[str] = None) -> None:
        super().__init__()
        self.prefix = prefix or ""

    def filter(self, record: logging.LogRecord) -> bool:
        if self.prefix and not record.name.startswith(self.prefix + "."):
            record.name = f"{self.prefix}.{record.name}"
        return True
