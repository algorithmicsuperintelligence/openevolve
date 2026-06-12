"""Top-level entry point for the deterministic hand-written dispatch pipeline (Pipeline D)."""

from pipeline.handwritten_dispatch.cli import *  # noqa: F401,F403
from pipeline.handwritten_dispatch.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
