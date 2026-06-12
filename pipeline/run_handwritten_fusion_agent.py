"""Top-level entry point for the hand-written context fusion pipeline (Pipeline E)."""

from pipeline.handwritten_fusion_agent.cli import *  # noqa: F401,F403
from pipeline.handwritten_fusion_agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
