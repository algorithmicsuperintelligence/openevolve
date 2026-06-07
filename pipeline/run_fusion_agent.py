"""Top-level entry point for the direct AtenIR fusion pipeline."""

from pipeline.fusion_agent.cli import *  # noqa: F401,F403
from pipeline.fusion_agent.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
