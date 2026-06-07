"""Top-level entry point for the primitive AtenIR lowering pipeline."""

from pipeline.primitive_atenir_lowering_agent.cli import *  # noqa: F401,F403
from pipeline.primitive_atenir_lowering_agent.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
