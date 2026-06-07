"""Top-level entry point for the kernel-aware AtenIR fusion pipeline."""

from pipeline.kernel_fusion_agent.cli import *  # noqa: F401,F403
from pipeline.kernel_fusion_agent.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
