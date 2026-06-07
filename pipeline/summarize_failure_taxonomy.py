"""Top-level entry point for summarizing pipeline verification failures."""

from pipeline.shared.summarize_failure_taxonomy import *  # noqa: F401,F403
from pipeline.shared.summarize_failure_taxonomy import main


if __name__ == "__main__":
    raise SystemExit(main())
