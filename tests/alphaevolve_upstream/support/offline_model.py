"""Importable fixed-proposal provider for real OE multiprocessing workers."""

import json
import os
import sys
from pathlib import Path

MARKER = "# AlphaEvolve upstream CI: evaluated fixed proposal."


class FixedProposalModel:
    def __init__(self, config):
        self.model = config.name

    async def generate_with_context(self, system_message, messages, **kwargs):
        guard = sys.modules.get("sitecustomize")
        if not getattr(guard, "ACTIVE", False):
            raise AssertionError("Offline tripwires must also be active in OE workers")
        with Path(os.environ["OE_INTEROP_CALL_LOG"]).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"model": self.model, "pid": os.getpid()}) + "\n")
        return (
            "<<<<<<< SEARCH\n# EVOLVE-BLOCK-START\n=======\n"
            f"# EVOLVE-BLOCK-START\n{MARKER}\n>>>>>>> REPLACE"
        )


def fixed_model(config):
    return FixedProposalModel(config)
