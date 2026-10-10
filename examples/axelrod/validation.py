"""Runs a final analysis with round robin and multiple rounds."""

import importlib.util
import os
import pprint

import axelrod as axl


if __name__ == "__main__":
    # Load the program
    spec = importlib.util.spec_from_file_location("program", os.path.join("examples", "axelrod", "best_program.py"))
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)

    players = [program.AdaptiveCycleBreakerTFT()] + [s() for s in axl.short_run_time_strategies]
    tournament = axl.Tournament(players, repetitions=5)
    results = tournament.play(progress_bar=True)

    summary = results.summarise(results)
    pprint.pprint(summary)
