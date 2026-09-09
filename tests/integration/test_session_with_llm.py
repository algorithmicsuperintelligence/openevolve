"""Integration checks for database sessions with real model generation."""

import json

import pytest

from openevolve.controller import OpenEvolve
from openevolve.database_memory import InMemoryProgramDatabase


class TestSessionWithLLM:
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_continue_with_same_database(
        self,
        optillm_server,
        evolution_config,
        test_program_file,
        test_evaluator_file,
        evolution_output_dir,
    ):
        database = InMemoryProgramDatabase(evolution_config.database)
        first = OpenEvolve(
            str(test_program_file),
            str(test_evaluator_file),
            evolution_config,
            output_dir=str(evolution_output_dir),
            database=database,
        )
        await first.run(iterations=2)
        assert database.get_state().last_iteration == 2
        second = OpenEvolve(
            str(test_program_file),
            str(test_evaluator_file),
            evolution_config,
            output_dir=str(evolution_output_dir),
            database=database,
        )
        await second.run(iterations=2)
        assert database.get_state().last_iteration == 4
        assert not (evolution_output_dir / "checkpoints").exists()

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_best_result_export(
        self,
        optillm_server,
        evolution_config,
        test_program_file,
        test_evaluator_file,
        evolution_output_dir,
    ):
        controller = OpenEvolve(
            str(test_program_file),
            str(test_evaluator_file),
            evolution_config,
            output_dir=str(evolution_output_dir),
        )
        best = await controller.run(iterations=2)
        assert best is not None
        best_dir = evolution_output_dir / "best"
        assert (best_dir / "best_program.py").read_text() == best.code
        assert json.loads((best_dir / "best_program_info.json").read_text())["id"] == best.id
