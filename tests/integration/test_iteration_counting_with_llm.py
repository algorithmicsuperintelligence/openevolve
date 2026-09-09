"""
Integration test for controller iteration behavior with real LLM inference.

Ported from tests/test_iteration_counting.py, which previously guarded this test
with a runtime `skipTest` when no optillm server was reachable. It now lives here so
it runs against the shared optillm+model fixtures (no skips) and uses TEST_MODEL via
`evolution_config` instead of a hardcoded model name.
"""

import pytest

from openevolve.controller import OpenEvolve


class TestIterationCountingWithLLM:
    """Real-LLM checks for iteration counting."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_controller_iteration_behavior(
        self,
        optillm_server,
        evolution_config,
        test_program_file,
        test_evaluator_file,
        evolution_output_dir,
    ):
        """Run a short real evolution and verify processed iteration progress."""
        evolution_config.max_iterations = 8
        evolution_config.evaluator.parallel_evaluations = 1
        evolution_config.evaluator.timeout = 30  # Longer timeout for small model

        controller = OpenEvolve(
            initial_program_path=str(test_program_file),
            evaluation_file=str(test_evaluator_file),
            config=evolution_config,
            output_dir=str(evolution_output_dir),
        )

        await controller.run(iterations=8)

        state = controller.database.get_state()
        assert state.program_count >= 1
        assert state.last_iteration == 8
