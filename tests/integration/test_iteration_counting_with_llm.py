"""
Integration test for controller iteration/checkpoint behavior with real LLM inference.

Ported from tests/test_iteration_counting.py, which previously guarded this test
with a runtime `skipTest` when no optillm server was reachable. It now lives here so
it runs against the shared optillm+model fixtures (no skips) and uses TEST_MODEL via
`evolution_config` instead of a hardcoded model name.
"""

import pytest

from openevolve.controller import OpenEvolve


class TestIterationCountingWithLLM:
    """Real-LLM checks for iteration counting and checkpoint alignment."""

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
        """Run a short real evolution and verify checkpoints align with the interval."""
        evolution_config.max_iterations = 8
        evolution_config.checkpoint_interval = 4
        evolution_config.evaluator.parallel_evaluations = 1
        evolution_config.evaluator.timeout = 30  # Longer timeout for small model

        controller = OpenEvolve(
            initial_program_path=str(test_program_file),
            evaluation_file=str(test_evaluator_file),
            config=evolution_config,
            output_dir=str(evolution_output_dir),
        )

        # Track checkpoint calls
        checkpoint_calls = []
        original_save = controller._save_checkpoint
        controller._save_checkpoint = lambda i: checkpoint_calls.append(i) or original_save(i)

        await controller.run(iterations=8)

        print(f"Checkpoint calls: {checkpoint_calls}")
        print(f"Total programs: {len(controller.database.programs)}")

        # Should always have at least the initial program.
        assert len(controller.database.programs) >= 1, "Should have at least the initial program"

        # If any evolution succeeded, checkpoints must be a subset of the expected
        # interval multiples (4, 8) - never at an unexpected iteration.
        if len(controller.database.programs) > 1:
            assert all(
                cp % evolution_config.checkpoint_interval == 0 for cp in checkpoint_calls
            ), f"Checkpoints must align with interval; got {checkpoint_calls}"
            assert set(checkpoint_calls).issubset(
                {4, 8}
            ), f"Unexpected checkpoint iterations: {checkpoint_calls}"
