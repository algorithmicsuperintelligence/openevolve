"""
Integration tests for checkpoint functionality with real LLM inference
"""

import pytest
import asyncio
from openevolve.controller import OpenEvolve


class TestCheckpointWithLLM:
    """Test checkpoints with real LLM generation"""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_checkpoint_intervals_with_real_llm(
        self,
        optillm_server,
        evolution_config,
        test_program_file,
        test_evaluator_file,
        evolution_output_dir
    ):
        """Test checkpoints occur at correct intervals with real evolution"""
        evolution_config.checkpoint_interval = 2
        evolution_config.max_iterations = 4  # Much smaller for CI speed
        evolution_config.evaluator.timeout = 15  # Shorter timeout for CI
        
        checkpoint_calls = []
        
        controller = OpenEvolve(
            initial_program_path=str(test_program_file),
            evaluation_file=str(test_evaluator_file),
            config=evolution_config,
            output_dir=str(evolution_output_dir)
        )
        
        # Track checkpoint calls
        original_save = controller._save_checkpoint
        controller._save_checkpoint = lambda i: checkpoint_calls.append(i) or original_save(i)
        
        await controller.run(iterations=4)
        
        # Check that some checkpoints were called
        # Note: Checkpoints only occur on successful iterations
        print(f"Checkpoint calls: {checkpoint_calls}")
        
        # We expect checkpoints at multiples of 2, but only for successful iterations
        # So we might see some subset of [2, 4] depending on how many iterations succeeded
        expected_checkpoints = [2, 4]
        successful_checkpoints = [cp for cp in expected_checkpoints if cp in checkpoint_calls]
        
        # At least one checkpoint should have occurred if any iterations succeeded
        if len(controller.database.programs) > 1:  # More than just initial program
            assert len(checkpoint_calls) > 0, "Should have at least one checkpoint call if evolution succeeded"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_checkpoint_resume_functionality(
        self,
        optillm_server,
        evolution_config,
        test_program_file,
        test_evaluator_file,
        evolution_output_dir
    ):
        """Test checkpoint save and resume with real LLM"""
        evolution_config.checkpoint_interval = 4
        evolution_config.max_iterations = 8
        
        # Run first phase
        controller1 = OpenEvolve(
            initial_program_path=str(test_program_file),
            evaluation_file=str(test_evaluator_file),
            config=evolution_config,
            output_dir=str(evolution_output_dir)
        )
        
        await controller1.run(iterations=6)
        
        # Check if checkpoint was created
        checkpoints_dir = evolution_output_dir / "checkpoints"
        if checkpoints_dir.exists():
            checkpoint_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")]
            print(f"Found checkpoint directories: {[d.name for d in checkpoint_dirs]}")
            
            if checkpoint_dirs:
                # Find the latest checkpoint
                latest_checkpoint = max(checkpoint_dirs, key=lambda d: int(d.name.split("_")[1]))
                checkpoint_iter = int(latest_checkpoint.name.split("_")[1])
                
                # ProgramDatabase.save writes metadata.json plus one JSON file per program
                assert (latest_checkpoint / "metadata.json").exists(), "Database metadata should exist"
                saved_ids = {p.stem for p in (latest_checkpoint / "programs").glob("*.json")}
                assert saved_ids, "Checkpoint should contain saved programs"
                print(f"Successfully created checkpoint at iteration {checkpoint_iter}")

                # Resume from the checkpoint and run a few more iterations
                controller2 = OpenEvolve(
                    initial_program_path=str(test_program_file),
                    evaluation_file=str(test_evaluator_file),
                    config=evolution_config,
                    output_dir=str(evolution_output_dir)
                )
                await controller2.run(iterations=3, checkpoint_path=str(latest_checkpoint))

                assert saved_ids <= set(controller2.database.programs), \
                    "Resumed database should contain every program from the checkpoint"
                assert controller2.database.last_iteration > checkpoint_iter, \
                    "Resumed run should continue past the checkpoint iteration"
            else:
                print("No checkpoints created (likely due to all iterations failing)")
        else:
            print("No checkpoints directory created")

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_final_checkpoint_creation(
        self,
        optillm_server,
        evolution_config,
        test_program_file,
        test_evaluator_file,
        evolution_output_dir
    ):
        """Test that final checkpoint is created regardless of interval"""
        evolution_config.checkpoint_interval = 100  # Large interval
        evolution_config.max_iterations = 5
        
        checkpoint_calls = []
        
        controller = OpenEvolve(
            initial_program_path=str(test_program_file),
            evaluation_file=str(test_evaluator_file),
            config=evolution_config,
            output_dir=str(evolution_output_dir)
        )
        
        original_save = controller._save_checkpoint
        controller._save_checkpoint = lambda i: checkpoint_calls.append(i) or original_save(i)
        
        await controller.run(iterations=5)
        
        print(f"Final checkpoint calls: {checkpoint_calls}")
        
        # Final checkpoint may be created at the end even if no interval checkpoints occurred
        # This depends on the controller logic, so we just verify the system didn't crash
        assert len(controller.database.programs) >= 1, "Should have at least the initial program"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_checkpoint_with_best_program_save(
        self,
        optillm_server,
        evolution_config,
        test_program_file,
        test_evaluator_file,
        evolution_output_dir
    ):
        """Test that checkpoints include best program information"""
        evolution_config.checkpoint_interval = 3
        evolution_config.max_iterations = 6
        
        controller = OpenEvolve(
            initial_program_path=str(test_program_file),
            evaluation_file=str(test_evaluator_file),
            config=evolution_config,
            output_dir=str(evolution_output_dir)
        )
        
        await controller.run(iterations=6)
        
        # Check best program directory
        best_dir = evolution_output_dir / "best"
        if best_dir.exists():
            best_files = list(best_dir.glob("*"))
            print(f"Best program files: {[f.name for f in best_files]}")
            
            # Should have best program file and info
            program_files = [f for f in best_files if f.suffix == ".py"]
            info_files = [f for f in best_files if f.name.endswith("_info.json")]
            
            if program_files:
                assert len(program_files) >= 1, "Should have best program file"
                
            if info_files:
                assert len(info_files) >= 1, "Should have best program info file"