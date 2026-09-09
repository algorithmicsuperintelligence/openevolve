"""
Integration tests for the full evolution pipeline with real LLM inference
"""

import pytest
import asyncio
from openevolve.controller import OpenEvolve


class TestEvolutionPipeline:
    """Test complete evolution with real LLM generation"""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_full_evolution_loop(
        self, 
        optillm_server, 
        evolution_config, 
        test_program_file, 
        test_evaluator_file,
        evolution_output_dir
    ):
        """Test complete evolution with real LLM"""
        # Configure smaller iteration count for testing
        evolution_config.max_iterations = 8
        
        # Run evolution
        controller = OpenEvolve(
            initial_program_path=str(test_program_file),
            evaluation_file=str(test_evaluator_file),
            config=evolution_config,
            output_dir=str(evolution_output_dir)
        )
        
        best_program = await controller.run(iterations=3)
        
        # Verify basic evolution functionality 
        assert len(controller.database.programs) >= 1, "Should have at least the initial program"
        assert best_program is not None, "Should have a best program"
        
        # Check no duplicate chains (validates our per-island MAP-Elites fix)
        program_ids = list(controller.database.programs.keys())
        migrant_programs = [pid for pid in program_ids if "_migrant_" in pid]
        assert len(migrant_programs) == 0, f"Found programs with _migrant_ suffix: {migrant_programs}"
        
        # Print stats for debugging
        total_programs = len(controller.database.programs)
        evolved_programs = [p for p in controller.database.programs.values() if p.iteration_found > 0]
        print(f"Evolution results: {total_programs} total programs, {len(evolved_programs)} evolved programs")
        
        # Verify evolution completed successfully
        assert len(controller.database.programs) >= 1, "Should have at least the initial program"
        
        # Check that programs are distributed across islands
        island_counts = {i: 0 for i in range(evolution_config.database.num_islands)}
        for program in controller.database.programs.values():
            island = program.metadata.get("island", 0)
            island_counts[island] += 1
        
        # At least one island should have programs
        populated_islands = [i for i, count in island_counts.items() if count > 0]
        assert len(populated_islands) >= 1, "At least one island should have programs"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_island_feature_maps_populated(
        self,
        optillm_server,
        evolution_config,
        test_program_file,
        test_evaluator_file,
        evolution_output_dir
    ):
        """Test that island feature maps are properly populated during evolution"""
        evolution_config.max_iterations = 6
        evolution_config.database.num_islands = 3
        
        controller = OpenEvolve(
            initial_program_path=str(test_program_file),
            evaluation_file=str(test_evaluator_file),
            config=evolution_config,
            output_dir=str(evolution_output_dir)
        )
        
        await controller.run(iterations=6)
        
        # Check that island feature maps have been populated
        total_mapped_programs = 0
        for island_idx, island_map in enumerate(controller.database.island_feature_maps):
            program_count = len(island_map)
            total_mapped_programs += program_count
            print(f"Island {island_idx}: {program_count} programs in feature map")
        
        assert total_mapped_programs > 0, "Island feature maps should be populated"
        
        # Verify that all programs in feature maps exist in database
        for island_idx, island_map in enumerate(controller.database.island_feature_maps):
            for coord, program_id in island_map.items():
                assert program_id in controller.database.programs, \
                    f"Program {program_id} in island {island_idx} feature map not found in database"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_evolution_with_small_model_succeeds(
        self,
        optillm_server,
        evolution_config,
        test_program_file,
        test_evaluator_file,
        evolution_output_dir
    ):
        """Test that evolution works with small local model (may not be perfect but should not crash)"""
        evolution_config.max_iterations = 4
        evolution_config.evaluator.timeout = 30  # Longer timeout for small model
        
        controller = OpenEvolve(
            initial_program_path=str(test_program_file),
            evaluation_file=str(test_evaluator_file),
            config=evolution_config,
            output_dir=str(evolution_output_dir)
        )
        
        # This should not crash, even if some LLM generations fail
        best_program = await controller.run(iterations=4)
        
        # Basic sanity checks
        assert controller.database.programs, "Should have at least the initial program"
        assert best_program is not None or len(controller.database.programs) >= 1, \
            "Should have a best program or at least the initial program"
        
        # Check that output directory was created and has some structure
        assert evolution_output_dir.exists(), "Output directory should exist"
        logs_dir = evolution_output_dir / "logs"
        if logs_dir.exists():
            log_files = list(logs_dir.glob("*.log"))
            # It's okay if no log files - depends on config
            print(f"Found {len(log_files)} log files")

    @pytest.mark.slow
    @pytest.mark.asyncio 
    async def test_best_program_tracking(
        self,
        optillm_server,
        evolution_config,
        test_program_file,
        test_evaluator_file,
        evolution_output_dir
    ):
        """Test that best program tracking works correctly"""
        evolution_config.max_iterations = 5
        
        controller = OpenEvolve(
            initial_program_path=str(test_program_file),
            evaluation_file=str(test_evaluator_file),
            config=evolution_config,
            output_dir=str(evolution_output_dir)
        )
        
        best_program = await controller.run(iterations=5)
        
        # Check best program tracking
        if controller.database.best_program_id:
            best_from_db = controller.database.get(controller.database.best_program_id)
            assert best_from_db is not None, "Best program should exist in database"
            
            if best_program:
                assert best_program.id == controller.database.best_program_id, \
                    "Returned best program should match tracked best program"
        
        # Alternative check: get best program from database
        best_from_query = controller.database.get_best_program()
        assert best_from_query is not None, "Should be able to get best program from database"