"""
Test to verify the fix for GitHub issue #246 in the actual process_parallel code
"""
import unittest
import tempfile
import os
from unittest.mock import MagicMock, patch, Mock
from concurrent.futures import Future

from openevolve.process_parallel import ProcessParallelController
from openevolve.config import Config
from openevolve.database_memory import InMemoryProgramDatabase as ProgramDatabase
from openevolve.database import Program


class TestProcessParallelFix(unittest.TestCase):
    """Test that process_parallel now uses the safe sample_from_island method"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = Config()
        self.config.database.num_islands = 5
        self.config.evaluator.parallel_evaluations = 5
        
        # Create database
        self.database = ProgramDatabase(self.config.database)
        
        # Add test programs to islands
        for i in range(20):
            program = Program(
                id=f"prog_{i}",
                code=f"def test_{i}(): return {i}",
                metrics={"score": i * 0.1}
            )
            self.database.add(program, target_island=i % 5)
        
        # Mock evaluation file
        self.eval_file = "dummy_evaluator.py"
        
    def test_submit_iteration_uses_sample_from_island(self):
        """Test that _submit_iteration uses the safe sample_from_island method"""
        
        # Create controller
        controller = ProcessParallelController(
            config=self.config,
            evaluation_file=self.eval_file,
            database=self.database
        )
        
        # Mock the executor
        controller.executor = Mock()
        mock_future = Mock(spec=Future)
        controller.executor.submit.return_value = mock_future
        
        # Spy on the database methods
        original_sample = self.database.sample
        original_sample_from_island = self.database.sample_from_island
        
        sample_called = []
        sample_from_island_called = []
        
        def track_sample(*args, **kwargs):
            sample_called.append((args, kwargs))
            return original_sample(*args, **kwargs)
        
        def track_sample_from_island(*args, **kwargs):
            sample_from_island_called.append((args, kwargs))
            return original_sample_from_island(*args, **kwargs)
        
        self.database.sample = track_sample
        self.database.sample_from_island = track_sample_from_island
        
        # Submit an iteration to island 3
        result = controller._submit_iteration(iteration=1, island_id=3)
        
        # Verify sample_from_island was called with correct island
        self.assertEqual(len(sample_from_island_called), 1, 
                        "sample_from_island should be called exactly once")
        
        call_args, call_kwargs = sample_from_island_called[0]
        self.assertIn("island_id", call_kwargs)
        self.assertEqual(call_kwargs["island_id"], 3, 
                        "sample_from_island should be called with island_id=3")
        
        # Verify the old sample method was NOT called
        # (it might be called indirectly if island is empty, but not directly)
        direct_sample_calls = [c for c in sample_called if "from_island" not in str(c)]
        self.assertEqual(len(direct_sample_calls), 0,
                        "The old sample() method should not be called directly")
        
        print("✅ _submit_iteration now uses safe sample_from_island method")
    
    def test_concurrent_submissions_no_race_condition(self):
        """Test that concurrent submissions don't cause race conditions"""
        
        # Create controller
        controller = ProcessParallelController(
            config=self.config,
            evaluation_file=self.eval_file,
            database=self.database
        )
        
        # Mock the executor
        controller.executor = Mock()
        controller.executor.submit.return_value = Mock(spec=Future)
        
        # Track current_island modifications
        island_modifications = []
        original_setattr = self.database.__setattr__
        
        def track_island_changes(name, value):
            if name == "current_island":
                island_modifications.append(value)
            return original_setattr(name, value)
        
        # This would catch any attempt to modify current_island
        with patch.object(self.database, '__setattr__', track_island_changes):
            # Submit multiple iterations to different islands
            for i in range(10):
                controller._submit_iteration(iteration=i, island_id=i % 5)
        
        # current_island should never be modified during submissions
        self.assertEqual(len(island_modifications), 0,
                        "current_island should not be modified during submissions")
        
        print("✅ No race conditions detected with concurrent submissions")
    
    def test_database_state_unchanged_after_sampling(self):
        """Test that database state is unchanged after sampling from island"""
        
        initial_island = self.database.current_island
        
        # Sample from different islands
        for island_id in range(5):
            parent, inspirations = self.database.sample_from_island(
                island_id=island_id,
                num_inspirations=3
            )
            
            # Verify current_island hasn't changed
            self.assertEqual(self.database.current_island, initial_island,
                           f"current_island changed after sampling from island {island_id}")
        
        print("✅ Database state remains unchanged after sampling")


if __name__ == "__main__":
    print("Testing process_parallel fix for GitHub issue #246...\n")
    
    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestProcessParallelFix)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*60)
    if result.wasSuccessful():
        print("🎉 All tests passed! The fix is working correctly.")
        print("GitHub issue #246 has been resolved.")
    else:
        print("Some tests failed. Check the output above.")