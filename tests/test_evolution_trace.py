"""
Tests for evolution trace functionality
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from openevolve.evolution_trace import (
    EvolutionTrace, 
    EvolutionTracer,
    extract_evolution_trace_from_checkpoint,
    extract_full_lineage_traces
)
from openevolve.database import Program


class TestEvolutionTrace(unittest.TestCase):
    """Test the EvolutionTrace dataclass"""
    
    def test_trace_creation(self):
        """Test creating an evolution trace entry"""
        trace = EvolutionTrace(
            iteration=1,
            timestamp=1234567890.0,
            parent_id="parent-123",
            child_id="child-456",
            parent_metrics={"score": 0.5, "accuracy": 0.8},
            child_metrics={"score": 0.6, "accuracy": 0.85}
        )
        
        self.assertEqual(trace.iteration, 1)
        self.assertEqual(trace.parent_id, "parent-123")
        self.assertEqual(trace.child_id, "child-456")
        
    def test_calculate_improvement(self):
        """Test improvement calculation"""
        trace = EvolutionTrace(
            iteration=1,
            timestamp=1234567890.0,
            parent_id="parent-123",
            child_id="child-456",
            parent_metrics={"score": 0.5, "accuracy": 0.8, "label": "good"},
            child_metrics={"score": 0.6, "accuracy": 0.75, "label": "better"}
        )
        
        improvement = trace.calculate_improvement()
        
        self.assertAlmostEqual(improvement["score"], 0.1)
        self.assertAlmostEqual(improvement["accuracy"], -0.05)
        self.assertNotIn("label", improvement)  # Non-numeric values excluded
        
    def test_to_dict(self):
        """Test conversion to dictionary"""
        trace = EvolutionTrace(
            iteration=1,
            timestamp=1234567890.0,
            parent_id="parent-123",
            child_id="child-456",
            parent_metrics={"score": 0.5},
            child_metrics={"score": 0.6},
            parent_code="def f(): pass",
            island_id=2
        )
        
        trace_dict = trace.to_dict()
        
        self.assertIn("iteration", trace_dict)
        self.assertIn("parent_code", trace_dict)
        self.assertIn("island_id", trace_dict)
        self.assertNotIn("llm_response", trace_dict)  # None values excluded


class TestEvolutionTracer(unittest.TestCase):
    """Test the EvolutionTracer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.parent_program = Program(
            id="parent-123",
            code="def f(): return 1",
            language="python",
            metrics={"score": 0.5, "accuracy": 0.8},
            generation=1
        )
        self.child_program = Program(
            id="child-456",
            code="def f(): return 2",
            language="python",
            parent_id="parent-123",
            metrics={"score": 0.6, "accuracy": 0.85},
            generation=2
        )
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_tracer_disabled(self):
        """Test that disabled tracer does nothing"""
        tracer = EvolutionTracer(enabled=False)
        
        # Should not create any files or log anything
        tracer.log_trace(
            iteration=1,
            parent_program=self.parent_program,
            child_program=self.child_program
        )
        
        stats = tracer.get_statistics()
        self.assertEqual(stats["total_traces"], 0)
        
    def test_jsonl_format(self):
        """Test JSONL format output"""
        output_path = Path(self.temp_dir) / "trace.jsonl"
        
        tracer = EvolutionTracer(
            output_path=str(output_path),
            format="jsonl",
            include_code=True,
            buffer_size=1  # Write immediately
        )
        
        tracer.log_trace(
            iteration=1,
            parent_program=self.parent_program,
            child_program=self.child_program,
            prompt={"system": "Test prompt"},
            island_id=0
        )
        
        # Check file was created
        self.assertTrue(output_path.exists())
        
        # Read and verify content
        with open(output_path, "r") as f:
            line = f.readline()
            data = json.loads(line)
            
        self.assertEqual(data["iteration"], 1)
        self.assertEqual(data["parent_id"], "parent-123")
        self.assertEqual(data["child_id"], "child-456")
        self.assertEqual(data["parent_code"], "def f(): return 1")
        self.assertEqual(data["island_id"], 0)
        
        tracer.close()
        
    def test_json_format(self):
        """Test JSON format output"""
        output_path = Path(self.temp_dir) / "trace.json"
        
        tracer = EvolutionTracer(
            output_path=str(output_path),
            format="json",
            include_prompts=False,
            buffer_size=1
        )
        
        tracer.log_trace(
            iteration=1,
            parent_program=self.parent_program,
            child_program=self.child_program
        )
        
        tracer.log_trace(
            iteration=2,
            parent_program=self.child_program,
            child_program=self.parent_program  # Reverse for testing
        )
        
        tracer.close()
        
        # Read and verify content
        with open(output_path, "r") as f:
            data = json.load(f)
            
        self.assertIn("metadata", data)
        self.assertIn("traces", data)
        self.assertEqual(len(data["traces"]), 2)
        self.assertEqual(data["traces"][0]["iteration"], 1)
        self.assertEqual(data["traces"][1]["iteration"], 2)
        
    def test_statistics(self):
        """Test statistics tracking"""
        tracer = EvolutionTracer(
            output_path=Path(self.temp_dir) / "trace.jsonl",
            buffer_size=10
        )
        
        # Log improvement
        tracer.log_trace(
            iteration=1,
            parent_program=self.parent_program,
            child_program=self.child_program
        )
        
        # Log decline
        tracer.log_trace(
            iteration=2,
            parent_program=self.child_program,
            child_program=self.parent_program
        )
        
        stats = tracer.get_statistics()
        
        self.assertEqual(stats["total_traces"], 2)
        self.assertIn("total_improvement", stats)
        self.assertIn("best_improvement", stats)
        self.assertIn("worst_decline", stats)
        self.assertIn("improvement_rate", stats)
        
        # Check improvement tracking
        self.assertAlmostEqual(stats["best_improvement"]["score"], 0.1)
        self.assertAlmostEqual(stats["worst_decline"]["score"], -0.1)
        
        tracer.close()
        
    def test_buffer_flushing(self):
        """Test buffer flushing behavior"""
        output_path = Path(self.temp_dir) / "trace.jsonl"
        
        tracer = EvolutionTracer(
            output_path=str(output_path),
            format="jsonl",
            buffer_size=2
        )
        
        # First trace - should not write yet
        tracer.log_trace(
            iteration=1,
            parent_program=self.parent_program,
            child_program=self.child_program
        )
        
        # File should not exist yet
        self.assertFalse(output_path.exists())
        
        # Second trace - should trigger flush
        tracer.log_trace(
            iteration=2,
            parent_program=self.parent_program,
            child_program=self.child_program
        )
        
        # File should now exist
        self.assertTrue(output_path.exists())
        
        # Verify two lines written
        with open(output_path, "r") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 2)
        
        tracer.close()
        
    def test_context_manager(self):
        """Test using tracer as context manager"""
        output_path = Path(self.temp_dir) / "trace.jsonl"
        
        with EvolutionTracer(
            output_path=str(output_path),
            buffer_size=10
        ) as tracer:
            tracer.log_trace(
                iteration=1,
                parent_program=self.parent_program,
                child_program=self.child_program
            )
            
        # File should exist after context exit (close called)
        self.assertTrue(output_path.exists())
        
    def test_artifact_logging(self):
        """Test logging with artifacts"""
        output_path = Path(self.temp_dir) / "trace.jsonl"
        
        tracer = EvolutionTracer(
            output_path=str(output_path),
            include_code=False,
            buffer_size=1
        )
        
        artifacts = {
            "execution_output": "Result: 42",
            "error": None,
            "timing": 0.123
        }
        
        tracer.log_trace(
            iteration=1,
            parent_program=self.parent_program,
            child_program=self.child_program,
            artifacts=artifacts
        )
        
        # Read and verify artifacts are included
        with open(output_path, "r") as f:
            data = json.loads(f.readline())
            
        self.assertIn("artifacts", data)
        self.assertEqual(data["artifacts"]["execution_output"], "Result: 42")
        self.assertEqual(data["artifacts"]["timing"], 0.123)
        
        tracer.close()


class TestEvolutionTraceIntegration(unittest.TestCase):
    """Test integration with OpenEvolve configuration"""
    
    def test_config_integration(self):
        """Test that evolution trace config is properly integrated"""
        from openevolve.config import Config, EvolutionTraceConfig
        
        # Create config with evolution trace enabled
        config = Config()
        config.evolution_trace = EvolutionTraceConfig(
            enabled=True,
            format="jsonl",
            include_code=True,
            include_prompts=False,
            output_path="/tmp/test_trace.jsonl"
        )
        
        # Verify configuration
        self.assertTrue(config.evolution_trace.enabled)
        self.assertEqual(config.evolution_trace.format, "jsonl")
        self.assertTrue(config.evolution_trace.include_code)
        self.assertFalse(config.evolution_trace.include_prompts)
        
    def test_yaml_config(self):
        """Test loading evolution trace config from YAML"""
        import yaml
        from openevolve.config import Config
        
        yaml_content = """
        evolution_trace:
          enabled: true
          format: json
          include_code: false
          include_prompts: true
          output_path: /tmp/evolution_trace.json
          buffer_size: 20
          compress: false
        """
        
        config_dict = yaml.safe_load(yaml_content)
        config = Config.from_dict(config_dict)
        
        self.assertTrue(config.evolution_trace.enabled)
        self.assertEqual(config.evolution_trace.format, "json")
        self.assertFalse(config.evolution_trace.include_code)
        self.assertTrue(config.evolution_trace.include_prompts)
        self.assertEqual(config.evolution_trace.buffer_size, 20)


class TestCheckpointExtraction(unittest.TestCase):
    """Test checkpoint extraction functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / "checkpoint"
        self.programs_dir = self.checkpoint_dir / "programs"
        self.programs_dir.mkdir(parents=True, exist_ok=True)
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def _create_test_program(self, prog_id, parent_id=None, iteration=0, metrics=None, generation=0):
        """Helper to create a test program JSON file"""
        program_data = {
            "id": prog_id,
            "code": f"def func_{prog_id}(): pass",
            "language": "python",
            "parent_id": parent_id,
            "iteration_found": iteration,
            "generation": generation,
            "timestamp": 1234567890.0 + iteration,
            "metrics": metrics or {"score": 0.5, "accuracy": 0.8},
            "metadata": {
                "island": iteration % 3,
                "changes": f"Changes for {prog_id}"
            }
        }
        
        program_file = self.programs_dir / f"{prog_id}.json"
        with open(program_file, "w") as f:
            json.dump(program_data, f)
            
        return program_data
        
    def test_extract_from_empty_checkpoint(self):
        """Test extraction from checkpoint with no programs"""
        traces = extract_evolution_trace_from_checkpoint(self.checkpoint_dir)
        self.assertEqual(len(traces), 0)
        
    def test_extract_single_parent_child(self):
        """Test extraction with single parent-child pair"""
        # Create parent and child programs
        parent = self._create_test_program("parent-1", None, 0, {"score": 0.5}, 0)
        child = self._create_test_program("child-1", "parent-1", 1, {"score": 0.6}, 1)
        
        traces = extract_evolution_trace_from_checkpoint(self.checkpoint_dir)
        
        self.assertEqual(len(traces), 1)
        trace = traces[0]
        
        self.assertEqual(trace.parent_id, "parent-1")
        self.assertEqual(trace.child_id, "child-1")
        self.assertEqual(trace.iteration, 1)
        self.assertEqual(trace.generation, 1)
        self.assertEqual(trace.parent_metrics["score"], 0.5)
        self.assertEqual(trace.child_metrics["score"], 0.6)
        
    def test_extract_multiple_generations(self):
        """Test extraction with multiple generations"""
        # Create a chain of programs
        prog1 = self._create_test_program("prog-1", None, 0, {"score": 0.5}, 0)
        prog2 = self._create_test_program("prog-2", "prog-1", 1, {"score": 0.6}, 1)
        prog3 = self._create_test_program("prog-3", "prog-2", 2, {"score": 0.7}, 2)
        prog4 = self._create_test_program("prog-4", "prog-2", 3, {"score": 0.65}, 2)
        
        traces = extract_evolution_trace_from_checkpoint(self.checkpoint_dir)
        
        self.assertEqual(len(traces), 3)  # Three parent-child pairs
        
        # Verify traces are sorted by iteration
        iterations = [t.iteration for t in traces]
        self.assertEqual(iterations, sorted(iterations))
        
    def test_extract_with_code_inclusion(self):
        """Test extraction with code included"""
        parent = self._create_test_program("parent-1", None, 0)
        child = self._create_test_program("child-1", "parent-1", 1)
        
        traces = extract_evolution_trace_from_checkpoint(
            self.checkpoint_dir,
            include_code=True
        )
        
        self.assertEqual(len(traces), 1)
        trace = traces[0]
        
        self.assertIn("func_parent-1", trace.parent_code)
        self.assertIn("func_child-1", trace.child_code)
        
    def test_extract_without_code(self):
        """Test extraction without code"""
        parent = self._create_test_program("parent-1", None, 0)
        child = self._create_test_program("child-1", "parent-1", 1)
        
        traces = extract_evolution_trace_from_checkpoint(
            self.checkpoint_dir,
            include_code=False
        )
        
        trace = traces[0]
        self.assertIsNone(trace.parent_code)
        self.assertIsNone(trace.child_code)
        
    def test_save_extracted_traces_jsonl(self):
        """Test saving extracted traces in JSONL format"""
        parent = self._create_test_program("parent-1", None, 0)
        child = self._create_test_program("child-1", "parent-1", 1)
        
        output_path = Path(self.temp_dir) / "extracted.jsonl"
        
        traces = extract_evolution_trace_from_checkpoint(
            self.checkpoint_dir,
            output_path=str(output_path),
            format="jsonl"
        )
        
        # Verify file was created
        self.assertTrue(output_path.exists())
        
        # Read and verify content
        with open(output_path, "r") as f:
            lines = f.readlines()
            
        self.assertEqual(len(lines), 1)
        data = json.loads(lines[0])
        self.assertEqual(data["parent_id"], "parent-1")
        self.assertEqual(data["child_id"], "child-1")
        
    def test_save_extracted_traces_json(self):
        """Test saving extracted traces in JSON format"""
        parent = self._create_test_program("parent-1", None, 0)
        child = self._create_test_program("child-1", "parent-1", 1)
        
        output_path = Path(self.temp_dir) / "extracted.json"
        
        traces = extract_evolution_trace_from_checkpoint(
            self.checkpoint_dir,
            output_path=str(output_path),
            format="json"
        )
        
        # Verify file was created
        self.assertTrue(output_path.exists())
        
        # Read and verify content
        with open(output_path, "r") as f:
            data = json.load(f)
            
        self.assertIn("metadata", data)
        self.assertIn("traces", data)
        self.assertEqual(len(data["traces"]), 1)
        self.assertEqual(data["traces"][0]["parent_id"], "parent-1")
        
    def test_extract_full_lineage(self):
        """Test full lineage extraction"""
        # Create a chain of programs with prompts
        prog1 = self._create_test_program("prog-1", None, 0, {"score": 0.5}, 0)
        prog2_data = self._create_test_program("prog-2", "prog-1", 1, {"score": 0.6}, 1)
        prog3_data = self._create_test_program("prog-3", "prog-2", 2, {"score": 0.7}, 2)
        
        # Add prompts to the child programs
        prog2_file = self.programs_dir / "prog-2.json"
        prog2_data["prompts"] = {
            "diff_user": {
                "system": "System prompt for prog-2",
                "user": "User prompt for prog-2",
                "responses": ["LLM response for prog-2"]
            }
        }
        with open(prog2_file, "w") as f:
            json.dump(prog2_data, f)
            
        prog3_file = self.programs_dir / "prog-3.json"
        prog3_data["prompts"] = {
            "full_rewrite_user": {
                "system": "System prompt for prog-3",
                "user": "User prompt for prog-3",
                "responses": ["LLM response for prog-3"]
            }
        }
        with open(prog3_file, "w") as f:
            json.dump(prog3_data, f)
        
        # Extract full lineage traces
        traces = extract_full_lineage_traces(self.checkpoint_dir)
        
        # Should have one trace (for prog-3, which has the longest lineage)
        self.assertGreaterEqual(len(traces), 1)
        
        # Find the trace for prog-3
        prog3_trace = None
        for trace in traces:
            if trace["final_program_id"] == "prog-3":
                prog3_trace = trace
                break
        
        self.assertIsNotNone(prog3_trace)
        self.assertEqual(prog3_trace["generation_depth"], 3)
        self.assertEqual(len(prog3_trace["improvement_steps"]), 2)
        
        # Check first improvement step (prog-1 to prog-2)
        step1 = prog3_trace["improvement_steps"][0]
        self.assertEqual(step1["parent_id"], "prog-1")
        self.assertEqual(step1["child_id"], "prog-2")
        self.assertIsNotNone(step1["action"])
        self.assertEqual(step1["action"]["template"], "diff_user")
        self.assertIn("LLM response for prog-2", step1["action"]["llm_response"])
        
        # Check improvement calculation
        self.assertIn("score", step1["improvement"])
        self.assertAlmostEqual(step1["improvement"]["score"], 0.1)
        
    def test_extract_full_lineage_json_output(self):
        """Test full lineage extraction with JSON output"""
        # Create simple lineage
        self._create_test_program("parent", None, 0, {"score": 0.5}, 0)
        child_data = self._create_test_program("child", "parent", 1, {"score": 0.6}, 1)
        
        # Add prompts
        child_file = self.programs_dir / "child.json"
        child_data["prompts"] = {
            "test_template": {
                "system": "Test system",
                "user": "Test user",
                "responses": ["Test response"]
            }
        }
        with open(child_file, "w") as f:
            json.dump(child_data, f)
        
        output_path = Path(self.temp_dir) / "lineage.json"
        
        traces = extract_full_lineage_traces(
            self.checkpoint_dir,
            output_path=str(output_path),
            format="json"
        )
        
        # Verify output file was created
        self.assertTrue(output_path.exists())
        
        # Load and verify content
        with open(output_path, "r") as f:
            data = json.load(f)
        
        self.assertIn("metadata", data)
        self.assertIn("traces", data)
        self.assertEqual(data["metadata"]["type"], "full_lineage")
        
    def test_extract_with_missing_parent(self):
        """Test extraction handles missing parents gracefully"""
        # Create child without parent in checkpoint
        child = self._create_test_program("child-1", "missing-parent", 1)
        
        traces = extract_evolution_trace_from_checkpoint(self.checkpoint_dir)
        
        # Should skip this program since parent is missing
        self.assertEqual(len(traces), 0)
        
    def test_extract_with_corrupted_file(self):
        """Test extraction handles corrupted files gracefully"""
        # Create a valid program
        parent = self._create_test_program("parent-1", None, 0)
        child = self._create_test_program("child-1", "parent-1", 1)
        
        # Create a corrupted file
        corrupted_file = self.programs_dir / "corrupted.json"
        with open(corrupted_file, "w") as f:
            f.write("not valid json {")
        
        # Should still extract valid programs
        traces = extract_evolution_trace_from_checkpoint(self.checkpoint_dir)
        self.assertEqual(len(traces), 1)
        
    def test_checkpoint_not_found(self):
        """Test error handling for non-existent checkpoint"""
        with self.assertRaises(FileNotFoundError):
            extract_evolution_trace_from_checkpoint("/nonexistent/path")
            
    def test_programs_dir_not_found(self):
        """Test error handling when programs directory is missing"""
        # Create checkpoint dir without programs subdirectory
        empty_checkpoint = Path(self.temp_dir) / "empty_checkpoint"
        empty_checkpoint.mkdir(parents=True, exist_ok=True)
        
        with self.assertRaises(FileNotFoundError):
            extract_evolution_trace_from_checkpoint(empty_checkpoint)


class TestTracerFailures(unittest.TestCase):
    """Failed iterations go to a separate file, only when include_failures is set"""

    FAILURE = dict(
        iteration=4,
        error="No valid diffs found in response",
        parent_id="p1",
        evolution_mode="diff",
        prompt={"system": "s", "user": "u"},
        llm_response="no diff here",
        token_usage={"completion_tokens": 3},
        island_id=1,
    )

    def _tracer(self, tmp, **kwargs):
        return EvolutionTracer(output_path=str(Path(tmp) / "trace.jsonl"), **kwargs)

    def _read(self, path):
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_writes_failure_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracer = self._tracer(tmp, include_failures=True)
            tracer.log_failure(**self.FAILURE)
            self.assertEqual(tracer.failures_path, Path(tmp) / "trace_failures.jsonl")
            [record] = self._read(tracer.failures_path)
            for key, value in self.FAILURE.items():
                self.assertEqual(record[key], value)
            # The trace itself only ever holds parent/child records
            self.assertFalse((Path(tmp) / "trace.jsonl").exists())

    def test_prompt_and_response_follow_include_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracer = self._tracer(tmp, include_failures=True, include_prompts=False)
            tracer.log_failure(**self.FAILURE)
            [record] = self._read(tracer.failures_path)
            self.assertNotIn("llm_response", record)
            self.assertNotIn("prompt", record)
            self.assertEqual(record["error"], self.FAILURE["error"])

    def test_off_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracer = self._tracer(tmp)
            tracer.log_failure(**self.FAILURE)
            self.assertFalse(tracer.failures_path.exists())


if __name__ == "__main__":
    unittest.main()