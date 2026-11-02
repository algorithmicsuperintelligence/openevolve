"""Regression tests for evolution traces recorded from reused mutable payloads."""

import json
import tempfile
import unittest
from pathlib import Path

from openevolve.database import Program
from openevolve.evolution_trace import EvolutionTracer


class TestEvolutionTraceSnapshots(unittest.TestCase):
    def test_reused_program_metrics_preserve_each_transition(self):
        for format, buffer_size in (("jsonl", 1), ("jsonl", 10), ("json", 1), ("json", 10)):
            with (
                self.subTest(format=format, buffer_size=buffer_size),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory) / f"trace.{format}"
                parent = Program(id="parent", code="pass", metrics={"combined_score": 1.0})
                child = Program(id="child", code="pass", metrics={"combined_score": 3.0})
                with EvolutionTracer(str(path), format=format, buffer_size=buffer_size) as tracer:
                    tracer.log_trace(1, parent, child)
                    parent.metrics["combined_score"] = 10.0
                    child.metrics["combined_score"] = 30.0
                    tracer.log_trace(2, parent, child)
                    parent.metrics.clear()
                    child.metrics["combined_score"] = 100.0

                if format == "jsonl":
                    traces = [json.loads(line) for line in path.read_text().splitlines()]
                else:
                    traces = json.loads(path.read_text())["traces"]
                self.assertEqual(len(traces), 2)
                for trace, parent_score, child_score in zip(traces, (1.0, 10.0), (3.0, 30.0)):
                    self.assertEqual(trace["parent_metrics"], {"combined_score": parent_score})
                    self.assertEqual(trace["child_metrics"], {"combined_score": child_score})
                    self.assertEqual(
                        trace["improvement_delta"], {"combined_score": child_score - parent_score}
                    )
                self.assertEqual(
                    tracer.get_statistics()["total_improvement"], {"combined_score": 22.0}
                )

    def test_prompt_artifacts_and_metadata_are_deep_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            parent = Program(id="parent", code="pass")
            child = Program(id="child", code="pass")
            prompt = {"system": "original"}
            artifacts = {"evaluation": {"messages": ["original"]}}
            metadata = {"worker": {"attempts": [1]}}
            with EvolutionTracer(str(path), buffer_size=10) as tracer:
                tracer.log_trace(
                    1, parent, child, prompt=prompt, artifacts=artifacts, metadata=metadata
                )
                prompt["system"] = "changed"
                artifacts["evaluation"]["messages"].append("changed")
                metadata["worker"]["attempts"].append(2)
                tracer.log_trace(
                    2, parent, child, prompt=prompt, artifacts=artifacts, metadata=metadata
                )
                prompt.clear()
                artifacts.clear()
                metadata.clear()

            first, second = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(first["prompt"], {"system": "original"})
            self.assertEqual(first["artifacts"], {"evaluation": {"messages": ["original"]}})
            self.assertEqual(first["metadata"], {"worker": {"attempts": [1]}})
            self.assertEqual(second["prompt"], {"system": "changed"})
            self.assertEqual(
                second["artifacts"], {"evaluation": {"messages": ["original", "changed"]}}
            )
            self.assertEqual(second["metadata"], {"worker": {"attempts": [1, 2]}})

    def test_excluded_prompts_stay_out_of_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            parent = Program(id="parent", code="pass")
            child = Program(id="child", code="pass")
            with EvolutionTracer(str(path), include_prompts=False) as tracer:
                tracer.log_trace(
                    1, parent, child, prompt={"system": "private"}, llm_response="private"
                )
            trace = json.loads(path.read_text())
            self.assertNotIn("prompt", trace)
            self.assertNotIn("llm_response", trace)
