"""
Tests for process-based parallel controller
"""

import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
import time
from concurrent.futures import Future, ProcessPoolExecutor


def _slow_test_worker(marker_path: str) -> str:
    Path(marker_path).write_text(str(os.getpid()))
    time.sleep(5)
    return "finished"


# Set dummy API key for testing
os.environ["OPENAI_API_KEY"] = "test"

from openevolve.config import Config, DatabaseConfig, EvaluatorConfig, LLMConfig, PromptConfig
from openevolve.database import Program, ProgramDatabase
from openevolve import process_parallel as process_parallel_module
from openevolve.process_parallel import (
    ProcessParallelController,
    SerializableResult,
    _with_archive_context,
)


class TestProcessParallel(unittest.TestCase):
    """Tests for process-based parallel controller"""

    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()

        # Create test config
        self.config = Config()
        self.config.max_iterations = 10
        self.config.evaluator.parallel_evaluations = 2
        self.config.evaluator.timeout = 10
        # One island per test program so each owns its MAP-Elites cell and none is
        # displaced (MAP-Elites removes programs displaced from their cell).
        self.config.database.num_islands = 3
        self.config.database.in_memory = True
        self.config.checkpoint_interval = 5

        # Create test evaluation file
        self.eval_content = """
def evaluate(program_path):
    return {"score": 0.5, "performance": 0.6}
"""
        self.eval_file = os.path.join(self.test_dir, "evaluator.py")
        with open(self.eval_file, "w") as f:
            f.write(self.eval_content)

        # Create test database
        self.database = ProgramDatabase(self.config.database)

        # Add some test programs, one per island so each survives as its cell owner
        for i in range(3):
            program = Program(
                id=f"test_{i}",
                code=f"def func_{i}(): return {i}",
                language="python",
                metrics={"score": 0.5 + i * 0.1, "performance": 0.4 + i * 0.1},
                iteration_found=0,
            )
            self.database.add(program, target_island=i)

    def tearDown(self):
        """Clean up test environment"""
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_controller_initialization(self):
        """Test that controller initializes correctly"""
        controller = ProcessParallelController(self.config, self.eval_file, self.database)

        self.assertEqual(controller.num_workers, 2)
        self.assertIsNone(controller.executor)
        self.assertIsNotNone(controller.shutdown_event)
        self.assertEqual(controller.llm_usage["llm_calls"], 0)
        self.assertEqual(controller.llm_usage["total_provider_tokens"], 0)

    def test_early_stopping_starts_from_best_archived_program(self):
        self.config.early_stopping_metric = "score"
        controller = ProcessParallelController(
            self.config, self.eval_file, self.database
        )

        self.assertEqual(controller._initial_early_stopping_score(), 0.7)

    def test_missing_custom_early_stopping_metric_is_not_averaged(self):
        self.config.early_stopping_metric = "missing"
        controller = ProcessParallelController(
            self.config, self.eval_file, self.database
        )

        self.assertEqual(
            controller._initial_early_stopping_score(), float("-inf")
        )

    def test_worker_config_serialization_preserves_llm_budgets(self):
        self.config.max_llm_calls = 9
        self.config.max_total_provider_tokens = 12_345
        self.config.program_identity_artifact = "program-identity.txt"
        self.config.phenotype_identity_artifact = "phenotype-identity.txt"
        controller = ProcessParallelController(
            self.config, self.eval_file, self.database
        )

        serialized = controller._serialize_config(self.config)

        self.assertEqual(serialized["max_llm_calls"], 9)
        self.assertEqual(serialized["max_total_provider_tokens"], 12_345)
        self.assertEqual(
            serialized["program_identity_artifact"], "program-identity.txt"
        )
        self.assertEqual(
            serialized["phenotype_identity_artifact"], "phenotype-identity.txt"
        )

    def test_worker_rejects_unchanged_program_identity(self):
        self.config.program_identity_artifact = "program-identity.txt"

        class FakeLLM:
            last_call_metadata = {
                "provider_response_id": "response-semantic-noop",
                "model": "fake-model",
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                },
            }

            async def generate_with_context(self, **_kwargs):
                return "\n".join(
                    [
                        "<" * 7 + " SEARCH",
                        "    return 1",
                        "=" * 7,
                        "    return 2",
                        ">" * 7 + " REPLACE",
                    ]
                )

        class IdentityEvaluator:
            async def evaluate_program(self, _code, _child_id):
                return {"combined_score": 0.8}

            def get_pending_artifacts(self, _child_id):
                return {"program-identity.txt": "same-identity"}

        class FakePromptSampler:
            def build_prompt(self, **_kwargs):
                return {"system": "system", "user": "user"}

        parent = Program(
            id="parent",
            code="def solve():\n    return 1\n",
            language="python",
            metrics={"combined_score": 0.5},
            metadata={"island": 0},
        )
        snapshot = {
            "programs": {"parent": parent.to_dict()},
            "artifacts": {
                "parent": {"program-identity.txt": "same-identity"}
            },
            "current_island": 0,
            "islands": [["parent"]],
            "feature_dimensions": ["combined_score"],
            "sampling_island": 0,
        }

        with (
            patch.object(
                process_parallel_module,
                "_lazy_init_worker_components",
                return_value=None,
            ),
            patch.object(
                process_parallel_module,
                "_worker_config",
                self.config,
                create=True,
            ),
            patch.object(
                process_parallel_module,
                "_worker_llm_ensemble",
                FakeLLM(),
                create=True,
            ),
            patch.object(
                process_parallel_module,
                "_worker_evaluator",
                IdentityEvaluator(),
                create=True,
            ),
            patch.object(
                process_parallel_module,
                "_worker_prompt_sampler",
                FakePromptSampler(),
                create=True,
            ),
        ):
            result = process_parallel_module._run_iteration_worker(
                9, snapshot, "parent", []
            )

        self.assertIsNone(result.child_program_dict)
        self.assertIn("same program identity", result.error)

    def test_worker_builds_deterministic_archive_context(self):
        self.config.prompt.archive_context_artifact = "program-structure.json"
        self.config.prompt.archive_context_max_items = 2
        parent_artifacts = {
            "program-structure.json": '{"schedule":[1]}',
            "guidance": "keep this",
        }
        snapshot = {
            "artifacts": {
                "z": {"program-structure.json": '{"schedule":[2]}'},
                "a": {"program-structure.json": '{"schedule":[1]}'},
                "m": {"program-structure.json": '{"schedule":[3]}'},
            }
        }

        with patch.object(
            process_parallel_module,
            "_worker_config",
            self.config,
            create=True,
        ):
            augmented = _with_archive_context(parent_artifacts, snapshot)

        self.assertIsNot(augmented, parent_artifacts)
        self.assertNotIn("archive-context.json", parent_artifacts)
        context = json.loads(augmented["archive-context.json"])
        self.assertEqual(context["archive_item_count"], 3)
        self.assertEqual(context["included_item_count"], 2)
        self.assertEqual(
            context["items"],
            ['{"schedule":[1]}', '{"schedule":[2]}'],
        )
        self.assertTrue(context["truncated"])

    def test_worker_filters_retained_proposal_options(self):
        self.config.prompt.proposal_neighborhood_artifact = (
            "proposal-neighborhood.json"
        )
        self.config.prompt.proposal_options_max_items = 1
        parent_artifacts = {
            "program-structure.txt": "parent",
            "proposal-neighborhood.json": json.dumps(
                {
                    "schema_version": 1,
                    "identity_artifact": "program-structure.txt",
                    "options": [
                        {"id": "occupied", "identity": "held", "edit": "skip"},
                        {"id": "first", "identity": "new-a", "edit": "use a"},
                        {"id": "second", "identity": "new-b", "edit": "use b"},
                    ],
                }
            ),
        }
        snapshot = {
            "artifacts": {
                "parent": {"program-structure.txt": "parent"},
                "other": {"program-structure.txt": "held"},
            }
        }

        with patch.object(
            process_parallel_module,
            "_worker_config",
            self.config,
            create=True,
        ):
            augmented = _with_archive_context(parent_artifacts, snapshot)

        context = json.loads(augmented["proposal-options.json"])
        self.assertEqual(context["declared_option_count"], 3)
        self.assertEqual(context["excluded_retained_count"], 1)
        self.assertEqual(context["available_option_count"], 2)
        self.assertEqual(context["included_option_count"], 1)
        self.assertEqual(context["options"][0]["id"], "first")
        self.assertTrue(context["truncated"])

    def test_worker_filters_options_from_complete_measured_history(self):
        self.config.prompt.proposal_neighborhood_artifact = (
            "proposal-neighborhood.json"
        )
        parent_artifacts = {
            "program-structure.txt": "parent",
            "proposal-neighborhood.json": json.dumps(
                {
                    "schema_version": 1,
                    "identity_artifact": "program-structure.txt",
                    "options": [
                        {"id": "live", "identity": "held", "edit": "skip"},
                        {
                            "id": "displaced",
                            "identity": "measured-before",
                            "edit": "skip",
                        },
                        {"id": "fresh", "identity": "new", "edit": "use"},
                    ],
                }
            ),
        }
        snapshot = {
            "artifacts": {
                "parent": {"program-structure.txt": "parent"},
                "other": {"program-structure.txt": "held"},
            },
            "historical_artifact_values": {
                "program-structure.txt": [
                    "parent",
                    "held",
                    "measured-before",
                ]
            },
        }

        with patch.object(
            process_parallel_module,
            "_worker_config",
            self.config,
            create=True,
        ):
            augmented = _with_archive_context(parent_artifacts, snapshot)

        context = json.loads(augmented["proposal-options.json"])
        self.assertEqual(context["excluded_retained_count"], 1)
        self.assertEqual(context["excluded_historical_count"], 2)
        self.assertEqual(context["excluded_known_count"], 2)
        self.assertEqual(
            [option["id"] for option in context["options"]],
            ["fresh"],
        )

    def test_worker_rejects_malformed_proposal_neighborhood(self):
        self.config.prompt.proposal_neighborhood_artifact = (
            "proposal-neighborhood.json"
        )
        with patch.object(
            process_parallel_module,
            "_worker_config",
            self.config,
            create=True,
        ):
            with self.assertRaisesRegex(ValueError, "schema_version 1"):
                _with_archive_context(
                    {"proposal-neighborhood.json": "{}"},
                    {"artifacts": {}},
                )

    def test_proposal_totals_include_rejections_without_live_scheduler(self):
        controller = ProcessParallelController(
            self.config, self.eval_file, self.database
        )

        controller._record_controller_result(
            0, SerializableResult(error="invalid diff")
        )
        controller._record_controller_result(
            0,
            SerializableResult(
                child_program_dict={"id": "child", "code": "pass"}
            ),
        )

        self.assertEqual(controller.llm_usage["rejected_proposals"], 1)
        self.assertEqual(controller.llm_usage["accepted_proposals"], 1)

    def test_live_archive_rejects_duplicate_program_identity(self):
        self.config.program_identity_artifact = "program-identity.txt"
        self.database.store_artifacts(
            "test_0",
            {"program-identity.txt": "existing-program"},
        )
        usage_path = Path(self.test_dir) / "identity-usage.jsonl"
        controller = ProcessParallelController(
            self.config,
            self.eval_file,
            self.database,
            usage_output_path=str(usage_path),
        )
        duplicate = SerializableResult(
            child_program_dict={"id": "duplicate", "code": "pass"},
            parent_id="test_0",
            llm_response="duplicate proposal",
            artifacts={"program-identity.txt": "existing-program"},
            target_island=2,
            llm_metadata={
                "provider_response_id": "identity-duplicate",
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            },
        )
        first_new = SerializableResult(
            child_program_dict={"id": "new", "code": "pass"},
            artifacts={"program-identity.txt": "new-program"},
        )
        concurrent_duplicate = SerializableResult(
            child_program_dict={"id": "duplicate-new", "code": "pass"},
            artifacts={"program-identity.txt": "new-program"},
        )

        controller._apply_program_identity_gate(duplicate)
        controller._record_controller_result(2, duplicate)
        controller._record_llm_usage(4, duplicate)
        controller._apply_program_identity_gate(first_new)
        controller._apply_program_identity_gate(concurrent_duplicate)

        self.assertIn("already exists", duplicate.error)
        self.assertIsNone(first_new.error)
        self.assertIn("already exists", concurrent_duplicate.error)
        receipt = json.loads(usage_path.read_text(encoding="utf-8"))
        self.assertFalse(receipt["accepted_for_evaluation"])
        self.assertFalse(receipt["accepted_for_archive"])
        self.assertTrue(receipt["candidate_measured"])
        self.assertIn("already exists", receipt["rejection_reason"])
        self.assertEqual(receipt["parent_id"], "test_0")
        self.assertEqual(
            receipt["llm_response_sha256"],
            "e7eaec4f27fa10c6425a346a7697c42320588cfc59bc6b"
            "ac458a5ddfb268c7a1",
        )
        self.assertEqual(receipt["target_island"], 2)
        self.assertEqual(receipt["cumulative_usage"]["rejected_proposals"], 1)
        self.assertEqual(
            receipt["cumulative_usage"]["measured_artifact_history"],
            {"program-identity.txt": 1},
        )

    def test_live_archive_rejects_duplicate_phenotype_identity(self):
        self.config.phenotype_identity_artifact = "phenotype-identity.txt"
        self.database.store_artifacts(
            "test_0",
            {"phenotype-identity.txt": "existing-phenotype"},
        )
        controller = ProcessParallelController(
            self.config,
            self.eval_file,
            self.database,
        )
        duplicate = SerializableResult(
            child_program_dict={"id": "duplicate", "code": "different source"},
            artifacts={"phenotype-identity.txt": "existing-phenotype"},
        )
        novel = SerializableResult(
            child_program_dict={"id": "novel", "code": "different source"},
            artifacts={"phenotype-identity.txt": "novel-phenotype"},
        )

        controller._apply_program_identity_gate(duplicate)
        controller._apply_program_identity_gate(novel)

        self.assertIn("phenotype identity already exists", duplicate.error)
        self.assertIsNone(novel.error)

    def test_measured_history_survives_behavior_rejection_and_snapshot(self):
        self.config.program_identity_artifact = "program-identity.txt"
        self.config.phenotype_identity_artifact = "phenotype-identity.txt"
        self.config.prompt.proposal_neighborhood_artifact = (
            "proposal-neighborhood.json"
        )
        neighborhood = json.dumps(
            {
                "schema_version": 1,
                "identity_artifact": "program-identity.txt",
                "options": [],
            }
        )
        self.database.store_artifacts(
            "test_0",
            {
                "program-identity.txt": "baseline-program",
                "phenotype-identity.txt": "baseline-behavior",
                "proposal-neighborhood.json": neighborhood,
            },
        )
        controller = ProcessParallelController(
            self.config, self.eval_file, self.database
        )
        measured = SerializableResult(
            child_program_dict={"id": "measured", "code": "pass"},
            artifacts={
                "program-identity.txt": "new-program",
                "phenotype-identity.txt": "baseline-behavior",
                "proposal-neighborhood.json": neighborhood,
            },
        )

        controller._record_historical_artifacts(measured.artifacts)
        controller._apply_program_identity_gate(measured)
        snapshot = controller._create_database_snapshot()

        self.assertIn("phenotype identity already exists", measured.error)
        self.assertIn(
            "new-program",
            snapshot["historical_artifact_values"]["program-identity.txt"],
        )
        self.assertIn(
            "new-program",
            self.database.historical_artifact_values[
                "program-identity.txt"
            ],
        )

    def test_controller_start_stop(self):
        """Test starting and stopping the controller"""
        controller = ProcessParallelController(self.config, self.eval_file, self.database)

        # Start controller
        controller.start()
        self.assertIsNotNone(controller.executor)

        # Stop controller
        controller.stop()
        self.assertIsNone(controller.executor)
        self.assertTrue(controller.shutdown_event.is_set())

    def test_controller_stop_terminates_running_workers(self):
        """Stopping the controller does not wait for stuck process-pool work."""
        controller = ProcessParallelController(self.config, self.eval_file, self.database)
        executor = ProcessPoolExecutor(max_workers=1)
        controller.executor = executor
        marker_path = os.path.join(self.test_dir, "worker.pid")
        future = executor.submit(_slow_test_worker, marker_path)

        deadline = time.monotonic() + 5
        while not os.path.exists(marker_path) and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(os.path.exists(marker_path))
        worker_pid = int(Path(marker_path).read_text())

        started = time.monotonic()
        controller.stop()
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 1)
        self.assertIsNone(controller.executor)
        self.assertTrue(controller.shutdown_event.is_set())
        deadline = time.monotonic() + 1
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(future.done())
        with self.assertRaises(ProcessLookupError):
            os.kill(worker_pid, 0)

        # Cleanup is idempotent after the executor reference is cleared.
        controller.stop()

    def test_process_pool_shutdown_escalates_to_kill(self):
        """Workers still alive after terminate are killed before returning."""
        process = Mock()
        process.is_alive.return_value = True
        executor = Mock(spec=["_processes", "shutdown"])
        executor._processes = {123: process}

        with patch.object(
            process_parallel_module,
            "_wait_for_processes",
            side_effect=[[process], []],
        ):
            process_parallel_module._terminate_process_pool(executor)

        executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()

    def test_database_snapshot_creation(self):
        """Test creating database snapshot for workers"""
        controller = ProcessParallelController(self.config, self.eval_file, self.database)

        snapshot = controller._create_database_snapshot()

        # Verify snapshot structure
        self.assertIn("programs", snapshot)
        self.assertIn("islands", snapshot)
        self.assertIn("current_island", snapshot)
        self.assertIn("artifacts", snapshot)

        # Verify programs are serialized
        self.assertEqual(len(snapshot["programs"]), 3)
        for pid, prog_dict in snapshot["programs"].items():
            self.assertIsInstance(prog_dict, dict)
            self.assertIn("id", prog_dict)
            self.assertIn("code", prog_dict)

    def test_database_checkpoint_preserves_measured_artifact_history(self):
        checkpoint = Path(self.test_dir) / "history-checkpoint"
        self.database.historical_artifact_values = {
            "program-identity.txt": {"first", "displaced"}
        }

        self.database.save(str(checkpoint), iteration=7)
        restored = ProgramDatabase(self.config.database)
        restored.load(str(checkpoint))

        self.assertEqual(
            restored.historical_artifact_values,
            {"program-identity.txt": {"first", "displaced"}},
        )

    def test_run_evolution_basic(self):
        """Test basic evolution run"""

        async def run_test():
            controller = ProcessParallelController(self.config, self.eval_file, self.database)

            # Mock the executor to avoid actually spawning processes
            with patch.object(controller, "_submit_iteration") as mock_submit:
                # Create mock futures that complete immediately
                mock_future1 = MagicMock()
                mock_result1 = SerializableResult(
                    child_program_dict={
                        "id": "child_1",
                        "code": "def evolved(): return 1",
                        "language": "python",
                        "parent_id": "test_0",
                        "generation": 1,
                        "metrics": {"score": 0.7, "performance": 0.8},
                        "iteration_found": 1,
                        "metadata": {"changes": "test", "island": 0},
                    },
                    parent_id="test_0",
                    iteration_time=0.1,
                    iteration=1,
                )
                mock_future1.done.return_value = True
                mock_future1.result.return_value = mock_result1
                mock_future1.cancel.return_value = True

                mock_submit.return_value = mock_future1

                # Start controller
                controller.start()

                # Run evolution for 1 iteration
                result = await controller.run_evolution(
                    start_iteration=1, max_iterations=1, target_score=None
                )

                # Verify iteration was submitted with island_id
                mock_submit.assert_called_once_with(1, 0)

                # Verify program was added to database
                self.assertIn("child_1", self.database.programs)
                child = self.database.get("child_1")
                self.assertEqual(child.metrics["score"], 0.7)

        # Run the async test
        asyncio.run(run_test())

    def test_target_score_drains_submitted_batch_without_new_work(self):
        """A reached target records the bounded in-flight batch and its true cursor."""

        async def run_test():
            controller = ProcessParallelController(
                self.config, self.eval_file, self.database
            )
            controller.executor = Mock()
            futures = {}
            for iteration in (1, 2, 3):
                future = MagicMock()
                future.done.return_value = True
                future.result.return_value = SerializableResult(
                    child_program_dict={
                        "id": f"target_child_{iteration}",
                        "code": f"def target_{iteration}(): return {iteration}",
                        "language": "python",
                        "parent_id": "test_0",
                        "generation": 1,
                        "metrics": {"combined_score": 1.0},
                        "iteration_found": iteration,
                        "metadata": {"changes": "target", "island": iteration - 1},
                    },
                    parent_id="test_0",
                    iteration=iteration,
                    target_island=iteration - 1,
                )
                futures[iteration] = future

            with patch.object(
                controller,
                "_submit_iteration",
                side_effect=lambda iteration, island_id: futures[iteration],
            ) as submit:
                await controller.run_evolution(
                    start_iteration=1,
                    max_iterations=6,
                    target_score=1.0,
                )

            self.assertEqual(submit.call_count, 3)
            self.assertTrue(controller.target_score_reached)
            self.assertEqual(controller.completion_reason, "target_score_reached")
            self.assertEqual(controller.completed_iteration_count, 3)
            self.assertEqual(controller.last_completed_iteration, 3)
            self.assertEqual(self.database.last_iteration, 3)
            for iteration in (1, 2, 3):
                self.assertIn(f"target_child_{iteration}", self.database.programs)

        asyncio.run(run_test())

    def test_llm_call_budget_reserves_exactly_and_stops_refill(self):
        """The logical call reservation cap never submits excess proposals."""

        async def run_test():
            self.config.max_llm_calls = 1
            controller = ProcessParallelController(
                self.config, self.eval_file, self.database
            )
            controller.executor = Mock()
            futures = {}
            for iteration in (1, 2, 3):
                future = MagicMock()
                future.done.return_value = True
                future.result.return_value = SerializableResult(
                    error="proposal rejected after generation",
                    iteration=iteration,
                    llm_metadata={
                        "provider_response_id": f"call-{iteration}",
                        "model": "test-model",
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                    },
                )
                futures[iteration] = future

            with patch.object(
                controller,
                "_submit_iteration",
                side_effect=lambda iteration, island_id: futures[iteration],
            ) as submit:
                await controller.run_evolution(
                    start_iteration=1,
                    max_iterations=6,
                )

            self.assertEqual(submit.call_count, 1)
            self.assertEqual(controller.completion_reason, "max_llm_calls_reached")
            self.assertEqual(controller.completed_iteration_count, 1)
            self.assertEqual(
                controller.llm_usage,
                {
                    "call_budget_counting_basis": "submitted_proposals",
                    "provider_usage_counting_basis": "unique_provider_receipts",
                    "llm_calls_submitted": 1,
                    "llm_calls": 1,
                    "accepted_proposals": 0,
                    "rejected_proposals": 1,
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_provider_tokens": 15,
                    "unreported_provider_token_calls": 0,
                    "max_llm_calls": 1,
                    "max_total_provider_tokens": None,
                    "llm_call_budget_overshoot": 0,
                    "provider_token_budget_overshoot": 0,
                    "limits_reached": ["max_llm_calls"],
                    "inflight_proposals_at_budget_stop": 1,
                    "measured_artifact_history": {},
                    "controller_allocation": {
                        "enabled": False,
                        "score_metric": "combined_score",
                        "diversity_artifact": None,
                        "leader_score_band": None,
                        "parent_score_band": None,
                        "reserve_islands": 1,
                        "active_parallelism": 3,
                        "adaptive_parallelism": 3,
                        "islands": [
                            {
                                "island": 0,
                                "calls": 0,
                                "accepted": 0,
                                "rejected": 0,
                                "distinct_behaviors": 0,
                                "best_score": 0.45,
                                "tokens": 0,
                            },
                            {
                                "island": 1,
                                "calls": 0,
                                "accepted": 0,
                                "rejected": 0,
                                "distinct_behaviors": 0,
                                "best_score": 0.55,
                                "tokens": 0,
                            },
                            {
                                "island": 2,
                                "calls": 0,
                                "accepted": 0,
                                "rejected": 0,
                                "distinct_behaviors": 0,
                                "best_score": 0.65,
                                "tokens": 0,
                            },
                        ],
                    },
                },
            )

        asyncio.run(run_test())

    def test_provider_token_budget_stops_refill_then_counts_drained_receipts(self):
        """Token usage can cross the cap once and includes every in-flight receipt."""

        async def run_test():
            self.config.max_total_provider_tokens = 20
            controller = ProcessParallelController(
                self.config, self.eval_file, self.database
            )
            controller.executor = Mock()
            futures = {}
            for iteration in (1, 2, 3, 4):
                future = MagicMock()
                future.done.return_value = True
                future.result.return_value = SerializableResult(
                    error="proposal rejected after generation",
                    iteration=iteration,
                    llm_metadata={
                        "provider_response_id": f"token-call-{iteration}",
                        "model": "test-model",
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                    },
                )
                futures[iteration] = future

            with patch.object(
                controller,
                "_submit_iteration",
                side_effect=lambda iteration, island_id: futures[iteration],
            ) as submit:
                await controller.run_evolution(
                    start_iteration=1,
                    max_iterations=8,
                )

            self.assertEqual(submit.call_count, 4)
            self.assertEqual(
                controller.completion_reason,
                "max_total_provider_tokens_reached",
            )
            self.assertEqual(controller.llm_usage["llm_calls"], 4)
            self.assertEqual(controller.llm_usage["llm_calls_submitted"], 4)
            self.assertEqual(controller.llm_usage["total_provider_tokens"], 60)
            self.assertEqual(controller.llm_usage["provider_token_budget_overshoot"], 40)
            self.assertEqual(
                controller.llm_usage["inflight_proposals_at_budget_stop"],
                2,
            )

        asyncio.run(run_test())

    def test_llm_call_budget_stops_exactly_when_refill_reserves_last_call(self):
        async def run_test():
            self.config.max_llm_calls = 4
            controller = ProcessParallelController(
                self.config, self.eval_file, self.database
            )
            controller.executor = Mock()
            futures = {}
            for iteration in (1, 2, 3, 4):
                future = MagicMock()
                future.done.return_value = True
                future.result.return_value = SerializableResult(
                    error="proposal rejected after generation",
                    iteration=iteration,
                    llm_metadata={
                        "provider_response_id": f"refill-call-{iteration}",
                        "usage": {"total_tokens": 10},
                    },
                )
                futures[iteration] = future

            with patch.object(
                controller,
                "_submit_iteration",
                side_effect=lambda iteration, island_id: futures[iteration],
            ) as submit:
                await controller.run_evolution(
                    start_iteration=1,
                    max_iterations=8,
                )

            self.assertEqual(submit.call_count, 4)
            self.assertEqual(controller.completion_reason, "max_llm_calls_reached")
            self.assertEqual(controller.llm_usage["llm_calls_submitted"], 4)
            self.assertEqual(controller.llm_usage["llm_calls"], 4)
            self.assertEqual(controller.llm_usage["llm_call_budget_overshoot"], 0)
            self.assertEqual(
                controller.llm_usage["inflight_proposals_at_budget_stop"],
                3,
            )

        asyncio.run(run_test())

    def test_usage_aggregation_requires_unique_provider_receipts(self):
        """Duplicate or unidentified metadata cannot consume a budget twice."""
        controller = ProcessParallelController(
            self.config, self.eval_file, self.database
        )

        controller._record_llm_usage(
            1,
            SerializableResult(
                llm_metadata={
                    "provider_response_id": "receipt-1",
                    "usage": {
                        "prompt_tokens": 6,
                        "completion_tokens": 4,
                        "total_tokens": 10,
                    },
                }
            ),
        )
        controller._record_llm_usage(
            2,
            SerializableResult(
                llm_metadata={
                    "provider_response_id": "receipt-1",
                    "usage": {"total_tokens": 10},
                }
            ),
        )
        controller._record_llm_usage(
            3,
            SerializableResult(
                llm_metadata={
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 100,
                        "total_tokens": 200,
                    },
                }
            ),
        )
        controller._record_llm_usage(
            4,
            SerializableResult(
                llm_metadata={
                    "provider_response_id": "receipt-2",
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 4,
                        "total_tokens": None,
                    },
                }
            ),
        )
        controller._record_llm_usage(
            5,
            SerializableResult(
                llm_metadata={
                    "provider_response_id": "receipt-3",
                    "usage": {},
                }
            ),
        )

        self.assertEqual(controller.llm_usage["llm_calls"], 3)
        self.assertEqual(controller.llm_usage["prompt_tokens"], 9)
        self.assertEqual(controller.llm_usage["completion_tokens"], 8)
        self.assertEqual(controller.llm_usage["total_provider_tokens"], 17)
        self.assertEqual(controller.llm_usage["unreported_provider_token_calls"], 1)

    def test_same_receipt_can_reach_both_budgets(self):
        self.config.max_llm_calls = 1
        self.config.max_total_provider_tokens = 10
        controller = ProcessParallelController(
            self.config, self.eval_file, self.database
        )

        reason = controller._register_submitted_proposal(inflight_proposals=1)
        self.assertEqual(reason, "max_llm_calls_reached")

        reason = controller._record_llm_usage(
            1,
            SerializableResult(
                llm_metadata={
                    "provider_response_id": "both-limits",
                    "usage": {"total_tokens": 12},
                }
            ),
            inflight_proposals=0,
        )

        self.assertEqual(
            reason,
            "max_llm_calls_reached",
        )
        self.assertEqual(
            controller.llm_usage["limits_reached"],
            ["max_llm_calls", "max_total_provider_tokens"],
        )
        self.assertEqual(controller.llm_usage["llm_call_budget_overshoot"], 0)
        self.assertEqual(controller.llm_usage["provider_token_budget_overshoot"], 2)
        self.assertEqual(controller.llm_usage["inflight_proposals_at_budget_stop"], 1)

    def test_target_score_takes_precedence_when_same_receipt_reaches_budget(self):
        """A solved target remains the primary run outcome while usage stays visible."""

        async def run_test():
            self.config.max_llm_calls = 1
            controller = ProcessParallelController(
                self.config, self.eval_file, self.database
            )
            controller.executor = Mock()
            futures = {}
            for iteration in (1, 2, 3):
                future = MagicMock()
                future.done.return_value = True
                future.result.return_value = SerializableResult(
                    child_program_dict={
                        "id": f"budget_target_child_{iteration}",
                        "code": f"def target_{iteration}(): return {iteration}",
                        "language": "python",
                        "parent_id": "test_0",
                        "generation": 1,
                        "metrics": {"combined_score": 1.0},
                        "iteration_found": iteration,
                        "metadata": {"changes": "target", "island": iteration - 1},
                    },
                    parent_id="test_0",
                    iteration=iteration,
                    target_island=iteration - 1,
                    llm_metadata={
                        "provider_response_id": f"target-call-{iteration}",
                        "usage": {"total_tokens": 10},
                    },
                )
                futures[iteration] = future

            with patch.object(
                controller,
                "_submit_iteration",
                side_effect=lambda iteration, island_id: futures[iteration],
            ):
                await controller.run_evolution(
                    start_iteration=1,
                    max_iterations=6,
                    target_score=1.0,
                )

            self.assertEqual(controller.completion_reason, "target_score_reached")
            self.assertEqual(controller.llm_usage["limits_reached"], ["max_llm_calls"])
            self.assertEqual(controller.llm_usage["llm_calls_submitted"], 1)
            self.assertEqual(controller.llm_usage["llm_call_budget_overshoot"], 0)

        asyncio.run(run_test())

    def test_evaluator_error_preserves_completed_model_call_usage(self):
        """A post-generation evaluator error still produces a provider receipt."""

        class FakeLLM:
            last_call_metadata = {
                "provider_response_id": "response-after-evaluator-error",
                "model": "fake-model",
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                },
            }

            async def generate_with_context(self, **_kwargs):
                return "\n".join(
                    [
                        "<" * 7 + " SEARCH",
                        "    return 1",
                        "=" * 7,
                        "    return 2",
                        ">" * 7 + " REPLACE",
                    ]
                )

        class FailingEvaluator:
            async def evaluate_program(self, _code, _child_id):
                raise RuntimeError("measurement failed")

        class FakePromptSampler:
            def build_prompt(self, **_kwargs):
                return {"system": "system", "user": "user"}

        parent = Program(
            id="parent",
            code="def solve():\n    return 1\n",
            language="python",
            metrics={"combined_score": 0.5},
            metadata={"island": 0},
        )
        snapshot = {
            "programs": {"parent": parent.to_dict()},
            "artifacts": {},
            "current_island": 0,
            "islands": [["parent"]],
            "feature_dimensions": [],
            "sampling_island": 0,
        }

        with (
            patch.object(
                process_parallel_module,
                "_lazy_init_worker_components",
                return_value=None,
            ),
            patch.object(
                process_parallel_module,
                "_worker_config",
                self.config,
                create=True,
            ),
            patch.object(
                process_parallel_module,
                "_worker_llm_ensemble",
                FakeLLM(),
                create=True,
            ),
            patch.object(
                process_parallel_module,
                "_worker_evaluator",
                FailingEvaluator(),
                create=True,
            ),
            patch.object(
                process_parallel_module,
                "_worker_prompt_sampler",
                FakePromptSampler(),
                create=True,
            ),
        ):
            result = process_parallel_module._run_iteration_worker(
                7, snapshot, "parent", []
            )

        self.assertIn("measurement failed", result.error)
        self.assertEqual(
            result.llm_metadata["provider_response_id"],
            "response-after-evaluator-error",
        )
        usage_path = Path(self.test_dir) / "usage.jsonl"
        controller = ProcessParallelController(
            self.config,
            self.eval_file,
            self.database,
            usage_output_path=str(usage_path),
        )
        controller._record_llm_usage(7, result)
        receipt = json.loads(usage_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["iteration"], 7)
        self.assertFalse(receipt["accepted_for_evaluation"])
        self.assertEqual(
            receipt["provider_response_id"], "response-after-evaluator-error"
        )

    def test_structured_evaluator_failure_is_not_added_to_an_island(self):
        """A measured failure remains a rejected proposal with recorded usage."""

        class FakeLLM:
            last_call_metadata = {
                "provider_response_id": "response-with-structured-failure",
                "model": "fake-model",
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                },
            }

            async def generate_with_context(self, **_kwargs):
                return "\n".join(
                    [
                        "<" * 7 + " SEARCH",
                        "    return 1",
                        "=" * 7,
                        "    return 2",
                        ">" * 7 + " REPLACE",
                    ]
                )

        class FailedEvaluator:
            async def evaluate_program(self, _code, _child_id):
                return {"__evaluation_failed__": 1.0, "error": 0.0}

            def get_pending_artifacts(self, _child_id):
                return {"stderr": "invalid candidate"}

        class FakePromptSampler:
            def build_prompt(self, **_kwargs):
                return {"system": "system", "user": "user"}

        parent = Program(
            id="parent",
            code="def solve():\n    return 1\n",
            language="python",
            metrics={"combined_score": 0.5},
            metadata={"island": 0},
        )
        snapshot = {
            "programs": {"parent": parent.to_dict()},
            "artifacts": {},
            "current_island": 0,
            "islands": [["parent"]],
            "feature_dimensions": ["combined_score"],
            "sampling_island": 0,
        }

        with (
            patch.object(
                process_parallel_module,
                "_lazy_init_worker_components",
                return_value=None,
            ),
            patch.object(
                process_parallel_module,
                "_worker_config",
                self.config,
                create=True,
            ),
            patch.object(
                process_parallel_module,
                "_worker_llm_ensemble",
                FakeLLM(),
                create=True,
            ),
            patch.object(
                process_parallel_module,
                "_worker_evaluator",
                FailedEvaluator(),
                create=True,
            ),
            patch.object(
                process_parallel_module,
                "_worker_prompt_sampler",
                FakePromptSampler(),
                create=True,
            ),
        ):
            result = process_parallel_module._run_iteration_worker(
                8, snapshot, "parent", []
            )

        self.assertIsNone(result.child_program_dict)
        self.assertIn("invalid candidate", result.error)
        self.assertEqual(
            result.llm_metadata["provider_response_id"],
            "response-with-structured-failure",
        )

    def test_request_shutdown(self):
        """Test graceful shutdown request"""
        controller = ProcessParallelController(self.config, self.eval_file, self.database)

        # Request shutdown
        controller.request_shutdown()

        # Verify shutdown event is set
        self.assertTrue(controller.shutdown_event.is_set())

    def test_serializable_result(self):
        """Test SerializableResult dataclass"""
        result = SerializableResult(
            child_program_dict={"id": "test", "code": "pass"},
            parent_id="parent",
            iteration_time=1.5,
            iteration=10,
            error=None,
        )

        # Verify attributes
        self.assertEqual(result.child_program_dict["id"], "test")
        self.assertEqual(result.parent_id, "parent")
        self.assertEqual(result.iteration_time, 1.5)
        self.assertEqual(result.iteration, 10)
        self.assertIsNone(result.error)

        # Test with error
        error_result = SerializableResult(error="Test error", iteration=5)
        self.assertEqual(error_result.error, "Test error")
        self.assertIsNone(error_result.child_program_dict)


if __name__ == "__main__":
    unittest.main()
