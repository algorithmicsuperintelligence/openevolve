"""
Regression and unit tests for concurrent OpenEvolve logging isolation (GitHub issue #295).

Historically, multiple OpenEvolve instances attached their FileHandler and StreamHandler
to the root logger (logging.getLogger()) with propagate enabled and no run isolation filter.
As a result:
1. All parallel or sequential runs cross-wrote into each other's log files.
2. Console output was duplicated for every instantiated controller.
3. The process-wide root logger was polluted with OpenEvolve handlers, affecting host applications.
4. Handlers and open file descriptors leaked because they were never detached or closed.

These tests verify that:
- Logging is attached to the 'openevolve' package logger with propagate=False.
- The root logger remains untouched.
- Handlers use RunIsolationFilter to only process records from their matching run context.
- Concurrent asyncio runs do not cross-write log files or duplicate console output.
- Handlers are cleanly detached and closed when run() completes or close() is called.
"""

import asyncio
import io
import logging
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from openevolve.config import Config
from openevolve.controller import OpenEvolve
from openevolve.utils.logging_utils import (
    LoggerPrefixFilter,
    RunIsolationFilter,
    current_run_id,
    get_current_run_id,
    reset_current_run_id,
    set_current_run_id,
)


class TestRunIsolationFilterUnit(unittest.TestCase):
    """Unit tests for RunIsolationFilter and contextvar helpers."""

    def setUp(self):
        self.orig_token = current_run_id.set(None)

    def tearDown(self):
        current_run_id.reset(self.orig_token)

    def test_filter_matches_active_run_id(self):
        """Filter should accept records when the active run context matches self.run_id."""
        filt = RunIsolationFilter("run-abc")
        token = set_current_run_id("run-abc")
        try:
            record = logging.LogRecord(
                "openevolve.controller", logging.INFO, "path", 1, "msg", (), None
            )
            self.assertTrue(filt.filter(record))
        finally:
            reset_current_run_id(token)

    def test_filter_rejects_different_run_id(self):
        """Filter should reject records when the active run context belongs to another run."""
        filt = RunIsolationFilter("run-abc")
        token = set_current_run_id("run-xyz")
        try:
            record = logging.LogRecord(
                "openevolve.controller", logging.INFO, "path", 1, "msg", (), None
            )
            self.assertFalse(filt.filter(record))
        finally:
            reset_current_run_id(token)

    def test_filter_allows_unscoped_records(self):
        """Filter should allow records when no active run context is set (e.g. general utility logs)."""
        filt = RunIsolationFilter("run-abc")
        set_current_run_id(None)
        record = logging.LogRecord(
            "openevolve.controller", logging.INFO, "path", 1, "msg", (), None
        )
        self.assertTrue(filt.filter(record))

    def test_get_and_set_current_run_id_helpers(self):
        """Verify helper functions get_current_run_id, set_current_run_id, and reset_current_run_id."""
        self.assertIsNone(get_current_run_id())
        tok = set_current_run_id("test-id-123")
        self.assertEqual(get_current_run_id(), "test-id-123")
        reset_current_run_id(tok)
        self.assertIsNone(get_current_run_id())


class TestParallelLoggingIsolation(unittest.TestCase):
    """Integration and regression tests for per-run logging isolation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="oe295_")
        self.saved_root_handlers = list(logging.getLogger().handlers)
        self.pkg_logger = logging.getLogger("openevolve")
        self.saved_pkg_handlers = list(self.pkg_logger.handlers)
        self.saved_pkg_level = self.pkg_logger.level
        self.saved_pkg_propagate = self.pkg_logger.propagate

    def tearDown(self):
        # Restore root logger handlers
        logging.getLogger().handlers = self.saved_root_handlers

        # Clean up any residual package logger handlers
        for handler in list(self.pkg_logger.handlers):
            if handler not in self.saved_pkg_handlers:
                self.pkg_logger.removeHandler(handler)
                try:
                    handler.close()
                except Exception:
                    pass
        self.pkg_logger.setLevel(self.saved_pkg_level)
        self.pkg_logger.propagate = self.saved_pkg_propagate

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_dummy_files(self, dirname: str):
        work_dir = os.path.join(self.tmpdir, dirname)
        os.makedirs(work_dir, exist_ok=True)
        program_path = os.path.join(work_dir, "initial.py")
        with open(program_path, "w", encoding="utf-8") as f:
            f.write("def run():\n    return 42\n")
        eval_path = os.path.join(work_dir, "eval.py")
        with open(eval_path, "w", encoding="utf-8") as f:
            f.write('def evaluate(p):\n    return {"combined_score": 1.0}\n')
        return program_path, eval_path, work_dir

    def _create_controller(self, name: str, **config_overrides) -> OpenEvolve:
        program_path, eval_path, work_dir = self._write_dummy_files(name)
        config = Config()
        config.random_seed = None
        config.log_dir = os.path.join(work_dir, "logs")
        for k, v in config_overrides.items():
            setattr(config, k, v)

        with (
            patch("openevolve.controller.LLMEnsemble"),
            patch("openevolve.controller.Evaluator"),
        ):
            controller = OpenEvolve(
                initial_program_path=program_path,
                evaluation_file=eval_path,
                config=config,
                output_dir=os.path.join(work_dir, "output"),
            )
        return controller

    def _read_single_log(self, controller: OpenEvolve) -> str:
        log_dir = controller.config.log_dir
        log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith(".log")]
        self.assertEqual(len(log_files), 1, f"Expected 1 log file in {log_dir}, found {log_files}")
        with open(log_files[0], "r", encoding="utf-8") as f:
            return f.read()

    def test_root_logger_is_not_polluted(self):
        """Constructing OpenEvolve must NOT attach handlers to the root logger (#295)."""
        root_count_before = len(logging.getLogger().handlers)
        controller = self._create_controller("root_test")
        try:
            root_count_after = len(logging.getLogger().handlers)
            self.assertEqual(
                root_count_after,
                root_count_before,
                "Root logger was polluted by OpenEvolve constructor handlers!",
            )
            self.assertFalse(
                self.pkg_logger.propagate,
                "openevolve package logger must have propagate=False",
            )
        finally:
            controller.close()

    def test_concurrent_runs_do_not_cross_write_logs(self):
        """Two concurrent OpenEvolve runs must only log to their respective log files."""
        c1 = self._create_controller("run_1")
        c2 = self._create_controller("run_2")

        try:
            # Simulate concurrent execution of two runs using asyncio
            logger = logging.getLogger("openevolve.controller")

            async def mock_run_1():
                token = set_current_run_id(c1.run_id)
                try:
                    logger.info("RUN_1_EVENT_START")
                    await asyncio.sleep(0.02)
                    logger.info("RUN_1_EVENT_END")
                finally:
                    reset_current_run_id(token)

            async def mock_run_2():
                token = set_current_run_id(c2.run_id)
                try:
                    logger.info("RUN_2_EVENT_START")
                    await asyncio.sleep(0.01)
                    logger.info("RUN_2_EVENT_END")
                finally:
                    reset_current_run_id(token)

            async def run_both():
                await asyncio.gather(mock_run_1(), mock_run_2())

            asyncio.run(run_both())

            # Verify contents of log file 1
            content1 = self._read_single_log(c1)
            self.assertIn("RUN_1_EVENT_START", content1)
            self.assertIn("RUN_1_EVENT_END", content1)
            self.assertNotIn("RUN_2_EVENT_START", content1)
            self.assertNotIn("RUN_2_EVENT_END", content1)

            # Verify contents of log file 2
            content2 = self._read_single_log(c2)
            self.assertIn("RUN_2_EVENT_START", content2)
            self.assertIn("RUN_2_EVENT_END", content2)
            self.assertNotIn("RUN_1_EVENT_START", content2)
            self.assertNotIn("RUN_1_EVENT_END", content2)

        finally:
            c1.close()
            c2.close()

    def test_console_output_is_not_duplicated_across_instances(self):
        """StreamHandler must not emit duplicate lines when multiple controllers are active."""
        c1 = self._create_controller("c1")
        c2 = self._create_controller("c2")

        try:
            # Capture console emissions on the console handlers
            output_stream = io.StringIO()
            for handler in c1._log_handlers + c2._log_handlers:
                if isinstance(handler, logging.StreamHandler) and not isinstance(
                    handler, logging.FileHandler
                ):
                    handler.stream = output_stream

            logger = logging.getLogger("openevolve.controller")

            token = set_current_run_id(c1.run_id)
            try:
                logger.info("UNIQUE_C1_CONSOLE_MESSAGE")
            finally:
                reset_current_run_id(token)

            console_output = output_stream.getvalue()
            # The message should appear exactly once, not twice
            count = console_output.count("UNIQUE_C1_CONSOLE_MESSAGE")
            self.assertEqual(
                count, 1, f"Expected 1 console emission, got {count}: {console_output}"
            )

        finally:
            c1.close()
            c2.close()

    def test_cleanup_removes_and_closes_handlers(self):
        """close() must detach handlers from the package logger and close file descriptors."""
        initial_handlers_count = len(self.pkg_logger.handlers)
        c = self._create_controller("cleanup_test")

        self.assertGreater(len(self.pkg_logger.handlers), initial_handlers_count)
        self.assertEqual(len(c._log_handlers), 2)  # FileHandler + StreamHandler

        # Call close
        c.close()
        self.assertEqual(len(c._log_handlers), 0)
        self.assertEqual(len(self.pkg_logger.handlers), initial_handlers_count)

    def test_run_lifecycle_manages_logging_context_and_cleanup(self):
        """await controller.run() should set run context during execution and clean up handlers on completion."""
        c = self._create_controller("lifecycle_test")

        # Mock _run_evolution to record whether run context was active during execution
        captured_run_id = None

        async def fake_run_evolution(*args, **kwargs):
            nonlocal captured_run_id
            captured_run_id = get_current_run_id()
            logging.getLogger("openevolve.controller").info("LIFECYCLE_MESSAGE")
            return None

        with patch.object(c, "_run_evolution", side_effect=fake_run_evolution):
            # Run the controller
            asyncio.run(c.run(iterations=1))

        # ContextVar was correctly active during run
        self.assertEqual(captured_run_id, c.run_id)

        # ContextVar was reset after run
        self.assertIsNone(get_current_run_id())

        # Handlers were cleanly detached after run
        self.assertEqual(len(c._log_handlers), 0)

    def test_logger_prefix_does_not_stack(self):
        """LoggerPrefixFilter must not stack prefixes when records pass through multiple handlers."""
        filt1 = LoggerPrefixFilter("RUN1")
        filt2 = LoggerPrefixFilter("RUN2")

        record = logging.LogRecord(
            name="openevolve.controller",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="test msg",
            args=(),
            exc_info=None,
        )

        filt1.filter(record)
        self.assertEqual(record.name, "RUN1.openevolve.controller")

        filt2.filter(record)
        self.assertEqual(record.name, "RUN2.openevolve.controller")
        self.assertNotIn("RUN2.RUN1", record.name)


if __name__ == "__main__":
    unittest.main()
