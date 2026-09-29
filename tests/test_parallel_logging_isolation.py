"""
Regression tests for concurrent OpenEvolve logging isolation (#295).

Multiple OpenEvolve instances used to attach handlers to the root logger with
no per-run filter, so each run duplicated the others' log lines into every
handler (and every log file).
"""

import logging
import os
import tempfile
import unittest

from openevolve.config import Config
from openevolve.controller import OpenEvolve, _active_run_id


class TestParallelLoggingIsolation(unittest.TestCase):
    def setUp(self):
        self._root_handlers = list(logging.root.handlers)
        self._pkg = logging.getLogger("openevolve")
        self._pkg_handlers = list(self._pkg.handlers)
        self._pkg_propagate = self._pkg.propagate
        self._prev_run_id = _active_run_id.set(None)
        self._dirs = []

    def tearDown(self):
        # Restore package logger to a clean state for other tests.
        for handler in list(self._pkg.handlers):
            if handler not in self._pkg_handlers:
                self._pkg.removeHandler(handler)
                handler.close()
        self._pkg.propagate = self._pkg_propagate
        _active_run_id.reset(self._prev_run_id)
        import shutil

        for path in self._dirs:
            shutil.rmtree(path, ignore_errors=True)

    def _stub(self, name: str) -> OpenEvolve:
        stub = OpenEvolve.__new__(OpenEvolve)
        stub.config = Config()
        stub.config.log_level = "INFO"
        stub.output_dir = tempfile.mkdtemp(prefix=f"oe_{name}_")
        self._dirs.append(stub.output_dir)
        return stub

    def _log_file_text(self, instance: OpenEvolve) -> str:
        log_dir = os.path.join(instance.output_dir, "logs")
        files = os.listdir(log_dir)
        self.assertEqual(len(files), 1, f"expected one log file, got {files}")
        with open(os.path.join(log_dir, files[0]), encoding="utf-8") as handle:
            return handle.read()

    def test_parallel_instances_do_not_cross_write_log_files(self):
        """Each run's file must contain only that run's messages (#295)."""
        root_before = len(logging.root.handlers)

        a = self._stub("a")
        b = self._stub("b")
        OpenEvolve._setup_logging(a)
        OpenEvolve._setup_logging(b)

        # Must not attach handlers to the root logger.
        self.assertEqual(len(logging.root.handlers), root_before)

        pkg_logger = logging.getLogger("openevolve.controller")

        token_a = a._activate_run_logging()
        try:
            pkg_logger.info("MESSAGE_FROM_RUN_A")
        finally:
            _active_run_id.reset(token_a)

        token_b = b._activate_run_logging()
        try:
            pkg_logger.info("MESSAGE_FROM_RUN_B")
        finally:
            _active_run_id.reset(token_b)

        text_a = self._log_file_text(a)
        text_b = self._log_file_text(b)

        self.assertIn("MESSAGE_FROM_RUN_A", text_a)
        self.assertNotIn("MESSAGE_FROM_RUN_B", text_a)
        self.assertIn("MESSAGE_FROM_RUN_B", text_b)
        self.assertNotIn("MESSAGE_FROM_RUN_A", text_b)

        a._cleanup_logging()
        b._cleanup_logging()

    def test_cleanup_removes_instance_handlers(self):
        instance = self._stub("cleanup")
        before = len(logging.getLogger("openevolve").handlers)
        OpenEvolve._setup_logging(instance)
        self.assertGreater(len(logging.getLogger("openevolve").handlers), before)
        instance._cleanup_logging()
        self.assertEqual(len(logging.getLogger("openevolve").handlers), before)


if __name__ == "__main__":
    unittest.main()
