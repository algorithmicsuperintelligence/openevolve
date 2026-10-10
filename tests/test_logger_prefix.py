"""
Tests for the optional logger prefix feature (GitHub issue #290).

A run can set ``Config.logger_prefix`` (or pass ``--logger-prefix`` on the CLI);
a handler-level ``LoggerPrefixFilter`` then rewrites ``record.name`` so every
downstream module logger shows up under the prefix, making it possible to tell
apart log output from multiple concurrent runs.

All tests are fully offline: ``LLMEnsemble``/``Evaluator`` are patched out and
only the constructor (which calls ``_setup_logging``) is exercised.
"""

import glob
import io
import logging
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from openevolve.config import Config
from openevolve.controller import OpenEvolve
from openevolve.utils.logging_utils import LoggerPrefixFilter


def _make_record(name: str) -> logging.LogRecord:
    return logging.LogRecord(name, logging.INFO, "path", 1, "msg", None, None)


class TestLoggerPrefixFilter(unittest.TestCase):
    """Unit tests for LoggerPrefixFilter"""

    def test_filter_rewrites_record_name(self):
        filt = LoggerPrefixFilter("run1")
        record = _make_record("openevolve.controller")
        self.assertTrue(filt.filter(record))
        self.assertEqual(record.name, "run1.openevolve.controller")

    def test_filter_is_idempotent(self):
        # The same record object flows through every handler attached to the
        # logger, so applying the filter twice must not stack the prefix.
        filt = LoggerPrefixFilter("run1")
        record = _make_record("openevolve.controller")
        self.assertTrue(filt.filter(record))
        self.assertTrue(filt.filter(record))
        self.assertEqual(record.name, "run1.openevolve.controller")

    def test_filter_with_empty_prefix_is_noop(self):
        filt = LoggerPrefixFilter("")
        record = _make_record("openevolve.controller")
        self.assertTrue(filt.filter(record))
        self.assertEqual(record.name, "openevolve.controller")

    def test_filter_returns_true_for_all_records(self):
        # The filter must never swallow records; it only renames them.
        filt = LoggerPrefixFilter("runA")
        for name in ("openevolve.controller", "openevolve.evaluator", "some.other.module"):
            self.assertTrue(filt.filter(_make_record(name)))


class TestLoggerPrefixIntegration(unittest.TestCase):
    """
    Integration tests: constructing OpenEvolve calls _setup_logging, which
    attaches handlers to the root logger. Verify the resulting log file (and
    console stream) carry the configured prefix, and that the default (no
    prefix) behavior is unchanged.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="oe290_")
        # _setup_logging mutates the process-wide root logger; save and restore.
        self._root = logging.getLogger()
        self._saved_handlers = self._root.handlers[:]
        self._saved_level = self._root.level

    def tearDown(self):
        current = self._root.handlers[:]
        for handler in current:
            self._root.removeHandler(handler)
        for handler in self._saved_handlers:
            self._root.addHandler(handler)
        self._root.setLevel(self._saved_level)
        for handler in current:
            if handler not in self._saved_handlers:
                handler.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_program_and_evaluator(self):
        program_path = os.path.join(self.tmpdir, "initial_program.py")
        with open(program_path, "w", encoding="utf-8") as f:
            f.write("def run():\n    return 1\n")
        evaluation_path = os.path.join(self.tmpdir, "evaluation.py")
        with open(evaluation_path, "w", encoding="utf-8") as f:
            f.write('def evaluate(program_path):\n    return {"combined_score": 1.0}\n')
        return program_path, evaluation_path

    def _make_config(self, log_dir_name, **overrides):
        config = Config()
        config.log_dir = os.path.join(self.tmpdir, log_dir_name)
        config.random_seed = None
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    def _run_controller(self, config, stderr=None):
        program_path, evaluation_path = self._write_program_and_evaluator()
        output_dir = os.path.join(self.tmpdir, "output")
        with (
            patch("openevolve.controller.LLMEnsemble"),
            patch("openevolve.controller.Evaluator"),
            patch("sys.stderr", stderr or io.StringIO()),
        ):
            OpenEvolve(program_path, evaluation_path, config, output_dir=output_dir)

    def _read_log(self, config):
        log_files = sorted(glob.glob(os.path.join(config.log_dir, "*.log")))
        self.assertTrue(log_files, f"no log files written to {config.log_dir}")
        with open(log_files[0], encoding="utf-8") as f:
            return f.read()

    def test_prefixed_run_writes_prefixed_names_to_log_file(self):
        config = self._make_config("logs_prefixed", logger_prefix="run1")
        self._run_controller(config)
        content = self._read_log(config)
        self.assertIn("run1.openevolve.controller", content)

    def test_prefixed_run_writes_prefixed_names_to_console(self):
        stderr = io.StringIO()
        config = self._make_config("logs_console", logger_prefix="run1")
        self._run_controller(config, stderr=stderr)
        self.assertIn("run1.openevolve.controller", stderr.getvalue())

    def test_no_prefix_keeps_default_names(self):
        config = self._make_config("logs_default")
        self._run_controller(config)
        content = self._read_log(config)
        # Default behavior is unchanged: plain module names, no prefix anywhere.
        self.assertIn("openevolve.controller", content)
        self.assertNotIn("run1.", content)


if __name__ == "__main__":
    unittest.main()
