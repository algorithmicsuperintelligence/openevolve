"""Tests that concurrent OpenEvolve instances do not attach to the root logger."""

import logging
import os
import tempfile
import unittest
from unittest.mock import patch

from openevolve.config import Config
from openevolve.controller import OpenEvolve, _RunIdFilter, _current_run_id


def _write_minimal_files(directory: str):
    program = os.path.join(directory, "program.py")
    eval_file = os.path.join(directory, "evaluator.py")
    with open(program, "w", encoding="utf-8") as handle:
        handle.write("def solve():\n    return 1\n")
    with open(eval_file, "w", encoding="utf-8") as handle:
        handle.write("def evaluate(program_path):\n    return {'score': 1.0}\n")
    return program, eval_file


def _minimal_config():
    config = Config()
    config.database.in_memory = True
    config.database.db_path = None
    return config


class TestLoggingIsolation(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.test_dir = self._tmpdir.name
        self._llm_patch = patch("openevolve.controller.LLMEnsemble")
        self._llm_patch.start()

    def tearDown(self):
        self._llm_patch.stop()
        self._tmpdir.cleanup()
        logging.getLogger("openevolve").handlers.clear()
        _current_run_id.set(None)

    def test_does_not_attach_handlers_to_root_logger(self):
        root_before = list(logging.getLogger().handlers)
        program, eval_file = _write_minimal_files(self.test_dir)
        controller = OpenEvolve(
            initial_program_path=program,
            evaluation_file=eval_file,
            config=_minimal_config(),
            output_dir=os.path.join(self.test_dir, "out"),
        )
        try:
            root_after = list(logging.getLogger().handlers)
            self.assertEqual(root_before, root_after)
            package = logging.getLogger("openevolve")
            self.assertFalse(package.propagate)
            self.assertGreaterEqual(len(controller._log_handlers), 2)
            for handler in controller._log_handlers:
                self.assertIn(handler, package.handlers)
        finally:
            controller._teardown_logging()

    def test_parallel_instances_use_distinct_handlers_and_filters(self):
        left_dir = os.path.join(self.test_dir, "left")
        right_dir = os.path.join(self.test_dir, "right")
        os.makedirs(left_dir)
        os.makedirs(right_dir)
        left_prog, left_eval = _write_minimal_files(left_dir)
        right_prog, right_eval = _write_minimal_files(right_dir)

        left = OpenEvolve(
            initial_program_path=left_prog,
            evaluation_file=left_eval,
            config=_minimal_config(),
            output_dir=os.path.join(left_dir, "out"),
        )
        if left._log_token is not None:
            _current_run_id.reset(left._log_token)
            left._log_token = None

        right = OpenEvolve(
            initial_program_path=right_prog,
            evaluation_file=right_eval,
            config=_minimal_config(),
            output_dir=os.path.join(right_dir, "out"),
        )
        try:
            self.assertNotEqual(left._run_id, right._run_id)
            self.assertFalse(set(left._log_handlers) & set(right._log_handlers))
            left_ids = {
                f.run_id
                for h in left._log_handlers
                for f in h.filters
                if isinstance(f, _RunIdFilter)
            }
            right_ids = {
                f.run_id
                for h in right._log_handlers
                for f in h.filters
                if isinstance(f, _RunIdFilter)
            }
            self.assertEqual(left_ids, {left._run_id})
            self.assertEqual(right_ids, {right._run_id})
        finally:
            left._teardown_logging()
            right._teardown_logging()

    def test_run_id_filter_matches_active_context(self):
        filt = _RunIdFilter("aaa")
        record = logging.LogRecord("n", logging.INFO, __file__, 1, "msg", (), None)
        token = _current_run_id.set("aaa")
        try:
            self.assertTrue(filt.filter(record))
        finally:
            _current_run_id.reset(token)
        token = _current_run_id.set("bbb")
        try:
            self.assertFalse(filt.filter(record))
        finally:
            _current_run_id.reset(token)


if __name__ == "__main__":
    unittest.main()
