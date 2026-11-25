"""A config path that does not exist must not silently run with the defaults."""

import asyncio
import contextlib
import io
import os
import tempfile
import unittest
from unittest.mock import patch

from openevolve import cli
from openevolve.config import Config, load_config


class TestMissingConfigFile(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.missing = os.path.join(temp.name, "confg.yaml")
        self.program = os.path.join(temp.name, "program.py")
        self.evaluator = os.path.join(temp.name, "evaluator.py")
        for path in (self.program, self.evaluator):
            open(path, "w").close()

    def test_load_config_raises_for_missing_file(self):
        with self.assertRaisesRegex(FileNotFoundError, "confg.yaml"):
            load_config(self.missing)

    def test_load_config_without_path_uses_defaults(self):
        config = load_config(None)
        self.assertEqual(config.max_iterations, Config().max_iterations)

    def test_cli_reports_missing_config_and_does_not_run(self):
        argv = ["openevolve-run", self.program, self.evaluator, "--config", self.missing]
        output = io.StringIO()
        with patch("sys.argv", argv), patch.object(cli, "OpenEvolve") as engine_class:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                exit_code = asyncio.run(cli.main_async())
        self.assertEqual(exit_code, 1, output.getvalue())
        self.assertIn(f"Error: Configuration file '{self.missing}' not found", output.getvalue())
        self.assertNotIn("Traceback", output.getvalue())
        engine_class.assert_not_called()


if __name__ == "__main__":
    unittest.main()
