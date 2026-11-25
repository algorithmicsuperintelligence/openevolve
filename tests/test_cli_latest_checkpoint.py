"""The end-of-run summary must not fail a finished run over the checkpoints folder."""

import asyncio
import contextlib
import io
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from openevolve import cli


class TestCliLatestCheckpoint(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        # The default output folder name, "openevolve_output", has an underscore
        self.output_dir = os.path.join(temp.name, "openevolve_output")
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        for name in ("checkpoint_9", "checkpoint_10"):
            os.makedirs(os.path.join(self.checkpoint_dir, name))
        self.program = os.path.join(temp.name, "program.py")
        self.evaluator = os.path.join(temp.name, "evaluator.py")
        for path in (self.program, self.evaluator):
            open(path, "w").close()

    def _run_cli(self):
        engine = MagicMock()
        engine.output_dir = self.output_dir
        engine.run = AsyncMock(return_value=MagicMock(metrics={"combined_score": 1.0}))
        argv = ["openevolve-run", self.program, self.evaluator]
        output = io.StringIO()
        with patch("sys.argv", argv), patch.object(cli, "OpenEvolve", return_value=engine):
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                exit_code = asyncio.run(cli.main_async())
        return exit_code, output.getvalue()

    def test_reports_highest_numbered_checkpoint(self):
        exit_code, output = self._run_cli()
        self.assertEqual(exit_code, 0, output)
        self.assertIn(f"--checkpoint {os.path.join(self.checkpoint_dir, 'checkpoint_10')}", output)

    def test_ignores_other_directories(self):
        # e.g. a checkpoint the user kept under another name, or a notes folder
        for name in ("checkpoint_10_backup", "notes"):
            os.makedirs(os.path.join(self.checkpoint_dir, name))
        exit_code, output = self._run_cli()
        self.assertEqual(exit_code, 0, output)
        self.assertIn(f"--checkpoint {os.path.join(self.checkpoint_dir, 'checkpoint_10')}", output)

    def test_no_numbered_checkpoints(self):
        for name in ("checkpoint_9", "checkpoint_10"):
            os.rmdir(os.path.join(self.checkpoint_dir, name))
        os.makedirs(os.path.join(self.checkpoint_dir, "notes"))
        exit_code, output = self._run_cli()
        self.assertEqual(exit_code, 0, output)
        self.assertNotIn("--checkpoint", output)


if __name__ == "__main__":
    unittest.main()
