"""
Tests that the visualizer does not enable Flask debug mode by default (#481)
"""

import os
import runpy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import flask
except ImportError:  # pragma: no cover - visualizer extras not installed
    flask = None

VISUALIZER = Path(__file__).resolve().parent.parent / "scripts" / "visualizer.py"


@unittest.skipIf(flask is None, "flask is not installed")
class TestVisualizerDebugFlag(unittest.TestCase):
    def _run(self, *args):
        # The script imports its sibling modules and sets EVOLVE_OUTPUT
        with patch.object(flask.Flask, "run") as mock_run, patch.object(
            sys, "argv", ["visualizer.py", "--path", ".", *args]
        ), patch.object(sys, "path", [str(VISUALIZER.parent), *sys.path]), patch.dict(
            os.environ
        ):
            runpy.run_path(str(VISUALIZER), run_name="__main__")
        mock_run.assert_called_once()
        return mock_run.call_args.kwargs

    def test_debug_off_by_default(self):
        kwargs = self._run()
        self.assertFalse(kwargs["debug"])
        self.assertEqual(kwargs["host"], "127.0.0.1")

    def test_debug_flag_enables_debug(self):
        self.assertTrue(self._run("--debug")["debug"])


if __name__ == "__main__":
    unittest.main()
