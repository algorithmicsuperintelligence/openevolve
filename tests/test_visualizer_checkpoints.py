"""Checkpoint discovery and visualizer loading regression tests for #481."""

import json
import os
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import flask
except ImportError:  # pragma: no cover - visualizer extras not installed
    flask = None

VISUALIZER = Path(__file__).resolve().parent.parent / "scripts" / "visualizer.py"


@unittest.skipIf(flask is None, "flask is not installed")
class TestVisualizerCheckpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(sys, "path", [str(VISUALIZER.parent), *sys.path]):
            cls.visualizer = runpy.run_path(str(VISUALIZER))

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkpoint = self._checkpoint("checkpoint_1")

    def _checkpoint(self, name):
        checkpoint = self.root / name
        programs = checkpoint / "programs"
        programs.mkdir(parents=True)
        (checkpoint / "metadata.json").write_text(
            json.dumps({"islands": [["seed"]], "archive": ["seed"]}), encoding="utf-8"
        )
        (programs / "seed.json").write_text(
            json.dumps(
                {
                    "id": "seed",
                    "code": "def run_search():\n    return 1\n",
                    "metrics": {"combined_score": 1.0},
                    "metadata": {},
                }
            ),
            encoding="utf-8",
        )
        return checkpoint

    def test_direct_checkpoint_with_trailing_separators(self):
        for suffix in ("", os.sep, os.sep * 3):
            with self.subTest(suffix=suffix):
                found = self.visualizer["find_latest_checkpoint"](str(self.checkpoint) + suffix)
                self.assertEqual(found, str(self.checkpoint))

    def test_relative_checkpoint_with_trailing_separator(self):
        relative = os.path.relpath(self.checkpoint)
        found = self.visualizer["find_latest_checkpoint"](relative + os.sep)
        self.assertEqual(found, relative)

    def test_parent_directory_still_selects_latest_checkpoint(self):
        newer = self._checkpoint("nested/checkpoint_2")
        os.utime(self.checkpoint, (100, 100))
        os.utime(newer, (200, 200))
        for suffix in ("", os.sep):
            with self.subTest(suffix=suffix):
                found = self.visualizer["find_latest_checkpoint"](str(self.root) + suffix)
                self.assertEqual(found, str(newer))

    def test_directory_without_checkpoints_returns_none(self):
        empty = self.root / "empty"
        empty.mkdir()
        self.assertIsNone(self.visualizer["find_latest_checkpoint"](str(empty) + os.sep))

    def test_data_endpoint_loads_checkpoint_with_trailing_separator(self):
        app = self.visualizer["app"]
        with app.test_client() as client:
            for suffix in ("", os.sep):
                with (
                    self.subTest(suffix=suffix),
                    patch.dict(os.environ, {"EVOLVE_OUTPUT": str(self.checkpoint) + suffix}),
                ):
                    response = client.get("/api/data")
                    self.assertEqual(response.status_code, 200)
                    data = response.get_json()
                    self.assertEqual(data["checkpoint_dir"], str(self.checkpoint))
                    self.assertEqual(data["archive"], ["seed"])
                    self.assertEqual([node["id"] for node in data["nodes"]], ["seed"])
                    self.assertEqual(data["nodes"][0]["metrics"], {"combined_score": 1.0})

    def test_static_export_loads_checkpoint_with_trailing_separator(self):
        output = self.root / "static-export"
        self.visualizer["run_static_export"](
            SimpleNamespace(path=str(self.checkpoint) + os.sep, static_output=str(output))
        )
        html = (output / "index.html").read_text(encoding="utf-8")
        payload = html.split("<script>window.STATIC_DATA = ", 1)[1].split(";</script>", 1)[0]
        data = json.loads(payload)
        self.assertEqual(data["checkpoint_dir"], str(self.checkpoint))
        self.assertEqual(data["archive"], ["seed"])
        self.assertEqual([node["id"] for node in data["nodes"]], ["seed"])
        self.assertTrue((output / "static").is_dir())


if __name__ == "__main__":
    unittest.main()
