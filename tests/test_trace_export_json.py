"""Regression tests for JSON trace export metadata ownership."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openevolve.utils.trace_export_utils import export_traces_json


class TestJsonExportMetadata(unittest.TestCase):
    def test_reusing_metadata_exports_fresh_statistics(self):
        metadata = {"run": "example"}
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.json"
            second_path = Path(directory) / "second.json"
            with patch("openevolve.utils.trace_export_utils.time") as clock:
                clock.time.side_effect = [1000.0, 2000.0]
                export_traces_json([{"id": "one"}], first_path, metadata)
                export_traces_json([{"id": "one"}, {"id": "two"}], second_path, metadata)

            first = json.loads(first_path.read_text())
            second = json.loads(second_path.read_text())

        self.assertEqual(first["metadata"]["total_traces"], len(first["traces"]))
        self.assertEqual(first["metadata"]["exported_at"], 1000.0)
        self.assertEqual(second["metadata"]["total_traces"], len(second["traces"]))
        self.assertEqual(second["metadata"]["exported_at"], 2000.0)
        self.assertEqual(metadata, {"run": "example"})

    def test_explicit_statistics_remain_overridable(self):
        metadata = {"run": "example", "total_traces": 42, "exported_at": 123.0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces.json"
            export_traces_json([{"id": "one"}], path, metadata)
            exported = json.loads(path.read_text())

        self.assertEqual(exported["metadata"], metadata)
        self.assertEqual(exported["traces"], [{"id": "one"}])

    def test_missing_and_empty_metadata_get_default_statistics(self):
        for metadata in (None, {}):
            with self.subTest(metadata=metadata), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "traces.json"
                with patch("openevolve.utils.trace_export_utils.time") as clock:
                    clock.time.return_value = 1000.0
                    export_traces_json([], path, metadata)
                exported = json.loads(path.read_text())

            self.assertEqual(exported["metadata"], {"total_traces": 0, "exported_at": 1000.0})
            self.assertEqual(exported["traces"], [])


if __name__ == "__main__":
    unittest.main()
