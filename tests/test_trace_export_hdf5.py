"""Regression tests for HDF5 trace metadata round trips (#574)."""

import tempfile
import unittest
from pathlib import Path

from openevolve.utils.trace_export_utils import export_traces, load_traces

try:
    import h5py
except ImportError:
    h5py = None


@unittest.skipIf(h5py is None, "h5py is not installed")
class TestHdf5MetadataRoundTrip(unittest.TestCase):
    def _round_trip(self, metadata):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.h5"
            export_traces([{"iteration": 1}], path, format="hdf5", metadata=metadata)
            _, loaded = load_traces(path)
        return loaded

    def test_lists_stay_lists(self):
        metadata = {"labels": ["alpha"], "empty": [], "nested": [{"a": [1, 2]}]}
        loaded = self._round_trip(metadata)
        for key, value in metadata.items():
            self.assertEqual(loaded[key], value)

    def test_dicts_and_none_round_trip(self):
        loaded = self._round_trip({"config": {"a": 1}, "nothing": None})
        self.assertEqual(loaded["config"], {"a": 1})
        self.assertIsNone(loaded["nothing"])

    def test_json_looking_strings_stay_strings(self):
        metadata = {"list": ["alpha"], "literal": '["alpha"]', "object": '{"a": 1}', "bad": "[x"}
        loaded = self._round_trip(metadata)
        self.assertEqual(loaded["list"], ["alpha"])
        self.assertEqual(loaded["literal"], '["alpha"]')
        self.assertEqual(loaded["object"], '{"a": 1}')
        self.assertEqual(loaded["bad"], "[x")

    def test_scalars_and_generated_fields_are_unchanged(self):
        loaded = self._round_trip({"name": "run", "count": 3, "rate": 0.5, "flag": True})
        self.assertEqual(loaded["name"], "run")
        self.assertEqual(loaded["count"], 3)
        self.assertEqual(loaded["rate"], 0.5)
        self.assertTrue(loaded["flag"])
        self.assertEqual(loaded["total_traces"], 1)
        self.assertIn("exported_at", loaded)
        self.assertNotIn("__json_encoded_keys__", loaded)

    def test_files_without_a_key_list_load_as_before(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.h5"
            with h5py.File(path, "w") as f:
                f.create_group("traces")
                attrs = f.create_group("metadata").attrs
                attrs["config"] = '{"a": 1}'
                attrs["labels"] = '["alpha"]'
                attrs["broken"] = "{not json"
            _, loaded = load_traces(path)
        self.assertEqual(loaded["config"], {"a": 1})
        self.assertEqual(loaded["labels"], '["alpha"]')
        self.assertEqual(loaded["broken"], "{not json")


if __name__ == "__main__":
    unittest.main()
