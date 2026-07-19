"""
Tests for the pre-evaluation novelty check in the worker process.

Issue #439: Novelty check runs after evaluation, wasting compute on rejected programs.
This test verifies that the embedding-based similarity check is performed before
the expensive evaluation step.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

# Set dummy API key for testing
os.environ.setdefault("OPENAI_API_KEY", "test")


class TestPreEvalNoveltyCheck(unittest.TestCase):
    """Tests for _pre_eval_novelty_check function in process_parallel.py"""

    def _make_snapshot(self, island_programs, all_programs=None):
        """Helper: build a minimal db_snapshot dict."""
        if all_programs is None:
            all_programs = island_programs

        programs_dict = {}
        for p in all_programs:
            programs_dict[p["id"]] = p

        return {
            "programs": programs_dict,
            "islands": [list(p["id"] for p in island_programs)],
            "current_island": 0,
            "feature_dimensions": [],
            "artifacts": {},
        }

    def _get_check_fn(self):
        """Import the function under test with mocked worker globals."""
        import openevolve.process_parallel as pp

        # Reset globals to a clean state for each test
        pp._worker_config = MagicMock()
        pp._worker_config.database.similarity_threshold = 0.99
        return pp._pre_eval_novelty_check

    def test_returns_none_when_no_embedding_client(self):
        """When _worker_embedding_client is None (novelty disabled), skip check."""
        import openevolve.process_parallel as pp

        fn = self._get_check_fn()
        pp._worker_embedding_client = None
        snapshot = self._make_snapshot([])
        result = fn("def foo(): pass", snapshot, 0)
        self.assertIsNone(result)

    def test_returns_none_when_threshold_zero(self):
        """When threshold <= 0, novelty checking is effectively disabled."""
        import openevolve.process_parallel as pp

        fn = self._get_check_fn()
        mock_client = MagicMock()
        pp._worker_embedding_client = mock_client
        pp._worker_config.database.similarity_threshold = 0.0

        snapshot = self._make_snapshot([])
        result = fn("def foo(): pass", snapshot, 0)
        self.assertIsNone(result)
        mock_client.get_embedding.assert_not_called()

    def test_returns_none_when_island_has_no_embeddings(self):
        """When existing programs have no embeddings, novel by default."""
        import openevolve.process_parallel as pp

        fn = self._get_check_fn()
        mock_client = MagicMock()
        mock_client.get_embedding.return_value = [1.0] + [0.0] * 9
        pp._worker_embedding_client = mock_client
        pp._worker_config.database.similarity_threshold = 0.99

        existing = {"id": "p1", "code": "def bar(): pass", "embedding": None}
        snapshot = self._make_snapshot([existing])
        result = fn("def foo(): pass", snapshot, 0)
        self.assertIsNone(result)

    def test_returns_error_when_code_too_similar(self):
        """When cosine similarity exceeds threshold, return an error string."""
        import openevolve.process_parallel as pp

        fn = self._get_check_fn()
        # Both use the same embedding vector → similarity = 1.0
        shared_embd = [1.0] + [0.0] * 9
        mock_client = MagicMock()
        mock_client.get_embedding.return_value = shared_embd
        pp._worker_embedding_client = mock_client
        pp._worker_config.database.similarity_threshold = 0.99

        existing = {"id": "p1", "code": "def bar(): pass", "embedding": shared_embd}
        snapshot = self._make_snapshot([existing])
        result = fn("def foo(): pass", snapshot, 0)
        self.assertIsNotNone(result)
        self.assertIn("novelty check failed", result)
        self.assertIn("p1", result)

    def test_returns_none_when_code_sufficiently_different(self):
        """When similarity is below threshold, program is novel."""
        import openevolve.process_parallel as pp

        fn = self._get_check_fn()
        child_embd = [1.0] + [0.0] * 9
        existing_embd = [0.0, 1.0] + [0.0] * 8  # orthogonal → similarity = 0
        mock_client = MagicMock()
        mock_client.get_embedding.return_value = child_embd
        pp._worker_embedding_client = mock_client
        pp._worker_config.database.similarity_threshold = 0.99

        existing = {"id": "p1", "code": "def bar(): pass", "embedding": existing_embd}
        snapshot = self._make_snapshot([existing])
        result = fn("def foo(): pass", snapshot, 0)
        self.assertIsNone(result)

    def test_returns_none_on_embedding_error(self):
        """When embedding client raises an exception, skip the check gracefully."""
        import openevolve.process_parallel as pp

        fn = self._get_check_fn()
        mock_client = MagicMock()
        mock_client.get_embedding.side_effect = RuntimeError("API error")
        pp._worker_embedding_client = mock_client
        pp._worker_config.database.similarity_threshold = 0.99

        snapshot = self._make_snapshot([])
        result = fn("def foo(): pass", snapshot, 0)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
