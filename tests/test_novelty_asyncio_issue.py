"""
Test for issue #313: asyncio.run() error in novelty checking
https://github.com/algorithmicsuperintelligence/openevolve/issues/313

This test reproduces the bug where calling database.add() from within an async context
triggers a novelty check that uses asyncio.run(), which fails because it's already
running in an event loop.
"""

import unittest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase


class MockLLM:
    """Mock LLM that implements the LLMEnsemble async interface"""

    async def generate_with_context(self, system_message: str, messages: list):
        """Mock async generate method that returns NOVEL with model_id"""
        return "NOVEL", 0


class TestNoveltyAsyncioIssue(unittest.TestCase):
    """Test for asyncio.run() error in novelty checking (issue #313)"""

    @patch('openevolve.embedding.EmbeddingClient')
    def setUp(self, mock_embedding_client_class):
        """Set up test database with novelty checking enabled"""
        # Mock the embedding client
        mock_instance = MagicMock()
        mock_instance.get_embedding.return_value = [0.1] * 1536  # Mock embedding vector
        mock_embedding_client_class.return_value = mock_instance

        config = Config()
        config.database.in_memory = True
        config.database.embedding_model = "text-embedding-3-small"
        config.database.similarity_threshold = 0.99
        config.database.novelty_llm = MockLLM()

        self.db = ProgramDatabase(config.database)
        self.mock_embedding_client_class = mock_embedding_client_class

    def test_novelty_check_from_async_context_works(self):
        """
        Test that novelty checking works correctly when called from within
        an async context (this was the bug in issue #313).

        Expected behavior: Should successfully run the novelty check without
        any asyncio.run() errors, properly using ThreadPoolExecutor to handle
        the async LLM call from within a running event loop.
        """
        import logging

        # Create two programs with similar embeddings to trigger LLM novelty check
        program1 = Program(
            id="prog1",
            code="def test(): return 1",
            language="python",
            metrics={"score": 0.5},
        )

        program2 = Program(
            id="prog2",
            code="def test(): return 2",
            language="python",
            metrics={"score": 0.6},
            parent_id="prog1",
        )

        async def async_add_programs():
            """Add programs from async context - this simulates controller.run()"""
            # Add first program (no novelty check, no similar programs yet)
            prog1_id = self.db.add(program1)
            self.assertIsNotNone(prog1_id)

            # Add second program - this triggers novelty check
            # Since embeddings are similar (both [0.1] * 1536), it will call
            # _llm_judge_novelty which should now work correctly
            prog2_id = self.db.add(program2)

            # The novelty check should succeed without errors
            # The program should be added (MockLLM returns "NOVEL")
            self.assertIsNotNone(prog2_id)

            return True

        # This should work without any errors now
        result = asyncio.run(async_add_programs())
        self.assertTrue(result)

        # Verify both programs were added
        self.assertIn("prog1", self.db.programs)
        self.assertIn("prog2", self.db.programs)

    def test_novelty_check_from_sync_context_works(self):
        """
        Test that novelty checking also works correctly when called from
        a synchronous (non-async) context.

        Expected behavior: Should successfully run the novelty check using
        asyncio.run() since there's no running event loop.
        """
        # Create two programs with similar embeddings to trigger LLM novelty check
        program1 = Program(
            id="prog3",
            code="def test(): return 3",
            language="python",
            metrics={"score": 0.5},
        )

        program2 = Program(
            id="prog4",
            code="def test(): return 4",
            language="python",
            metrics={"score": 0.6},
            parent_id="prog3",
        )

        # Add programs from synchronous context (no event loop running)
        prog1_id = self.db.add(program1)
        self.assertIsNotNone(prog1_id)

        prog2_id = self.db.add(program2)
        self.assertIsNotNone(prog2_id)

        # Verify both programs were added
        self.assertIn("prog3", self.db.programs)
        self.assertIn("prog4", self.db.programs)

    def test_novelty_check_disabled_works_fine(self):
        """
        Test that when novelty checking is disabled, adding programs
        from async context works fine (this is the workaround from issue #313).
        """
        # Create a new database with novelty checking disabled
        config = Config()
        config.database.in_memory = True
        config.database.similarity_threshold = 0.0  # Disable novelty checking
        db_no_novelty = ProgramDatabase(config.database)

        program1 = Program(
            id="prog1",
            code="def test(): return 1",
            language="python",
            metrics={"score": 0.5},
        )

        program2 = Program(
            id="prog2",
            code="def test(): return 2",
            language="python",
            metrics={"score": 0.6},
        )

        async def async_add_programs():
            """Add programs from async context"""
            db_no_novelty.add(program1)
            db_no_novelty.add(program2)
            return True

        # This should work fine without novelty checking
        result = asyncio.run(async_add_programs())
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
