import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from openevolve.embedding import EmbeddingClient


class TestEmbeddingClient(unittest.TestCase):
    @patch("openevolve.embedding.openai.OpenAI")
    @patch.dict(
        os.environ,
        {"GEMINI_API_KEY": "gemini-key", "GOOGLE_API_KEY": "google-key"},
        clear=True,
    )
    def test_uses_gemini_key_and_openai_compatible_endpoint(self, mock_openai):
        client = EmbeddingClient("gemini-embedding-001")

        mock_openai.assert_called_once_with(
            api_key="gemini-key",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self.assertEqual(client.model, "gemini-embedding-001")

    @patch("openevolve.embedding.openai.OpenAI")
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "google-key"}, clear=True)
    def test_falls_back_to_google_api_key(self, mock_openai):
        EmbeddingClient("gemini-embedding-001")

        self.assertEqual(mock_openai.call_args.kwargs["api_key"], "google-key")

    def test_rejects_unknown_embedding_model(self):
        with self.assertRaisesRegex(ValueError, "Invalid embedding model: unknown-model"):
            EmbeddingClient("unknown-model")

    def test_get_embedding_returns_a_single_embedding(self):
        client = EmbeddingClient.__new__(EmbeddingClient)
        client.model = "gemini-embedding-001"
        client.client = MagicMock()
        client.client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2])]
        )

        result = client.get_embedding("def example(): pass")

        self.assertEqual(result, [0.1, 0.2])
        client.client.embeddings.create.assert_called_once_with(
            model="gemini-embedding-001",
            input=["def example(): pass"],
            encoding_format="float",
        )

    def test_get_embedding_returns_batch_embeddings(self):
        client = EmbeddingClient.__new__(EmbeddingClient)
        client.model = "gemini-embedding-001"
        client.client = MagicMock()
        client.client.embeddings.create.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[0.1, 0.2]),
                SimpleNamespace(embedding=[0.3, 0.4]),
            ]
        )

        result = client.get_embedding(["first", "second"])

        self.assertEqual(result, [[0.1, 0.2], [0.3, 0.4]])
        client.client.embeddings.create.assert_called_once_with(
            model="gemini-embedding-001",
            input=["first", "second"],
            encoding_format="float",
        )

    def test_single_request_failure_returns_empty_vector_and_can_recover(self):
        client = EmbeddingClient.__new__(EmbeddingClient)
        client.model = "text-embedding-3-small"
        client.client = MagicMock()
        client.client.embeddings.create.side_effect = [
            RuntimeError("embedding unavailable"),
            SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])]),
        ]

        with self.assertLogs("openevolve.embedding", level="INFO") as logs:
            result = client.get_embedding("first")
        self.assertEqual(result, [])
        self.assertFalse(result)
        self.assertIn("embedding unavailable", logs.output[0])
        self.assertEqual(client.get_embedding("second"), [0.1, 0.2])

    def test_batch_request_failure_preserves_input_cardinality(self):
        for batch in ([], ["first"], ["first", "second", "third"]):
            with self.subTest(batch=batch):
                client = EmbeddingClient.__new__(EmbeddingClient)
                client.model = "text-embedding-3-small"
                client.client = MagicMock()
                client.client.embeddings.create.side_effect = RuntimeError("embedding unavailable")
                result = client.get_embedding(batch)
                self.assertEqual(result, [[] for _ in batch])
                client.client.embeddings.create.assert_called_once_with(
                    model=client.model, input=batch, encoding_format="float"
                )

    def test_failed_embedding_does_not_break_database_novelty_checks(self):
        from openevolve.config import DatabaseConfig
        from openevolve.database import Program, ProgramDatabase

        for dimensions, failures in ((2, 1), (3, 2)):
            with self.subTest(dimensions=dimensions, failures=failures):
                seed = [1.0] + [0.0] * (dimensions - 1)
                recovered = [0.0, 1.0] + [0.0] * (dimensions - 2)
                client = EmbeddingClient.__new__(EmbeddingClient)
                client.model = "text-embedding-3-small"
                client.client = MagicMock()
                client.client.embeddings.create.side_effect = [
                    SimpleNamespace(data=[SimpleNamespace(embedding=seed)]),
                    *[RuntimeError("embedding unavailable") for _ in range(failures)],
                    SimpleNamespace(data=[SimpleNamespace(embedding=recovered)]),
                ]
                config = DatabaseConfig(
                    in_memory=True,
                    num_islands=1,
                    embedding_model=client.model,
                    similarity_threshold=0.9,
                )
                with patch("openevolve.embedding.EmbeddingClient", return_value=client):
                    database = ProgramDatabase(config)
                programs = [
                    Program(id=str(i), code=f"x = {i}", metrics={"combined_score": i})
                    for i in range(failures + 2)
                ]
                for program in programs:
                    self.assertEqual(database.add(program), program.id)
                for program in programs[1:-1]:
                    self.assertEqual(program.embedding, [])
                self.assertEqual(programs[-1].embedding, recovered)
                self.assertIn(programs[-1].id, database.islands[0])


if __name__ == "__main__":
    unittest.main()
