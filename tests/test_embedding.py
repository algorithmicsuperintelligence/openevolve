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


if __name__ == "__main__":
    unittest.main()
