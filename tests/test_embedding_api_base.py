"""
Tests for custom OpenAI-compatible embedding endpoints (#426)
"""

import os
import unittest
from unittest.mock import patch

from openevolve.config import Config, DatabaseConfig
from openevolve.embedding import EmbeddingClient

ENV = {"OPENAI_API_KEY": "llm-key", "OPENAI_EMBEDDING_API_KEY": "embed-key"}


class TestEmbeddingApiBase(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, ENV)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("OPENAI_EMBEDDING_BASE_URL", None)

    @patch("openevolve.embedding.openai.OpenAI")
    def test_api_base_argument_is_used(self, mock_openai):
        client = EmbeddingClient("qwen/qwen3-embedding-8b", api_base="https://openrouter.ai/api/v1")
        mock_openai.assert_called_once_with(
            api_key="embed-key", base_url="https://openrouter.ai/api/v1"
        )
        self.assertEqual(client.model, "qwen/qwen3-embedding-8b")

    @patch("openevolve.embedding.openai.OpenAI")
    def test_env_var_is_used(self, mock_openai):
        with patch.dict(os.environ, {"OPENAI_EMBEDDING_BASE_URL": "http://localhost:8000/v1"}):
            client = EmbeddingClient("local-embedder")
        mock_openai.assert_called_once_with(api_key="embed-key", base_url="http://localhost:8000/v1")
        self.assertEqual(client.model, "local-embedder")

    @patch("openevolve.embedding.openai.OpenAI")
    def test_argument_overrides_env_var(self, mock_openai):
        with patch.dict(os.environ, {"OPENAI_EMBEDDING_BASE_URL": "http://env/v1"}):
            EmbeddingClient("m", api_base="http://arg/v1")
        self.assertEqual(mock_openai.call_args.kwargs["base_url"], "http://arg/v1")

    @patch("openevolve.embedding.openai.OpenAI")
    def test_falls_back_to_openai_api_key(self, mock_openai):
        with patch.dict(os.environ, {}, clear=True):
            os.environ["OPENAI_API_KEY"] = "llm-key"
            EmbeddingClient("m", api_base="http://arg/v1")
        self.assertEqual(mock_openai.call_args.kwargs["api_key"], "llm-key")

    @patch("openevolve.embedding.openai.OpenAI")
    def test_known_openai_model_without_base_url_is_unchanged(self, mock_openai):
        EmbeddingClient("text-embedding-3-small")
        mock_openai.assert_called_once_with(api_key="embed-key")

    def test_unknown_model_without_base_url_still_rejected(self):
        with self.assertRaises(ValueError):
            EmbeddingClient("qwen/qwen3-embedding-8b")

    def test_config_field(self):
        self.assertIsNone(DatabaseConfig().embedding_api_base)
        config = Config.from_dict(
            {"database": {"embedding_model": "m", "embedding_api_base": "http://x/v1"}}
        )
        self.assertEqual(config.database.embedding_api_base, "http://x/v1")

    @patch("openevolve.embedding.EmbeddingClient")
    def test_database_passes_api_base(self, mock_client):
        from openevolve.database import ProgramDatabase

        ProgramDatabase(
            DatabaseConfig(in_memory=True, embedding_model="m", embedding_api_base="http://x/v1")
        )
        mock_client.assert_called_once_with("m", "http://x/v1")


if __name__ == "__main__":
    unittest.main()
