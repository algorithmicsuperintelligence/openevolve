"""Tests that programs rejected by the novelty check are not kept in the database."""

import tempfile
import unittest
from unittest.mock import MagicMock, patch

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase


class NotNovelLLM:
    async def generate_with_context(self, system_message: str, messages: list) -> str:
        return "NOT_NOVEL"


class TestNoveltyRejection(unittest.TestCase):
    @patch("openevolve.embedding.EmbeddingClient")
    def setUp(self, mock_embedding_client_class) -> None:
        # Identical embeddings so every candidate is sent to the (rejecting) LLM judge
        mock_embedding_client_class.return_value = MagicMock(
            get_embedding=MagicMock(return_value=[1.0, 0.0])
        )
        config = Config()
        config.database.in_memory = True
        config.database.db_path = None
        config.database.num_islands = 1
        config.database.population_size = 3
        config.database.embedding_model = "text-embedding-3-small"
        config.database.similarity_threshold = 0.5
        config.database.novelty_llm = NotNovelLLM()
        self.db = ProgramDatabase(config.database)

        self.seed = Program(id="seed", code="x = 0", metrics={"combined_score": 0.1})
        self.assertEqual(self.db.add(self.seed), "seed")

    def add_rejected(self, n: int) -> list:
        return [
            self.db.add(
                Program(
                    id=f"dup{i}",
                    code=f"x = {i + 1}",
                    parent_id="seed",
                    metrics={"combined_score": 0.9},
                )
            )
            for i in range(n)
        ]

    def test_rejected_program_is_not_stored(self) -> None:
        results = self.add_rejected(5)

        self.assertEqual(results, [None] * 5)
        self.assertEqual(set(self.db.programs), {"seed"})
        self.assertEqual(self.db.islands, [{"seed"}])
        self.assertLessEqual(len(self.db.programs), self.db.config.population_size)

    def test_rejected_program_is_never_sampled_or_ranked(self) -> None:
        self.add_rejected(5)

        for _ in range(20):
            self.assertEqual(self.db._sample_random_parent().id, "seed")
        self.assertEqual([p.id for p in self.db.get_top_programs(n=10)], ["seed"])

    def test_rejected_program_is_not_checkpointed(self) -> None:
        self.add_rejected(2)

        with tempfile.TemporaryDirectory() as tmp:
            self.db.save(tmp)
            config = Config()
            config.database.in_memory = True
            config.database.db_path = None
            config.database.num_islands = 1
            restored = ProgramDatabase(config.database)
            restored.load(tmp)

        self.assertEqual(set(restored.programs), {"seed"})

    def test_rejection_does_not_touch_feature_stats(self) -> None:
        stats_before = {
            k: dict(v, values=list(v["values"])) for k, v in self.db.feature_stats.items()
        }

        self.db.add(Program(id="long", code="y = 1\n" * 500, metrics={"combined_score": 0.9}))

        self.assertEqual(self.db.feature_stats, stats_before)


if __name__ == "__main__":
    unittest.main()
