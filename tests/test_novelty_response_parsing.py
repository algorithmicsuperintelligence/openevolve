"""Tests for parsing responses from the LLM novelty judge."""

import unittest

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase


class MockNoveltyLLM:
    def __init__(self, response: str):
        self.response = response

    async def generate_with_context(self, system_message: str, messages: list) -> str:
        return self.response


class TestNoveltyResponseParsing(unittest.TestCase):
    def setUp(self) -> None:
        config = Config()
        config.database.in_memory = True
        config.database.novelty_llm = MockNoveltyLLM("NOVEL")
        self.db = ProgramDatabase(config.database)
        self.existing_program = Program(id="existing", code="return 1")
        self.proposed_program = Program(id="proposed", code="return 2")

    def judge(self, response: str) -> bool:
        self.db.novelty_llm = MockNoveltyLLM(response)
        return self.db._llm_judge_novelty(self.proposed_program, self.existing_program)

    def test_prompted_not_novel_responses_are_rejected(self) -> None:
        responses = [
            "NOT_NOVEL",
            "NOT_NOVEL: the programs are equivalent",
            "**NOT_NOVEL**",
            "NOT NOVEL",
        ]

        for response in responses:
            with self.subTest(response=response):
                self.assertFalse(self.judge(response))

    def test_novel_response_is_accepted(self) -> None:
        self.assertTrue(self.judge("NOVEL: the programs behave differently"))


if __name__ == "__main__":
    unittest.main()
