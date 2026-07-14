"""
Tests that inspiration programs are not duplicated with the top/diverse programs
already shown in the prompt (GitHub issue #452, point 3).
"""

import unittest

from openevolve.config import Config
from openevolve.prompt.sampler import PromptSampler


class TestInspirationDedup(unittest.TestCase):
    def setUp(self):
        config = Config()
        config.prompt.num_top_programs = 2
        config.prompt.num_diverse_programs = 0  # keep the "diverse" sub-section out of the way
        config.prompt.include_artifacts = False
        self.sampler = PromptSampler(config.prompt)

    def _prog(self, pid, marker, score):
        # Unique code marker per program so we can count its appearances in the prompt
        return {"id": pid, "code": f"def {marker}(): return {score}", "metrics": {"score": score}}

    def test_overlapping_inspiration_not_shown_twice(self):
        top = [
            self._prog("A", "prog_A", 0.9),
            self._prog("B", "prog_B", 0.8),
        ]
        # Inspirations include A (already a top program) and a unique program C.
        inspirations = [
            self._prog("A", "prog_A", 0.9),
            self._prog("C", "prog_C", 0.3),
        ]

        user = self.sampler.build_prompt(
            current_program="def cur(): pass",
            parent_program="def cur(): pass",
            program_metrics={"score": 0.5},
            top_programs=top,
            inspirations=inspirations,
        )["user"]

        # A appears only once (in the top section), not re-listed as an inspiration.
        self.assertEqual(user.count("def prog_A()"), 1, "Overlapping program must not be duplicated")
        # The unique inspiration C is still present.
        self.assertIn("def prog_C()", user)
        # B (top) present as usual.
        self.assertIn("def prog_B()", user)

    def test_non_overlapping_inspirations_all_shown(self):
        top = [self._prog("A", "prog_A", 0.9)]
        inspirations = [self._prog("C", "prog_C", 0.3), self._prog("D", "prog_D", 0.2)]

        user = self.sampler.build_prompt(
            current_program="def cur(): pass",
            parent_program="def cur(): pass",
            program_metrics={"score": 0.5},
            top_programs=top,
            inspirations=inspirations,
        )["user"]

        self.assertIn("def prog_C()", user)
        self.assertIn("def prog_D()", user)

    def test_inspirations_without_ids_are_kept(self):
        # Programs lacking an "id" cannot be deduped and must be preserved.
        top = [self._prog("A", "prog_A", 0.9)]
        inspirations = [{"code": "def prog_noid(): pass", "metrics": {"score": 0.1}}]

        user = self.sampler.build_prompt(
            current_program="def cur(): pass",
            parent_program="def cur(): pass",
            program_metrics={"score": 0.5},
            top_programs=top,
            inspirations=inspirations,
        )["user"]

        self.assertIn("def prog_noid()", user)


if __name__ == "__main__":
    unittest.main()
