"""Configuration checks for evaluator-defined program identities."""

import unittest

from openevolve.config import Config


class ProgramIdentityConfigTests(unittest.TestCase):
    def test_identity_artifact_round_trips_from_mapping(self):
        config = Config.from_dict(
            {"program_identity_artifact": "program-identity.txt"}
        )

        self.assertEqual(
            config.program_identity_artifact,
            "program-identity.txt",
        )
        self.assertEqual(
            config.to_dict()["program_identity_artifact"],
            "program-identity.txt",
        )

    def test_empty_identity_artifact_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "program_identity_artifact must be a non-empty string",
        ):
            Config.from_dict({"program_identity_artifact": "  "})

    def test_archive_context_configuration_round_trips(self):
        config = Config.from_dict(
            {
                "prompt": {
                    "archive_context_artifact": "program-structure.json",
                    "archive_context_max_items": 48,
                    "artifact_include_names": [
                        "archive-context.json",
                        "guidance.txt",
                    ],
                }
            }
        )

        self.assertEqual(
            config.prompt.archive_context_artifact,
            "program-structure.json",
        )
        self.assertEqual(config.prompt.archive_context_max_items, 48)
        self.assertEqual(
            config.prompt.artifact_include_names,
            ["archive-context.json", "guidance.txt"],
        )

    def test_empty_archive_context_artifact_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "archive_context_artifact must be a non-empty string",
        ):
            Config.from_dict(
                {"prompt": {"archive_context_artifact": "  "}}
            )

    def test_artifact_include_names_require_archive_context(self):
        with self.assertRaisesRegex(
            ValueError,
            "must include archive-context.json",
        ):
            Config.from_dict(
                {
                    "prompt": {
                        "archive_context_artifact": "search-map.txt",
                        "artifact_include_names": ["guidance"],
                    }
                }
            )

    def test_duplicate_artifact_include_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            Config.from_dict(
                {
                    "prompt": {
                        "artifact_include_names": ["guidance", "guidance"],
                    }
                }
            )

    def test_proposal_neighborhood_configuration_round_trips(self):
        config = Config.from_dict(
            {
                "prompt": {
                    "proposal_neighborhood_artifact": "neighborhood.json",
                    "proposal_options_max_items": 17,
                    "artifact_include_names": ["proposal-options.json"],
                }
            }
        )

        self.assertEqual(
            config.prompt.proposal_neighborhood_artifact,
            "neighborhood.json",
        )
        self.assertEqual(config.prompt.proposal_options_max_items, 17)

    def test_proposal_neighborhood_requires_rendered_options(self):
        with self.assertRaisesRegex(
            ValueError,
            "must include proposal-options.json",
        ):
            Config.from_dict(
                {
                    "prompt": {
                        "proposal_neighborhood_artifact": "neighborhood.json",
                        "artifact_include_names": ["guidance"],
                    }
                }
            )

    def test_proposal_option_limit_must_be_positive(self):
        with self.assertRaisesRegex(
            ValueError,
            "proposal_options_max_items must be a positive integer",
        ):
            Config.from_dict(
                {"prompt": {"proposal_options_max_items": 0}}
            )


if __name__ == "__main__":
    unittest.main()
