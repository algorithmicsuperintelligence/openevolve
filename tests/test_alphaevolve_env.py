"""Explicit, offline imports of AlphaEvolve example .env files."""

import builtins
import copy
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from openevolve.compat.alphaevolve import config_from_alphaevolve
from openevolve.compat.alphaevolve_env import config_from_alphaevolve_env


class TestAlphaEvolveEnv(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "example.env"

    def convert(self, text, **kwargs):
        self.path.write_text(text, encoding="utf-8")
        return config_from_alphaevolve_env(
            self.path,
            problem_description="Pack circles into a unit square.",
            api_base="http://localhost:8080/v1",
            **kwargs,
        )

    def test_single_model_matches_dictionary_import(self):
        actual = self.convert("MODEL=local\nMAX_PROGRAMS_GENERATED=3\n", api_key="local-key")
        expected = config_from_alphaevolve(
            {
                "program_language": "python",
                "problem_description": "Pack circles into a unit square.",
                "run_settings": {"max_programs": 3},
                "generation_settings": {"models": [{"name": "local", "weight": 1.0}]},
            },
            api_base="http://localhost:8080/v1",
            api_key="local-key",
        )
        self.assertEqual(actual.to_dict(), expected.to_dict())

    def test_numbered_models_are_sorted_numerically_with_relative_weights(self):
        config = self.convert(
            "MODEL_10=large\nMODEL_10_WEIGHT=0.3\nMODEL_2=fast\nMODEL_2_WEIGHT=0.7\n"
            "MAX_PROGRAMS_GENERATED=10\n"
        )
        self.assertEqual(
            [(m.name, m.weight) for m in config.llm.models], [("fast", 0.7), ("large", 0.3)]
        )

    def test_missing_weights_default_to_one_and_single_weight_is_supported(self):
        for text, weights in (
            ("MODEL=local\n", [1.0]),
            ("MODEL=local\nMODEL_WEIGHT=0.5\n", [0.5]),
            ("MODEL_1=a\nMODEL_2=b\n", [1.0, 1.0]),
        ):
            with self.subTest(text=text):
                config = self.convert(text + "MAX_PROGRAMS_GENERATED=1\n")
                self.assertEqual([m.weight for m in config.llm.models], weights)

    def test_dotenv_quotes_comments_export_bom_and_crlf(self):
        config = self.convert(
            '\ufeff# example\r\nexport MODEL = "local#tag" # inline\r\n'
            "MAX_PROGRAMS_GENERATED = '2'\r\n"
        )
        self.assertEqual(config.llm.models[0].name, "local#tag")
        self.assertEqual(config.max_iterations, 2)

    def test_missing_empty_or_mixed_models_and_orphan_weights_raise(self):
        for text in (
            "",
            "MODEL=\n",
            "MODEL\n",
            "MODEL='   '\n",
            "MODEL_WEIGHT=1\n",
            "MODEL_1_WEIGHT=1\n",
            "MODEL_1=a\nMODEL_2=\n",
            "MODEL_0=a\n",
            "MODEL_01=a\n",
            "MODEL=a\nMODEL_1=b\n",
        ):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "MODEL"):
                self.convert(text + "MAX_PROGRAMS_GENERATED=1\n")

    def test_invalid_weights_are_rejected_without_echoing_values(self):
        for weight in ("", "secret-invalid-weight", "0", "-1", "nan", "inf", "1e999"):
            with self.subTest(weight=weight), self.assertRaises(ValueError) as caught:
                self.convert(f"MODEL=local\nMODEL_WEIGHT={weight}\nMAX_PROGRAMS_GENERATED=1\n")
            self.assertNotIn("secret-invalid-weight", str(caught.exception))
        with self.assertRaisesRegex(ValueError, "finite sum"):
            self.convert(
                "MODEL_1=a\nMODEL_2=b\nMODEL_1_WEIGHT=1e308\nMODEL_2_WEIGHT=1e308\nMAX_PROGRAMS_GENERATED=1\n"
            )

    def test_generation_budget_is_required_and_strictly_positive(self):
        for budget in (None, "", "0", "-1", "1.5", "True", "${BUDGET}"):
            text = "MODEL=local\n"
            if budget is not None:
                text += f"MAX_PROGRAMS_GENERATED={budget}\n"
            with (
                self.subTest(budget=budget),
                self.assertRaisesRegex(ValueError, "MAX_PROGRAMS_GENERATED"),
            ):
                self.convert(text)
        with self.assertRaisesRegex(ValueError, "MAX_PROGRAMS_GENERATED"):
            self.convert("MODEL=local\nMAX_PROGRAMS_EVALUATED=3\n")

    def test_equal_evaluation_budget_warns_that_it_is_not_a_hard_cap(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = self.convert(
                "MODEL=local\nMAX_PROGRAMS_GENERATED=2\nMAX_PROGRAMS_EVALUATED=2\n"
            )
        self.assertEqual(config.max_iterations, 2)
        self.assertEqual(len(caught), 1)
        self.assertIn("MAX_PROGRAMS_EVALUATED", str(caught[0].message))
        self.assertIn("seed", str(caught[0].message))

    def test_conflicting_or_invalid_evaluation_budget_is_not_silently_dropped(self):
        for budget in ("1", "3", "", "-1", "no", "1.5"):
            with (
                self.subTest(budget=budget),
                self.assertRaisesRegex(ValueError, "MAX_PROGRAMS_EVALUATED"),
            ):
                self.convert(
                    f"MODEL=local\nMAX_PROGRAMS_GENERATED=2\nMAX_PROGRAMS_EVALUATED={budget}\n"
                )

    def test_environment_is_neither_read_for_expansion_nor_modified(self):
        with patch.dict(
            os.environ,
            {"MODEL": "ambient-model", "MAX_PROGRAMS_GENERATED": "999", "PRIVATE": "do-not-copy"},
        ):
            before = copy.deepcopy(dict(os.environ))
            config = self.convert("MODEL=file-model\nMAX_PROGRAMS_GENERATED=2\n")
            self.assertEqual(config.llm.models[0].name, "file-model")
            self.assertEqual(config.max_iterations, 2)
            with self.assertRaisesRegex(ValueError, "MODEL"):
                self.convert("MAX_PROGRAMS_GENERATED=2\n")
            for value in ("${PRIVATE}", "$PRIVATE", "$(touch marker)", "`touch marker`"):
                with (
                    self.subTest(value=value),
                    self.assertRaisesRegex(ValueError, "MODEL") as caught,
                ):
                    self.convert(f"MODEL='{value}'\nMAX_PROGRAMS_GENERATED=1\n")
                self.assertNotIn("do-not-copy", str(caught.exception))
            self.assertEqual(dict(os.environ), before)
            self.assertFalse((Path(self.directory.name) / "marker").exists())

    def test_cloud_credentials_endpoints_and_concurrency_are_not_imported(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = self.convert(
                "MODEL=local\nMAX_PROGRAMS_GENERATED=2\nPROJECT_ID=private-project\n"
                "BASE_URL=https://cloud.invalid\nOPENAI_API_KEY=secret-from-file\n"
                "CONCURRENCY=4\nWORKER_CONCURRENCY=8\nPARALLEL_EVALUATION=True\n",
                api_key="caller-key",
            )
        messages = "\n".join(str(w.message) for w in caught)
        for name in (
            "PROJECT_ID",
            "BASE_URL",
            "OPENAI_API_KEY",
            "CONCURRENCY",
            "WORKER_CONCURRENCY",
            "PARALLEL_EVALUATION",
        ):
            self.assertIn(name, messages)
        for value in ("private-project", "cloud.invalid", "secret-from-file"):
            self.assertNotIn(value, messages)
        self.assertEqual(config.evaluator.parallel_evaluations, 1)
        for model in config.llm.models + config.llm.evaluator_models:
            self.assertEqual(model.api_key, "caller-key")
            self.assertEqual(model.api_base, "http://localhost:8080/v1")

    def test_missing_file_does_not_search_parent_or_default_env(self):
        (Path(self.directory.name) / ".env").write_text(
            "MODEL=ambient\nMAX_PROGRAMS_GENERATED=100\n"
        )
        with self.assertRaises(FileNotFoundError):
            config_from_alphaevolve_env(
                self.path, problem_description="Problem", api_base="http://localhost/v1"
            )

    def test_duplicate_assignments_and_malformed_syntax_fail_without_values(self):
        for text in (
            "MODEL=local\nMODEL=secret-duplicate\nMAX_PROGRAMS_GENERATED=1\n",
            'MODEL="secret-unclosed\nMAX_PROGRAMS_GENERATED=1\n',
            "MODEL=local\nMAX_PROGRAMS_GENERATED=1\ninvalid syntax secret-malformed\n",
        ):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "line") as caught:
                self.convert(text)
            self.assertNotIn("secret-", str(caught.exception))

    def test_no_model_client_or_network_is_created(self):
        with (
            patch("socket.socket.connect", side_effect=AssertionError("network")),
            patch("openevolve.llm.openai.openai.AsyncOpenAI", side_effect=AssertionError("client")),
        ):
            self.assertEqual(
                self.convert("MODEL=local\nMAX_PROGRAMS_GENERATED=1\n").max_iterations, 1
            )

    def test_optional_dependency_error_is_actionable(self):
        original_import = builtins.__import__

        def without_dotenv(name, *args, **kwargs):
            if name == "dotenv.parser":
                raise ImportError("not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=without_dotenv):
            with self.assertRaisesRegex(ImportError, "openevolve\\[alphaevolve-env\\]"):
                self.convert("MODEL=local\nMAX_PROGRAMS_GENERATED=1\n")


if __name__ == "__main__":
    unittest.main()
