"""Offline tests for the explicit OpenEvolve-to-AlphaEvolve config subset."""

import copy
import json
import secrets
import unittest
import warnings
from unittest.mock import patch

from openevolve.compat.alphaevolve import config_from_alphaevolve
from openevolve.compat.alphaevolve_export import config_to_alphaevolve
from openevolve.config import Config, LLMModelConfig


def config():
    return Config.from_dict(
        {
            "language": "python",
            "max_iterations": 12,
            "llm": {
                "models": [
                    {"name": "gemini-2.5-flash", "weight": 7},
                    {"name": "gemini-3.1-pro-preview", "weight": 3},
                ]
            },
            "prompt": {"system_message": "Pack circles into a unit square. 中文。"},
        }
    )


class TestAlphaEvolveConfigExport(unittest.TestCase):
    def convert(self, supplied=None, **options):
        if supplied is None:
            supplied = config()
        settings = {
            "title": "Circle Packing",
            "problem_description": supplied.prompt.system_message,
            **options,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return config_to_alphaevolve(supplied, **settings)

    def test_supported_subset_is_json_serializable_and_does_not_mutate_config(self):
        supplied = config()
        original = copy.deepcopy(supplied)
        result = self.convert(supplied)
        self.assertEqual(
            result,
            {
                "title": "Circle Packing",
                "problem_description": supplied.prompt.system_message,
                "program_language": "python",
                "run_settings": {"max_programs": 12},
                "generation_settings": {
                    "models": [
                        {"name": "gemini-2.5-flash", "weight": 7.0},
                        {"name": "gemini-3.1-pro-preview", "weight": 3.0},
                    ]
                },
            },
        )
        self.assertEqual(supplied, original)
        json.dumps(result, ensure_ascii=False, allow_nan=False)

    def test_non_gemini_model_requires_an_explicit_mapping(self):
        supplied = config()
        supplied.llm.models[0].name = "local-private-model"
        with self.assertRaisesRegex(ValueError, r"llm.models\[0\].name") as caught:
            self.convert(supplied)
        self.assertNotIn("local-private-model", str(caught.exception))
        result = self.convert(supplied, model_mapping={"local-private-model": "gemini-2.5-flash"})
        self.assertEqual(result["generation_settings"]["models"][0]["name"], "gemini-2.5-flash")
        self.assertEqual(supplied.llm.models[0].name, "local-private-model")

    def test_remapping_warns_by_path_without_disclosing_source_names(self):
        supplied = config()
        supplied.llm.models[0].name = "private-model"
        mapping = {"private-model": "gemini-2.5-flash"}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config_to_alphaevolve(
                supplied, title="Example", problem_description="Problem", model_mapping=mapping
            )
        messages = "\n".join(str(w.message) for w in caught)
        self.assertIn("Model names explicitly remapped: llm.models[0].name", messages)
        self.assertNotIn("private-model", messages)
        self.assertEqual(mapping, {"private-model": "gemini-2.5-flash"})

    def test_duplicate_targets_are_merged_by_summing_relative_weights(self):
        supplied = config()
        supplied.llm.models.append(LLMModelConfig(name="local", weight=2))
        result = self.convert(supplied, model_mapping={"local": "gemini-2.5-flash"})
        self.assertEqual(
            result["generation_settings"]["models"],
            [
                {"name": "gemini-2.5-flash", "weight": 9.0},
                {"name": "gemini-3.1-pro-preview", "weight": 3.0},
            ],
        )
        self.assertEqual(len(supplied.llm.models), 3)

    def test_more_than_two_distinct_targets_are_rejected(self):
        supplied = config()
        supplied.llm.models.append(LLMModelConfig(name="gemini-third", weight=1))
        with self.assertRaisesRegex(ValueError, "at most two"):
            self.convert(supplied)

    def test_invalid_model_names_are_rejected(self):
        for name in (None, "", " ", "Gemini-test", "gemini-", "gemini-name with space"):
            supplied = config()
            supplied.llm.models[0].name = name
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.convert(supplied)

    def test_invalid_weights_are_rejected(self):
        for weight in (None, True, "1", 0, -1, float("nan"), float("inf"), 10**1000):
            supplied = config()
            supplied.llm.models[0].weight = weight
            with self.subTest(weight=weight), self.assertRaisesRegex(ValueError, "weight"):
                self.convert(supplied)

    def test_overflowing_mixture_or_merged_weight_is_rejected(self):
        for same_target in (False, True):
            supplied = config()
            for model in supplied.llm.models:
                model.weight = 1e308
            if same_target:
                supplied.llm.models[1].name = supplied.llm.models[0].name
            with (
                self.subTest(same_target=same_target),
                self.assertRaisesRegex(ValueError, "finite sum"),
            ):
                self.convert(supplied)

    def test_generation_budget_must_be_an_explicit_positive_integer(self):
        for budget in (None, True, 0, -1, 1.5, "12"):
            supplied = config()
            supplied.max_iterations = budget
            with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, "max_iterations"):
                self.convert(supplied)

    def test_generation_concurrency_is_explicit_and_not_copied_from_workers(self):
        supplied = config()
        supplied.evaluator.parallel_evaluations = 8
        self.assertNotIn("concurrency", self.convert(supplied)["run_settings"])
        self.assertEqual(
            self.convert(supplied, generation_concurrency=3)["run_settings"],
            {"max_programs": 12, "concurrency": 3},
        )
        self.assertEqual(supplied.evaluator.parallel_evaluations, 8)
        for concurrency in (True, 0, -1, "3", 1.5):
            with (
                self.subTest(concurrency=concurrency),
                self.assertRaisesRegex(ValueError, "generation_concurrency"),
            ):
                self.convert(supplied, generation_concurrency=concurrency)

    def test_title_and_problem_are_required_nonempty_literal_text(self):
        for field in ("title", "problem_description"):
            for value in (None, "", "  ", 12):
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaisesRegex(ValueError, field),
                ):
                    self.convert(**{field: value})
        result = self.convert(title="标题", problem_description="system_message")
        self.assertEqual(result["title"], "标题")
        self.assertEqual(result["problem_description"], "system_message")

    def test_language_must_explicitly_be_python(self):
        for language in (None, "cpp", "Python", ""):
            supplied = config()
            supplied.language = language
            with self.subTest(language=language), self.assertRaisesRegex(ValueError, "language"):
                self.convert(supplied)

    def test_invalid_mapping_contracts_are_rejected(self):
        for mapping in ([], "gemini-name", {None: "gemini-name"}, {"": "gemini-name"}):
            with self.subTest(mapping=mapping), self.assertRaises((TypeError, ValueError)):
                self.convert(model_mapping=mapping)
        for target in (None, "", "local-name", "gemini-name with space"):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "target"):
                self.convert(model_mapping={"gemini-2.5-flash": target})
        with self.assertRaisesRegex(ValueError, "absent"):
            self.convert(model_mapping={"typo": "gemini-2.5-flash"})

    def test_invalid_config_or_model_collection_is_rejected(self):
        for supplied in (None, {}, "config"):
            with self.subTest(config=supplied), self.assertRaises(TypeError):
                config_to_alphaevolve(supplied, title="Example", problem_description="Problem")
        for models in (None, [], {}, "models", [None], [{"name": "gemini-test"}]):
            supplied = config()
            supplied.llm.models = models
            with self.subTest(models=models), self.assertRaises((TypeError, ValueError)):
                self.convert(supplied)

    def test_legacy_model_aliases_use_the_resolved_ensemble(self):
        supplied = Config.from_dict(
            {
                "language": "python",
                "llm": {
                    "primary_model": "gemini-2.5-flash",
                    "primary_model_weight": 4,
                    "secondary_model": "gemini-3.1-pro-preview",
                    "secondary_model_weight": 1,
                },
            }
        )
        self.assertEqual(
            [entry["weight"] for entry in self.convert(supplied)["generation_settings"]["models"]],
            [4.0, 1.0],
        )

    def test_omission_warning_covers_defaults_and_custom_local_controls_without_values(self):
        supplied = config()
        # A generated sentinel is not an account credential or a hardcoded key.
        sentinel = "private-" + secrets.token_urlsafe(24)
        supplied.llm.api_key = sentinel
        supplied.llm.models[0].api_key = sentinel
        supplied.llm.models[0].api_base = "https://private-endpoint.example"
        supplied.llm.models[0].max_budget_usd = 17
        supplied.prompt.template_dir = "private-template-directory"
        supplied.evaluator.evaluator_args = {"private-argument-name": "private-argument-value"}
        supplied.database.db_path = "private-database-directory"
        supplied.early_stopping_patience = 5
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = config_to_alphaevolve(supplied, title="Example", problem_description="Problem")
        messages = "\n".join(str(w.message) for w in caught)
        for path in (
            "database.num_islands",
            "database.feature_dimensions",
            "database.migration_interval",
            "evaluator.parallel_evaluations",
            "evaluator.timeout",
            "evaluator.evaluator_args",
            "prompt.template_dir",
            "llm.evaluator_models",
            "llm.models[0].api_key",
            "llm.models[0].api_base",
            "llm.models[0].max_budget_usd",
            "early_stopping_patience",
        ):
            self.assertIn(path, messages)
        self.assertIn("including defaults", messages)
        self.assertNotIn(sentinel, messages + json.dumps(result))
        self.assertNotIn("private-", messages + json.dumps(result))

    def test_export_does_not_load_templates_copy_clients_write_files_or_access_network(self):
        class LocalClient:
            def __call__(self, *args, **kwargs):
                raise AssertionError("client must not be called")

            def __deepcopy__(self, memo):
                raise AssertionError("client must not be copied")

        supplied = config()
        supplied.prompt.template_dir = "/nonexistent-template-directory"
        supplied.llm.models[0].init_client = LocalClient()
        supplied.database.novelty_llm = LocalClient()
        with (
            patch("builtins.open", side_effect=AssertionError("file access forbidden")),
            patch("socket.socket.connect", side_effect=AssertionError("network forbidden")),
        ):
            result = self.convert(supplied, problem_description="Literal problem text")
        self.assertEqual(result["problem_description"], "Literal problem text")

    def test_output_is_independent_of_the_input_and_other_exports(self):
        supplied = config()
        first, second = self.convert(supplied), self.convert(supplied)
        first["generation_settings"]["models"][0]["weight"] = 100
        first["run_settings"]["max_programs"] = 1
        self.assertEqual(second["generation_settings"]["models"][0]["weight"], 7)
        self.assertEqual(supplied.llm.models[0].weight, 7)
        self.assertEqual(supplied.max_iterations, 12)

    def test_supported_fields_round_trip_through_the_existing_importer(self):
        supplied = config()
        data = self.convert(supplied)
        imported = config_from_alphaevolve(data, api_base="http://localhost:8080/v1")
        self.assertEqual(imported.max_iterations, supplied.max_iterations)
        self.assertEqual(imported.language, supplied.language)
        self.assertEqual(
            [(model.name, model.weight) for model in imported.llm.models],
            [(model.name, model.weight) for model in supplied.llm.models],
        )
        self.assertEqual(
            imported.prompt.system_message, "Problem description:\n" + data["problem_description"]
        )
        self.assertEqual(imported.llm.models[0].api_base, "http://localhost:8080/v1")


if __name__ == "__main__":
    unittest.main()
