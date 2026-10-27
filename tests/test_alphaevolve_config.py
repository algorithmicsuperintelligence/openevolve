"""Offline configuration import for local Python AlphaEvolve examples (#468)."""

import copy
import tempfile
import unittest
import warnings

from openevolve.api import run_evolution
from openevolve.compat.alphaevolve import adapt_evaluator, config_from_alphaevolve
from openevolve.config import Config
from openevolve.prompt.sampler import PromptSampler


def experiment():
    return {
        "title": "Circle Packing",
        "program_language": "python",
        "problem_description": "Pack circles into a unit square.",
        "run_settings": {"max_programs": 10},
        "generation_settings": {
            "models": [{"name": "local-fast", "weight": 7}, {"name": "local-large", "weight": 3}]
        },
    }


class OfflineModel:
    def __init__(self, config):
        self.model = config.name

    async def generate_with_context(self, system_message, messages, **kwargs):
        assert "Pack circles into a unit square." in system_message
        return "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE"


def offline_model(config):
    return OfflineModel(config)


class TestAlphaEvolveConfig(unittest.TestCase):
    def convert(self, data, **kwargs):
        return config_from_alphaevolve(data, api_base="http://localhost:8080/v1", **kwargs)

    def test_imports_models_budget_language_and_problem_without_mutating_input(self):
        data = experiment()
        original = copy.deepcopy(data)
        config = self.convert(data, api_key="local-placeholder")
        self.assertEqual(data, original)
        self.assertEqual(config.max_iterations, 10)
        self.assertEqual(config.language, "python")
        self.assertEqual(config.file_suffix, ".py")
        self.assertEqual(
            [(m.name, m.weight) for m in config.llm.models],
            [("local-fast", 7.0), ("local-large", 3.0)],
        )
        for model in config.llm.models + config.llm.evaluator_models:
            self.assertEqual(model.api_base, "http://localhost:8080/v1")
            self.assertEqual(model.api_key, "local-placeholder")
        prompt = PromptSampler(config.prompt).build_prompt(current_program="x = 1")
        self.assertIn(data["problem_description"], prompt["system"])
        self.assertFalse(config.evaluator.cascade_evaluation)

    def test_missing_or_null_weight_defaults_to_one(self):
        for models in ([{"name": "local"}], [{"name": "local", "weight": None}]):
            data = experiment()
            data["generation_settings"]["models"] = models
            self.assertEqual(self.convert(data).llm.models[0].weight, 1.0)

    def test_description_cannot_be_mistaken_for_a_template_name(self):
        data = experiment()
        data["problem_description"] = "system_message"
        config = self.convert(data)
        self.assertEqual(
            PromptSampler(config.prompt).build_prompt(current_program="x=1")["system"],
            "Problem description:\nsystem_message",
        )

    def test_context_is_included_in_prompt(self):
        data = experiment()
        data["generation_settings"]["context"] = "Prefer short programs."
        prompt = self.convert(data).prompt.system_message
        self.assertIn(data["problem_description"], prompt)
        self.assertIn("Prefer short programs.", prompt)

    def test_missing_or_invalid_generation_budget_never_falls_back_to_10000(self):
        for budget in (None, 0, -1, True, 1.5, "10"):
            data = experiment()
            data["run_settings"]["max_programs"] = budget
            with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, "max_programs"):
                self.convert(data)

    def test_missing_or_malformed_sections_raise(self):
        for key in ("run_settings", "generation_settings"):
            for value in (None, [], "not a mapping"):
                data = experiment()
                data[key] = value
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    self.convert(data)
        with self.assertRaises(ValueError):
            self.convert([])

    def test_invalid_model_mixtures_raise(self):
        for models in (
            None,
            [],
            "local",
            [None],
            [{}],
            [{"name": ""}],
            [{"name": "local", "weight": 0}],
            [{"name": "local", "weight": -1}],
            [{"name": "local", "weight": True}],
            [{"name": "local", "weight": "1"}],
            [{"name": "local", "weight": float("nan")}],
            [{"name": "local", "weight": float("inf")}],
            [{"name": "a", "weight": 1e308}, {"name": "b", "weight": 1e308}],
        ):
            data = experiment()
            data["generation_settings"]["models"] = models
            with self.subTest(models=models), self.assertRaises(ValueError):
                self.convert(data)

    def test_language_description_endpoint_and_context_are_explicit(self):
        for key, value in (
            ("program_language", "cpp"),
            ("program_language", None),
            ("problem_description", ""),
            ("problem_description", None),
        ):
            data = experiment()
            data[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.convert(data)
        for endpoint in (None, "", "   "):
            with self.subTest(endpoint=endpoint), self.assertRaisesRegex(ValueError, "api_base"):
                config_from_alphaevolve(experiment(), api_base=endpoint)
        data = experiment()
        data["generation_settings"]["context"] = 42
        with self.assertRaisesRegex(ValueError, "context"):
            self.convert(data)

    def test_unsupported_settings_warn_by_path_without_leaking_values(self):
        data = experiment()
        data["run_settings"]["concurrency"] = 4
        data["generation_settings"]["include_full_program_in_prompt"] = False
        data["generation_settings"]["models"][0]["api_key"] = "do-not-print-me"
        data["project_id"] = "private-project"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = self.convert(data)
        messages = "\n".join(str(w.message) for w in caught)
        for path in (
            "run_settings.concurrency",
            "generation_settings.include_full_program_in_prompt",
            "generation_settings.models[0].api_key",
            "project_id",
        ):
            self.assertIn(path, messages)
        self.assertNotIn("do-not-print-me", messages)
        self.assertNotIn("private-project", messages)
        self.assertIsNone(config.llm.models[0].api_key)
        self.assertEqual(config.evaluator.parallel_evaluations, 1)

    def test_unimplemented_wall_clock_limit_is_rejected(self):
        data = experiment()
        data["run_settings"]["max_duration"] = "60s"
        with self.assertRaisesRegex(ValueError, "max_duration"):
            self.convert(data)

    def test_sdk_null_optional_fields_do_not_warn(self):
        data = experiment()
        data["run_settings"]["max_duration"] = None
        data["run_settings"]["concurrency"] = None
        data["generation_settings"]["context"] = None
        data["generation_settings"]["include_full_program_in_prompt"] = None
        data["evolution_settings"] = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.convert(data)
        self.assertEqual(caught, [])

    def test_configs_are_independent(self):
        first, second = self.convert(experiment()), self.convert(experiment())
        first.llm.models[0].weight = 50
        self.assertEqual(second.llm.models[0].weight, 7)
        self.assertEqual(Config().max_iterations, 10000)

    def test_imported_config_runs_with_existing_adapter_and_real_worker(self):
        data = experiment()
        data["run_settings"]["max_programs"] = 1
        config = self.convert(data)
        for model in config.llm.models + config.llm.evaluator_models:
            model.init_client = offline_model
        config.database.num_islands = 1

        def score(candidate):
            namespace = {}
            exec(candidate["content"]["files"][0]["content"], namespace)
            return {"scores": {"scores": [{"metric": "fitness", "score": namespace["x"]}]}}

        with tempfile.TemporaryDirectory() as directory:
            result = run_evolution(
                "# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END\n",
                adapt_evaluator(score, objective_metric="fitness"),
                config=config,
                output_dir=directory,
            )
            self.assertEqual(result.best_score, 2)
            self.assertEqual(result.best_program.generation, 1)


if __name__ == "__main__":
    unittest.main()
