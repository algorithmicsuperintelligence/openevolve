"""Regression tests for per-island evolution system messages."""

import asyncio
import tempfile
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from openevolve import process_parallel
from openevolve.config import Config
from openevolve.controller import OpenEvolve
from openevolve.database import Program, ProgramDatabase
from openevolve.prompt.sampler import PromptSampler


class TestIslandSystemMessages(unittest.TestCase):
    def test_defaults_overrides_and_templates(self):
        config = Config.from_dict(
            {
                "database": {"num_islands": 4},
                "prompt": {
                    "system_message": "default",
                    "island_system_messages": ["island zero", None, "evaluator_system_message"],
                },
            }
        )
        sampler = PromptSampler(config.prompt)

        self.assertEqual(sampler.build_prompt()["system"], "default")
        self.assertEqual(sampler.build_prompt(island_id=0)["system"], "island zero")
        self.assertEqual(sampler.build_prompt(island_id=1)["system"], "default")
        self.assertIn("expert code reviewer", sampler.build_prompt(island_id=2)["system"])
        self.assertEqual(sampler.build_prompt(island_id=3)["system"], "default")
        default_sampler = PromptSampler(Config().prompt)
        self.assertEqual(
            default_sampler.build_prompt()["system"],
            default_sampler.build_prompt(island_id=0)["system"],
        )

    def test_rejects_more_entries_than_islands(self):
        with self.assertRaisesRegex(ValueError, "island_system_messages.*num_islands"):
            Config.from_dict(
                {
                    "database": {"num_islands": 1},
                    "prompt": {"island_system_messages": ["first", "second"]},
                }
            )

    def test_parallel_worker_calls_use_scheduled_island(self):
        config = Config.from_dict(
            {
                "database": {"num_islands": 3},
                "prompt": {
                    "system_message": "default",
                    "island_system_messages": ["parent island", "island one", "island two"],
                },
            }
        )
        config.language = "python"
        parent = Program(
            id="parent", code="x = 1", metrics={"combined_score": 0.5}, metadata={"island": 0}
        )
        database = ProgramDatabase(config.database)
        database.add(parent)
        controller = process_parallel.ProcessParallelController(config, "evaluator.py", database)
        with (
            patch.object(database, "sample_from_island", return_value=(parent, [])),
            patch.object(controller, "executor") as executor,
        ):
            for iteration, island_id in enumerate((2, 1, 2)):
                controller._submit_iteration(iteration, island_id=island_id)
        submitted = [call.args[1:] for call in executor.submit.call_args_list]
        llm = MagicMock()
        llm.generate_with_context = AsyncMock(return_value="no diff")

        with (
            patch.object(process_parallel, "_lazy_init_worker_components"),
            patch.multiple(
                process_parallel,
                create=True,
                _worker_config=config,
                _worker_prompt_sampler=PromptSampler(config.prompt),
                _worker_llm_ensemble=llm,
            ),
        ):
            with ThreadPoolExecutor(max_workers=3) as executor:
                list(
                    executor.map(
                        lambda args: process_parallel._run_iteration_worker(*args),
                        submitted,
                    )
                )

        self.assertEqual(
            Counter(
                call.kwargs["system_message"] for call in llm.generate_with_context.call_args_list
            ),
            Counter(["island two", "island one", "island two"]),
        )

    def test_checkpoint_resume_with_same_config_keeps_override(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(
                "database:\n  num_islands: 2\nprompt:\n"
                "  system_message: default\n"
                "  island_system_messages: [island zero, island one]\n",
                encoding="utf-8",
            )
            config = Config.from_yaml(config_path)
            program_path = Path(directory) / "initial.py"
            program_path.write_text("x = 1\n", encoding="utf-8")
            evaluator_path = Path(directory) / "evaluator.py"
            evaluator_path.write_text("", encoding="utf-8")
            with (
                patch.object(OpenEvolve, "_setup_logging"),
                patch("openevolve.controller.LLMEnsemble"),
                patch("openevolve.controller.Evaluator"),
            ):
                original = OpenEvolve(
                    str(program_path), str(evaluator_path), config, output_dir=directory
                )
                original.database.add(
                    Program(id="parent", code="x = 1", metrics={"combined_score": 1.0})
                )
                original._save_checkpoint(3)
                checkpoint = Path(directory) / "checkpoints" / "checkpoint_3"

                resumed = OpenEvolve(
                    str(program_path),
                    str(evaluator_path),
                    Config.from_yaml(config_path),
                    output_dir=directory,
                )
                with patch("openevolve.controller.ProcessParallelController") as parallel:
                    parallel.return_value.run_evolution = AsyncMock(return_value=None)
                    asyncio.run(resumed.run(iterations=1, checkpoint_path=str(checkpoint)))

            self.assertEqual(resumed.database.last_iteration, 3)
            self.assertIn("parent", resumed.database.programs)
            self.assertEqual(
                resumed.prompt_sampler.build_prompt(island_id=1)["system"],
                "island one",
            )


if __name__ == "__main__":
    unittest.main()
