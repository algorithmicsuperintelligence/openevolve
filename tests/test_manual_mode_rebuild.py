"""Manual mode survives the model rebuild used by CLI overrides."""

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openevolve import cli
from openevolve.config import Config
from openevolve.llm.openai import OpenAILLM


class TestManualModeRebuild(unittest.TestCase):
    def test_rebuild_preserves_manual_setting_for_both_ensembles(self):
        for setting in (True, False, None):
            with self.subTest(manual_mode=setting):
                values = {"primary_model": "old-primary", "secondary_model": "old-secondary"}
                if setting is not None:
                    values["manual_mode"] = setting
                config = Config.from_dict({"llm": values})
                config.llm.primary_model = "new-primary"
                config.llm.secondary_model = "new-secondary"

                config.llm.rebuild_models()

                expected = setting if setting is not None else False
                for models in (config.llm.models, config.llm.evaluator_models):
                    self.assertEqual(
                        [model.name for model in models], ["new-primary", "new-secondary"]
                    )
                    self.assertEqual([model.manual_mode for model in models], [expected, expected])

    def test_cli_model_overrides_keep_manual_mode(self):
        class StopBeforeEvolution(BaseException):
            pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / "initial.py"
            evaluator = root / "evaluate.py"
            config_file = root / "config.yaml"
            initial.write_text("def run():\n    return 1\n")
            evaluator.write_text("def evaluate(path):\n    return {'score': 1.0}\n")
            config_file.write_text("llm:\n  manual_mode: true\n  primary_model: original\n")
            for flag in ("--primary-model", "--secondary-model"):
                with self.subTest(flag=flag):
                    captured = []

                    def capture_config(*args, **kwargs):
                        captured.append(kwargs["config"])
                        raise StopBeforeEvolution()

                    argv = [
                        "openevolve-run",
                        str(initial),
                        str(evaluator),
                        "--config",
                        str(config_file),
                        flag,
                        "override",
                    ]
                    with (
                        patch.object(sys, "argv", argv),
                        patch.object(cli, "OpenEvolve", side_effect=capture_config),
                    ):
                        with self.assertRaises(StopBeforeEvolution):
                            asyncio.run(cli.main_async())
                    self.assertEqual(len(captured), 1)
                    llm = captured[0].llm
                    self.assertTrue(llm.manual_mode)
                    for models in (llm.models, llm.evaluator_models):
                        self.assertIn("override", [model.name for model in models])
                        self.assertTrue(all(model.manual_mode is True for model in models))

    def test_rebuilt_models_select_manual_or_api_provider_as_configured(self):
        with tempfile.TemporaryDirectory() as directory:
            for setting in (True, False, None):
                with self.subTest(manual_mode=setting):
                    values = {"primary_model": "original"}
                    if setting is not None:
                        values["manual_mode"] = setting
                    config = Config.from_dict({"llm": values})
                    config.llm.primary_model = "override"
                    config.llm.rebuild_models()
                    for model in config.llm.models + config.llm.evaluator_models:
                        # The controller supplies the queue directory at startup.
                        model._manual_queue_dir = directory
                        with patch("openevolve.llm.openai.openai.AsyncOpenAI") as client:
                            provider = OpenAILLM(model)
                        self.assertEqual(provider.manual_mode, setting is True)
                        if setting is True:
                            client.assert_not_called()
                            self.assertIsNone(provider.client)
                            self.assertEqual(provider.manual_queue_dir, Path(directory).resolve())
                        else:
                            client.assert_called_once()


if __name__ == "__main__":
    unittest.main()
