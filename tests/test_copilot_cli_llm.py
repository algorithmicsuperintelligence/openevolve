"""Tests for the GitHub Copilot CLI LLM backend."""

import asyncio
import unittest
from unittest.mock import MagicMock, patch

from openevolve.llm.copilot_cli import CopilotCLILLM, init_copilot_cli_client


def _make_cfg(**overrides):
    defaults = {
        "name": "claude-sonnet-4.6",
        "system_message": None,
        "timeout": 10,
        "weight": 1.0,
        "retries": 3,
        "retry_delay": 5,
        "reasoning_effort": None,
        "max_ai_credits": None,
        "allow_all_tools": None,
        "cwd": None,
    }
    defaults.update(overrides)
    cfg = MagicMock()
    for key, value in defaults.items():
        setattr(cfg, key, value)
    return cfg


def _ok(stdout="Generated response text"):
    return MagicMock(returncode=0, stdout=stdout, stderr="")


class TestCopilotCLILLM(unittest.TestCase):
    def test_init_defaults(self):
        llm = CopilotCLILLM(_make_cfg())
        self.assertEqual(llm.model, "claude-sonnet-4.6")
        self.assertEqual(llm.timeout, 10)
        self.assertEqual(llm.weight, 1.0)
        self.assertEqual(llm.retries, 3)
        self.assertFalse(llm.allow_all_tools)

    def test_init_falls_back_when_config_values_are_none(self):
        llm = CopilotCLILLM(_make_cfg(name=None, timeout=None, retries=None, retry_delay=None))
        self.assertEqual(llm.model, "auto")
        self.assertEqual(llm.timeout, 300)
        self.assertEqual(llm.retries, 3)
        self.assertEqual(llm.retry_delay, 5)

    def test_init_keeps_zero_retry_delay(self):
        llm = CopilotCLILLM(_make_cfg(retry_delay=0))
        self.assertEqual(llm.retry_delay, 0)

    def test_factory_function(self):
        llm = init_copilot_cli_client(_make_cfg())
        self.assertIsInstance(llm, CopilotCLILLM)
        self.assertEqual(llm.model, "claude-sonnet-4.6")

    @patch("openevolve.llm.copilot_cli.subprocess.run")
    def test_generate_calls_cli(self, mock_run):
        mock_run.return_value = _ok()
        llm = CopilotCLILLM(_make_cfg())
        result = asyncio.run(llm.generate("test prompt"))
        self.assertEqual(result, "Generated response text")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[0], "copilot")
        self.assertEqual(cmd[cmd.index("-p") + 1], "test prompt")
        self.assertEqual(cmd[cmd.index("--model") + 1], "claude-sonnet-4.6")

    @patch("openevolve.llm.copilot_cli.subprocess.run")
    def test_non_interactive_flags_are_set(self, mock_run):
        mock_run.return_value = _ok()
        llm = CopilotCLILLM(_make_cfg())
        asyncio.run(llm.generate("test prompt"))
        cmd = mock_run.call_args[0][0]
        for flag in (
            "--silent",
            "--no-color",
            "--no-ask-user",
            "--no-custom-instructions",
            "--disable-builtin-mcps",
        ):
            self.assertIn(flag, cmd)

    @patch("openevolve.llm.copilot_cli.subprocess.run")
    def test_system_message_is_prepended_to_prompt(self, mock_run):
        mock_run.return_value = _ok()
        llm = CopilotCLILLM(_make_cfg())
        asyncio.run(llm.generate("prompt", system_message="You are an expert."))
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[cmd.index("-p") + 1], "You are an expert.\n\nprompt")

    @patch("openevolve.llm.copilot_cli.subprocess.run")
    def test_reasoning_effort_forwarded(self, mock_run):
        mock_run.return_value = _ok()
        llm = CopilotCLILLM(_make_cfg(reasoning_effort="high"))
        asyncio.run(llm.generate("prompt"))
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[cmd.index("--effort") + 1], "high")

    @patch("openevolve.llm.copilot_cli.subprocess.run")
    def test_invalid_reasoning_effort_raises(self, mock_run):
        llm = CopilotCLILLM(_make_cfg(reasoning_effort="turbo", retries=0))
        with self.assertRaisesRegex(ValueError, "Invalid reasoning_effort: turbo"):
            asyncio.run(llm.generate("prompt"))
        mock_run.assert_not_called()

    @patch("openevolve.llm.copilot_cli.subprocess.run")
    def test_max_ai_credits_forwarded(self, mock_run):
        mock_run.return_value = _ok()
        llm = CopilotCLILLM(_make_cfg(max_ai_credits=2.5))
        asyncio.run(llm.generate("prompt"))
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[cmd.index("--max-ai-credits") + 1], "2.5")

    @patch("openevolve.llm.copilot_cli.subprocess.run")
    def test_tools_are_not_approved_by_default(self, mock_run):
        mock_run.return_value = _ok()
        llm = CopilotCLILLM(_make_cfg())
        asyncio.run(llm.generate("prompt"))
        self.assertNotIn("--allow-all-tools", mock_run.call_args[0][0])

    @patch("openevolve.llm.copilot_cli.subprocess.run")
    def test_allow_all_tools_opt_in(self, mock_run):
        mock_run.return_value = _ok()
        llm = CopilotCLILLM(_make_cfg(allow_all_tools=True))
        asyncio.run(llm.generate("prompt"))
        self.assertIn("--allow-all-tools", mock_run.call_args[0][0])

    @patch("openevolve.llm.copilot_cli.subprocess.run")
    def test_non_zero_exit_raises(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="model not available")
        llm = CopilotCLILLM(_make_cfg(retries=0))
        with self.assertRaisesRegex(RuntimeError, "model not available"):
            asyncio.run(llm.generate("test prompt"))

    @patch("openevolve.llm.copilot_cli.subprocess.run")
    def test_empty_response_raises(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="   ", stderr="")
        llm = CopilotCLILLM(_make_cfg(retries=0))
        with self.assertRaisesRegex(RuntimeError, "Empty response"):
            asyncio.run(llm.generate("test prompt"))

    @patch("openevolve.llm.copilot_cli.subprocess.run")
    def test_missing_cli_raises_without_retrying(self, mock_run):
        mock_run.side_effect = FileNotFoundError("copilot")
        llm = CopilotCLILLM(_make_cfg(retries=3, retry_delay=0))
        with self.assertRaises(FileNotFoundError):
            asyncio.run(llm.generate("test prompt"))
        self.assertEqual(mock_run.call_count, 1)

    @patch("openevolve.llm.copilot_cli.subprocess.run")
    def test_retry_on_failure(self, mock_run):
        mock_run.side_effect = [
            MagicMock(returncode=1, stdout="", stderr="transient error"),
            _ok("success after retry"),
        ]
        llm = CopilotCLILLM(_make_cfg(retries=1, retry_delay=0))
        result = asyncio.run(llm.generate("test prompt", retry_delay=0))
        self.assertEqual(result, "success after retry")
        self.assertEqual(mock_run.call_count, 2)

    @patch("openevolve.llm.copilot_cli.subprocess.run")
    def test_retries_exhausted_raises(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="persistent error")
        llm = CopilotCLILLM(_make_cfg(retries=2, retry_delay=0))
        with self.assertRaises(RuntimeError):
            asyncio.run(llm.generate("test prompt", retry_delay=0))
        self.assertEqual(mock_run.call_count, 3)

    @patch("openevolve.llm.copilot_cli.subprocess.run")
    def test_generate_with_context(self, mock_run):
        mock_run.return_value = _ok("ctx response")
        llm = CopilotCLILLM(_make_cfg())
        result = asyncio.run(
            llm.generate_with_context(
                system_message="sys",
                messages=[
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "ignored"},
                    {"role": "user", "content": "second"},
                ],
            )
        )
        self.assertEqual(result, "ctx response")
        prompt = mock_run.call_args[0][0][2]
        self.assertEqual(prompt, "sys\n\nfirst\n\nsecond")
        self.assertNotIn("ignored", prompt)


class TestCopilotCLIConfig(unittest.TestCase):
    def test_new_fields_default_to_none(self):
        from openevolve.config import LLMModelConfig

        cfg = LLMModelConfig()
        self.assertIsNone(cfg.max_ai_credits)
        self.assertIsNone(cfg.allow_all_tools)

    def test_fields_from_dict(self):
        from openevolve.config import Config

        config = Config.from_dict(
            {
                "llm": {
                    "provider": "copilot_cli",
                    "models": [
                        {
                            "name": "claude-sonnet-4.6",
                            "weight": 1.0,
                            "max_ai_credits": 5.0,
                            "allow_all_tools": True,
                            "reasoning_effort": "medium",
                        }
                    ],
                }
            }
        )
        model = config.llm.models[0]
        self.assertEqual(model.provider, "copilot_cli")
        self.assertEqual(model.max_ai_credits, 5.0)
        self.assertTrue(model.allow_all_tools)
        self.assertEqual(model.reasoning_effort, "medium")


class TestProviderRegistry(unittest.TestCase):
    def test_copilot_cli_in_registry(self):
        from openevolve.llm.ensemble import _PROVIDER_REGISTRY

        self.assertIn("copilot_cli", _PROVIDER_REGISTRY)

    def test_ensemble_creates_copilot_cli(self):
        from openevolve.llm.ensemble import _create_model

        cfg = _make_cfg()
        cfg.init_client = None
        cfg.provider = "copilot_cli"
        self.assertIsInstance(_create_model(cfg), CopilotCLILLM)

    def test_provider_propagates_from_llm_config(self):
        from openevolve.config import Config

        config = Config.from_dict(
            {
                "llm": {
                    "provider": "copilot_cli",
                    "models": [{"name": "gpt-5.4", "weight": 1.0}],
                }
            }
        )
        self.assertEqual(config.llm.models[0].provider, "copilot_cli")
        self.assertEqual(config.llm.evaluator_models[0].provider, "copilot_cli")


if __name__ == "__main__":
    unittest.main()
