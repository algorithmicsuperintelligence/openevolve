"""Tests for the Claude Code CLI LLM backend."""

import asyncio
import unittest
from unittest.mock import MagicMock, patch

from openevolve.llm.claude_code import ClaudeCodeLLM, init_claude_code_client


class TestClaudeCodeLLM(unittest.TestCase):
    def _make_cfg(self, **overrides):
        cfg = MagicMock()
        cfg.name = overrides.get("name", "sonnet")
        cfg.system_message = overrides.get("system_message", None)
        cfg.max_tokens = overrides.get("max_tokens", 16000)
        cfg.timeout = overrides.get("timeout", 60)
        cfg.weight = overrides.get("weight", 1.0)
        cfg.retries = overrides.get("retries", 3)
        cfg.retry_delay = overrides.get("retry_delay", 5)
        cfg.max_budget_usd = overrides.get("max_budget_usd", 1.0)
        cfg.cwd = overrides.get("cwd", None)
        return cfg

    def test_init_defaults(self):
        llm = ClaudeCodeLLM(self._make_cfg())
        self.assertEqual(llm.model, "sonnet")
        self.assertEqual(llm.max_tokens, 16000)
        self.assertEqual(llm.timeout, 60)
        self.assertEqual(llm.weight, 1.0)

    def test_init_with_custom_model(self):
        llm = ClaudeCodeLLM(self._make_cfg(name="opus"))
        self.assertEqual(llm.model, "opus")

    def test_factory_function(self):
        cfg = self._make_cfg()
        llm = init_claude_code_client(cfg)
        self.assertIsInstance(llm, ClaudeCodeLLM)
        self.assertEqual(llm.model, "sonnet")

    @patch("openevolve.llm.claude_code.subprocess.run")
    def test_generate_calls_cli(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="Generated response text", stderr="")
        llm = ClaudeCodeLLM(self._make_cfg(timeout=10))
        result = asyncio.run(llm.generate("test prompt"))
        self.assertEqual(result, "Generated response text")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[0], "claude")
        self.assertIn("-p", cmd)
        self.assertIn("--model", cmd)
        self.assertIn("sonnet", cmd)
        self.assertIn("test prompt", cmd)

    @patch("openevolve.llm.claude_code.subprocess.run")
    def test_system_message_passed(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="response", stderr="")
        llm = ClaudeCodeLLM(self._make_cfg(timeout=10))
        asyncio.run(llm.generate("prompt", system_message="You are an expert."))
        cmd = mock_run.call_args[0][0]
        idx = cmd.index("--system-prompt")
        self.assertEqual(cmd[idx + 1], "You are an expert.")

    @patch("openevolve.llm.claude_code.subprocess.run")
    def test_empty_response_raises(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error msg")
        llm = ClaudeCodeLLM(self._make_cfg(timeout=10, retries=0))
        with self.assertRaises(RuntimeError):
            asyncio.run(llm.generate("test prompt"))

    @patch("openevolve.llm.claude_code.subprocess.run")
    def test_retry_on_failure(self, mock_run):
        mock_run.side_effect = [
            MagicMock(returncode=1, stdout="", stderr="transient error"),
            MagicMock(returncode=0, stdout="success after retry", stderr=""),
        ]
        llm = ClaudeCodeLLM(self._make_cfg(timeout=10, retries=1, retry_delay=0))
        result = asyncio.run(llm.generate("test prompt", retry_delay=0))
        self.assertEqual(result, "success after retry")
        self.assertEqual(mock_run.call_count, 2)

    @patch("openevolve.llm.claude_code.subprocess.run")
    def test_retries_exhausted_raises(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="persistent error")
        llm = ClaudeCodeLLM(self._make_cfg(timeout=10, retries=2, retry_delay=0))
        with self.assertRaises(RuntimeError):
            asyncio.run(llm.generate("test prompt", retry_delay=0))
        self.assertEqual(mock_run.call_count, 3)

    @patch("openevolve.llm.claude_code.subprocess.run")
    def test_generate_with_context(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ctx response", stderr="")
        llm = ClaudeCodeLLM(self._make_cfg(timeout=10))
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
        cmd = mock_run.call_args[0][0]
        self.assertIn("first\n\nsecond", cmd[-1])


class TestMaxBudgetConfig(unittest.TestCase):
    def test_max_budget_usd_in_model_config(self):
        from openevolve.config import LLMModelConfig

        cfg = LLMModelConfig(max_budget_usd=2.5)
        self.assertEqual(cfg.max_budget_usd, 2.5)

    def test_max_budget_usd_default_none(self):
        from openevolve.config import LLMModelConfig

        cfg = LLMModelConfig()
        self.assertIsNone(cfg.max_budget_usd)

    def test_max_budget_usd_from_dict(self):
        from openevolve.config import Config

        config = Config.from_dict(
            {
                "llm": {
                    "provider": "claude_code",
                    "models": [{"name": "sonnet", "max_budget_usd": 3.0, "weight": 1.0}],
                }
            }
        )
        self.assertEqual(config.llm.models[0].max_budget_usd, 3.0)


class TestProviderRegistry(unittest.TestCase):
    def test_claude_code_in_registry(self):
        from openevolve.llm.ensemble import _PROVIDER_REGISTRY

        self.assertIn("claude_code", _PROVIDER_REGISTRY)

    def test_ensemble_creates_claude_code(self):
        from openevolve.llm.ensemble import _create_model

        cfg = MagicMock()
        cfg.init_client = None
        cfg.provider = "claude_code"
        cfg.name = "sonnet"
        cfg.system_message = None
        cfg.max_tokens = 4096
        cfg.timeout = 60
        cfg.weight = 1.0
        cfg.max_budget_usd = 1.0
        cfg.cwd = None
        model = _create_model(cfg)
        self.assertIsInstance(model, ClaudeCodeLLM)


if __name__ == "__main__":
    unittest.main()
