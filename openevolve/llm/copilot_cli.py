"""
GitHub Copilot CLI interface for LLMs.

Uses the GitHub Copilot CLI (`copilot -p`) as a non-interactive LLM backend,
enabling OpenEvolve to run with any model offered by a GitHub Copilot
subscription without requiring direct API keys — authentication is handled by
the CLI's own login session.

Usage in config.yaml:
    llm:
      provider: "copilot_cli"
      models:
        - name: "claude-sonnet-4.6"
          weight: 1.0
          timeout: 300
          reasoning_effort: "medium"
          max_ai_credits: 5.0

Or inject programmatically:
    from openevolve.llm.copilot_cli import init_copilot_cli_client
    for model_cfg in config.llm.models:
        model_cfg.init_client = init_copilot_cli_client

Note that `temperature`, `top_p` and `max_tokens` have no equivalent CLI flag
and are therefore ignored by this backend.
"""

import asyncio
import logging
import subprocess
from typing import Any, Dict, List

from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)

VALID_REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def _cfg_value(model_cfg: Any, attr: str, default: Any) -> Any:
    value = getattr(model_cfg, attr, None)
    return default if value is None else value


class CopilotCLILLM(LLMInterface):
    """LLM interface that uses the GitHub Copilot CLI for generation.

    Requires the `copilot` CLI to be installed and authenticated
    (run `copilot login` first).
    """

    def __init__(self, model_cfg=None):
        self.model = _cfg_value(model_cfg, "name", "auto")
        self.system_message = getattr(model_cfg, "system_message", None)
        self.timeout = _cfg_value(model_cfg, "timeout", 300)
        self.weight = _cfg_value(model_cfg, "weight", 1.0)
        self.retries = _cfg_value(model_cfg, "retries", 3)
        self.retry_delay = _cfg_value(model_cfg, "retry_delay", 5)
        self.reasoning_effort = getattr(model_cfg, "reasoning_effort", None)
        self.max_ai_credits = getattr(model_cfg, "max_ai_credits", None)
        self.allow_all_tools = _cfg_value(model_cfg, "allow_all_tools", False)
        self.cwd = getattr(model_cfg, "cwd", None)
        logger.info(f"Initialized CopilotCLILLM with model: {self.model}")

    async def generate(self, prompt: str, **kwargs) -> str:
        sys_msg = kwargs.pop("system_message", self.system_message) or ""
        return await self.generate_with_context(
            system_message=sys_msg,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        user_content = "\n\n".join(
            m.get("content", "") for m in messages if m.get("role") == "user"
        )

        cmd = self._build_command(system_message, user_content, **kwargs)

        timeout = kwargs.get("timeout", self.timeout)
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)

        loop = asyncio.get_event_loop()
        for attempt in range(retries + 1):
            try:
                return await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: self._run_cli(cmd, timeout)),
                    timeout=timeout + 30,
                )
            except FileNotFoundError:
                logger.error("The `copilot` CLI was not found. Install it and run `copilot login`.")
                raise
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(
                        f"Copilot CLI timeout on attempt {attempt + 1}/{retries + 1}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with timeout")
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Copilot CLI error on attempt {attempt + 1}/{retries + 1}: {e}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with error: {e}")
                    raise

    def _build_command(self, system_message: str, user_content: str, **kwargs) -> List[str]:
        prompt = f"{system_message}\n\n{user_content}" if system_message else user_content

        cmd = [
            "copilot",
            "-p",
            prompt,
            "--model",
            self.model,
            "--silent",
            "--no-color",
            "--no-ask-user",
            "--no-custom-instructions",
            "--disable-builtin-mcps",
        ]

        effort = kwargs.get("reasoning_effort", self.reasoning_effort)
        if effort is not None:
            if effort not in VALID_REASONING_EFFORTS:
                raise ValueError(
                    f"Invalid reasoning_effort: {effort}. "
                    f"Expected one of {', '.join(VALID_REASONING_EFFORTS)}"
                )
            cmd.extend(["--effort", effort])

        credits = kwargs.get("max_ai_credits", self.max_ai_credits)
        if credits is not None:
            cmd.extend(["--max-ai-credits", str(credits)])

        if kwargs.get("allow_all_tools", self.allow_all_tools):
            cmd.append("--allow-all-tools")

        return cmd

    def _run_cli(self, cmd: List[str], timeout: int) -> str:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.cwd,
            )
        except subprocess.TimeoutExpired:
            raise asyncio.TimeoutError("Copilot CLI subprocess timed out")

        stderr = (result.stderr or "").strip()
        if result.returncode != 0:
            raise RuntimeError(
                f"Copilot CLI exited with code {result.returncode}. stderr: {stderr[:500]}"
            )
        if stderr:
            logger.warning(f"Copilot CLI stderr: {stderr[:500]}")

        output = (result.stdout or "").strip()
        if not output:
            raise RuntimeError(f"Empty response from Copilot CLI. stderr: {stderr[:500]}")
        return output


def init_copilot_cli_client(model_cfg):
    """Factory function compatible with OpenEvolve's init_client config hook."""
    return CopilotCLILLM(model_cfg)
