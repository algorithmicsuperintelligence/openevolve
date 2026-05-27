"""
Claude Code LLM interface.

Wraps the official `claude-agent-sdk` so OpenEvolve can call Claude through a
locally-installed Claude Code CLI session instead of an Anthropic API key. When
the user is logged into Claude Code (Pro/Max subscription), no API key is
required — the SDK inherits the CLI's auth.

Notes / limitations vs OpenAILLM:
  - `temperature`, `top_p`, `seed` are not exposed by the SDK; ignored.
  - `max_tokens` is not directly settable; `max_thinking_tokens` is supported.
  - The agent loop is forced to a single turn with all tools disabled so the
    call behaves as a pure prompt-in / text-out completion.
  - Subscription rate limits (5-hour windows) apply; long evolution runs may
    hit them quickly. Tune `iterations` and ensemble weights accordingly.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)


class ClaudeCodeLLM(LLMInterface):
    """LLM interface backed by claude-agent-sdk (Claude Code session auth)."""

    def __init__(self, model_cfg: Optional[dict] = None):
        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "ClaudeCodeLLM requires the `claude-agent-sdk` package. "
                "Install with: pip install claude-agent-sdk"
            ) from e

        self.model = model_cfg.name
        self.system_message = model_cfg.system_message
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries if model_cfg.retries is not None else 0
        self.retry_delay = model_cfg.retry_delay if model_cfg.retry_delay is not None else 0
        self.max_thinking_tokens = getattr(model_cfg, "max_thinking_tokens", None)
        self.reasoning_effort = getattr(model_cfg, "reasoning_effort", None)
        self.cli_path = getattr(model_cfg, "cli_path", None)
        self.cwd = getattr(model_cfg, "cwd", None)
        self.extra_sdk_options: Dict[str, Any] = (
            getattr(model_cfg, "claude_code_options", None) or {}
        )

        if not hasattr(logger, "_initialized_models"):
            logger._initialized_models = set()
        key = f"claude_code::{self.model}"
        if key not in logger._initialized_models:
            logger.info(f"Initialized Claude Code LLM with model: {self.model}")
            logger._initialized_models.add(key)

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        prompt_text = self._messages_to_prompt(messages)
        timeout = kwargs.get("timeout", self.timeout)
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)

        for attempt in range(retries + 1):
            try:
                coro = self._query_once(system_message, prompt_text, **kwargs)
                if timeout is not None:
                    return await asyncio.wait_for(coro, timeout=timeout)
                return await coro
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(
                        f"[claude_code] Timeout {attempt + 1}/{retries + 1}, retrying."
                    )
                    await asyncio.sleep(retry_delay)
                    continue
                logger.error(f"[claude_code] All {retries + 1} attempts timed out")
                raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"[claude_code] Error {attempt + 1}/{retries + 1}: {e}. Retrying."
                    )
                    await asyncio.sleep(retry_delay)
                    continue
                logger.error(f"[claude_code] All {retries + 1} attempts failed: {e}")
                raise

    async def _query_once(
        self, system_message: Optional[str], prompt_text: str, **kwargs
    ) -> str:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            query,
        )

        opts_kwargs: Dict[str, Any] = {
            "max_turns": 4,
            "allowed_tools": [],
            "disallowed_tools": [],
            "permission_mode": "bypassPermissions",
        }
        if system_message:
            opts_kwargs["system_prompt"] = system_message
        if self.model:
            opts_kwargs["model"] = self.model
        if self.max_thinking_tokens is not None:
            opts_kwargs["max_thinking_tokens"] = self.max_thinking_tokens
        effort = kwargs.get("reasoning_effort", self.reasoning_effort)
        if effort is not None:
            opts_kwargs["effort"] = effort
        if self.cli_path is not None:
            opts_kwargs["cli_path"] = self.cli_path
        if self.cwd is not None:
            opts_kwargs["cwd"] = self.cwd
        opts_kwargs.update(self.extra_sdk_options)

        options = ClaudeAgentOptions(**opts_kwargs)

        text_chunks: List[str] = []
        result_text: Optional[str] = None

        async for msg in query(prompt=prompt_text, options=options):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        text_chunks.append(block.text)
            elif isinstance(msg, ResultMessage):
                result_text = getattr(msg, "result", None) or result_text

        final = result_text if result_text else "".join(text_chunks)
        logger.debug(f"[claude_code] response chars={len(final)}")
        return final

    @staticmethod
    def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
        """Flatten chat history into one prompt string.

        The SDK's one-shot `query()` takes a single user prompt, not a chat
        array. We join roles inline so the model still sees prior turns.
        """
        if len(messages) == 1 and messages[0].get("role") == "user":
            return messages[0].get("content", "")
        parts: List[str] = []
        for m in messages:
            role = str(m.get("role", "user")).upper()
            parts.append(f"### {role}\n{m.get('content', '')}")
        return "\n\n".join(parts).strip() + "\n"
