"""
LLM module initialization
"""

from openevolve.llm.base import LLMInterface
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.llm.openai import OpenAILLM
from openevolve.llm.claude_code import ClaudeCodeLLM, init_claude_code_client
from openevolve.llm.copilot_cli import CopilotCLILLM, init_copilot_cli_client
from openevolve.llm.orcarouter import (
    PROVIDER_API_KEY as ORCAROUTER_PROVIDER_API_KEY,
    PROVIDER_OAUTH as ORCAROUTER_PROVIDER_OAUTH,
    OrcaRouterLLM,
    init_orcarouter_client,
    init_orcarouter_oauth_client,
)

__all__ = [
    "LLMInterface",
    "OpenAILLM",
    "ClaudeCodeLLM",
    "init_claude_code_client",
    "CopilotCLILLM",
    "init_copilot_cli_client",
    "LLMEnsemble",
    "OrcaRouterLLM",
    "init_orcarouter_client",
    "init_orcarouter_oauth_client",
    "ORCAROUTER_PROVIDER_API_KEY",
    "ORCAROUTER_PROVIDER_OAUTH",
]
