"""
LLM module initialization
"""

from openevolve.llm.base import LLMInterface
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.llm.openai import OpenAILLM
from openevolve.llm.claude_code import ClaudeCodeLLM, init_claude_code_client
from openevolve.llm.copilot_cli import CopilotCLILLM, init_copilot_cli_client

__all__ = [
    "LLMInterface",
    "OpenAILLM",
    "ClaudeCodeLLM",
    "init_claude_code_client",
    "CopilotCLILLM",
    "init_copilot_cli_client",
    "LLMEnsemble",
]
