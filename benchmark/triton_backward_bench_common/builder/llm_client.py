"""Minimal OpenAI-compatible client for Stage 1 builder synthesis."""

from __future__ import annotations

import os

import openai


REASONING_MODEL_PREFIXES = (
    "o1",
    "o1-",
    "o3",
    "o3-",
    "o4-",
    "gpt-5",
    "gpt-5-",
)


def generate_with_openai_compatible_api(
    *,
    prompt: str,
    system_message: str,
    model: str,
    api_base: str,
    api_key: str | None,
    max_tokens: int,
    temperature: float | None,
    timeout: int,
) -> str:
    """Call an OpenAI-compatible chat completion endpoint."""
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_api_key:
        raise ValueError("Missing API key. Set OPENAI_API_KEY or pass --api-key.")

    client = openai.OpenAI(api_key=resolved_api_key, base_url=api_base, timeout=timeout)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    model_lower = model.lower()
    if model_lower.startswith(REASONING_MODEL_PREFIXES):
        params = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
    else:
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    response = client.chat.completions.create(**params)
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("LLM response did not contain message content")
    return content
