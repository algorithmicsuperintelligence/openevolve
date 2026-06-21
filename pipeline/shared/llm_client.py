"""Minimal synchronous LLM client for synthesis agents."""

from __future__ import annotations

import os
import re
import time

import openai


def _parse_retry_after(error_msg: str) -> float | None:
    """Extract the suggested wait time from an OpenAI rate-limit error message.

    The API embeds text like 'try again in 876ms' or 'try again in 1.5s'.
    Returns seconds, or None if not found.
    """
    m = re.search(r"try again in (\d+(?:\.\d+)?)(ms|s)", str(error_msg))
    if not m:
        return None
    val, unit = float(m.group(1)), m.group(2)
    seconds = val / 1000.0 if unit == "ms" else val
    return seconds + 0.25  # small buffer on top of the suggested wait


def _uses_unsupported_max_tokens(exc: openai.APIStatusError) -> bool:
    """Return True when a model wants max_completion_tokens instead."""
    if exc.status_code != 400:
        return False
    msg = str(exc)
    return "max_tokens" in msg and "max_completion_tokens" in msg


def _uses_unsupported_temperature(exc: openai.APIStatusError) -> bool:
    """Return True when a model only supports default temperature."""
    if exc.status_code != 400:
        return False
    msg = str(exc)
    return "temperature" in msg and (
        "unsupported" in msg.lower() or "only the default" in msg.lower()
    )


def _apply_parameter_fallback(
    params: dict,
    exc: openai.APIStatusError,
    applied_fallbacks: set[str],
) -> bool:
    """Mutate request params for known model-specific 400s.

    Returns True when a fallback was applied and the caller should retry
    immediately without treating it as a transient API failure.
    """
    if "max_completion_tokens" not in applied_fallbacks and _uses_unsupported_max_tokens(exc):
        applied_fallbacks.add("max_completion_tokens")
        params["max_completion_tokens"] = params.pop("max_tokens")
        print("  [api parameter] retry with max_completion_tokens")
        return True

    if "temperature" not in applied_fallbacks and _uses_unsupported_temperature(exc):
        applied_fallbacks.add("temperature")
        params.pop("temperature", None)
        print("  [api parameter] retry with default temperature")
        return True

    return False


def _prefers_max_completion_tokens(model: str) -> bool:
    """Models in newer OpenAI families reject legacy max_tokens."""
    model_l = model.lower()
    return model_l.startswith(("gpt-5", "o1", "o3", "o4"))


def _requires_default_temperature(model: str) -> bool:
    """Some newer OpenAI models only accept the default temperature."""
    model_l = model.lower()
    return model_l.startswith(("gpt-5", "o1", "o3", "o4"))


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
    max_retries: int = 8,
    initial_delay: float = 2.0,
) -> str:
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY") or "none"
    client = openai.OpenAI(
        api_key=resolved_api_key,
        base_url=api_base,
        timeout=timeout,
    )
    params: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
    }
    if _prefers_max_completion_tokens(model):
        params["max_completion_tokens"] = max_tokens
    else:
        params["max_tokens"] = max_tokens
    if temperature is not None and not _requires_default_temperature(model):
        params["temperature"] = temperature

    delay = initial_delay
    applied_parameter_fallbacks: set[str] = set()
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(**params)
            return response.choices[0].message.content

        except openai.RateLimitError as exc:
            if attempt == max_retries:
                raise
            wait = _parse_retry_after(str(exc)) or delay
            print(f"  [rate limit] retry {attempt + 1}/{max_retries} in {wait:.1f}s")
            time.sleep(wait)
            delay = min(delay * 2, 60.0)

        except openai.APIStatusError as exc:
            if _apply_parameter_fallback(params, exc, applied_parameter_fallbacks):
                continue
            # Retry on transient 5xx errors; re-raise on 4xx (except 429 above).
            if attempt == max_retries or exc.status_code < 500:
                raise
            print(f"  [api error {exc.status_code}] retry {attempt + 1}/{max_retries} in {delay:.1f}s")
            time.sleep(delay)
            delay = min(delay * 2, 60.0)

        except openai.APIConnectionError:
            if attempt == max_retries:
                raise
            print(f"  [connection error] retry {attempt + 1}/{max_retries} in {delay:.1f}s")
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
