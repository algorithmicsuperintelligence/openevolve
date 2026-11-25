"""
Amazon Bedrock interface for LLMs.

Calls models on Amazon Bedrock through the Converse API with boto3, so any
Bedrock model that supports Converse can be used (Anthropic Claude, Amazon
Nova, Meta Llama, Mistral, DeepSeek, Qwen, ...). Authentication uses the
standard AWS credential chain (environment, shared config/profile, SSO,
or an IAM role on EC2 / ECS / Lambda) - no API key and no OpenAI-compatible
proxy are needed.

Usage in config.yaml (nothing hard-coded: model, region and credentials come
from your environment):
    llm:
      provider: "bedrock"
      aws_region: "${AWS_REGION}"          # optional; defaults to AWS_REGION / AWS_DEFAULT_REGION
      models:
        - name: "${BEDROCK_MODEL_ID}"      # any Converse model ID or inference profile ID/ARN
          weight: 1.0
          max_tokens: 16000

`name`, `aws_region` and `api_base` accept ${VAR} references. Credentials follow
the standard AWS chain, so a named profile works with AWS_PROFILE=<profile>.

Requires `boto3` (pip install boto3).
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from openevolve.config import _resolve_env_var
from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)


def _region(model_cfg) -> Optional[str]:
    return (
        _resolve_env_var(getattr(model_cfg, "aws_region", None))
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
    )


def _endpoint_url(model_cfg) -> Optional[str]:
    # `api_base` is shared from the top-level llm config and defaults to the
    # OpenAI URL, so only treat it as an endpoint override when it points at Bedrock
    # (e.g. a VPC endpoint or a FIPS endpoint).
    api_base = _resolve_env_var(getattr(model_cfg, "api_base", None))
    if isinstance(api_base, str) and "bedrock" in api_base:
        return api_base
    return None


class BedrockLLM(LLMInterface):
    """LLM interface for Amazon Bedrock (Converse API)."""

    def __init__(self, model_cfg=None, client: Any = None):
        self.model = _resolve_env_var(getattr(model_cfg, "name", None))
        if not self.model:
            raise ValueError(
                "Bedrock provider needs a model name (a Bedrock model ID or inference profile)"
            )
        self.system_message = getattr(model_cfg, "system_message", None)
        self.temperature = getattr(model_cfg, "temperature", None)
        self.top_p = getattr(model_cfg, "top_p", None)
        self.max_tokens = getattr(model_cfg, "max_tokens", None) or 4096
        self.timeout = getattr(model_cfg, "timeout", None) or 300
        self.retries = getattr(model_cfg, "retries", None)
        self.retries = 3 if self.retries is None else self.retries
        self.retry_delay = getattr(model_cfg, "retry_delay", None)
        self.retry_delay = 5 if self.retry_delay is None else self.retry_delay
        self.weight = getattr(model_cfg, "weight", 1.0)
        self.region = _region(model_cfg)
        self.endpoint_url = _endpoint_url(model_cfg)
        self.last_usage: Optional[Dict[str, Any]] = None
        # Sampling knobs this model rejected once (e.g. newer Claude models reject
        # `temperature`); they are dropped from later requests.
        self._unsupported: set = set()
        self._client = client or self._make_client()
        logger.info(
            f"Initialized BedrockLLM with model: {self.model} (region: {self.region or 'default'})"
        )

    def _make_client(self):
        try:
            import boto3
            from botocore.config import Config as BotoConfig
        except ImportError as e:  # pragma: no cover - exercised only without boto3
            raise ImportError("The bedrock provider needs boto3: pip install boto3") from e
        boto_cfg = BotoConfig(
            read_timeout=self.timeout,
            connect_timeout=min(60, self.timeout),
            retries={"max_attempts": 1, "mode": "standard"},  # retries are handled below
        )
        kwargs = {"config": boto_cfg}
        if self.region:
            kwargs["region_name"] = self.region
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
        return boto3.client("bedrock-runtime", **kwargs)

    async def generate(self, prompt: str, **kwargs) -> str:
        sys_msg = kwargs.pop("system_message", self.system_message) or ""
        return await self.generate_with_context(
            system_message=sys_msg,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    def _request(self, system_message: str, messages: List[Dict[str, str]], kwargs: Dict) -> Dict:
        converse_messages = []
        for m in messages:
            role = m.get("role", "user")
            if role == "system":
                # Converse takes the system prompt separately
                system_message = "\n\n".join(x for x in (system_message, m.get("content", "")) if x)
                continue
            role = "assistant" if role == "assistant" else "user"
            text = m.get("content", "") or ""
            # Converse requires alternating roles; merge consecutive turns of the same role
            if converse_messages and converse_messages[-1]["role"] == role:
                converse_messages[-1]["content"][0]["text"] += "\n\n" + text
            else:
                converse_messages.append({"role": role, "content": [{"text": text}]})

        inference = {"maxTokens": kwargs.get("max_tokens", self.max_tokens)}
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        # Only send the knobs that are set: several Bedrock models reject
        # temperature and top_p together, or reject temperature with extended thinking.
        if temperature is not None and "temperature" not in self._unsupported:
            inference["temperature"] = temperature
        if top_p is not None and "topP" not in self._unsupported:
            inference["topP"] = top_p

        request = {
            "modelId": self.model,
            "messages": converse_messages,
            "inferenceConfig": inference,
        }
        if system_message:
            request["system"] = [{"text": system_message}]
        return request

    def _converse(self, request: Dict) -> Dict:
        """Call Converse; if the model rejects a sampling knob, drop it and retry once per knob."""
        while True:
            try:
                return self._client.converse(**request)
            except Exception as e:
                message = str(e)
                if "ValidationException" not in message and "validation" not in message.lower():
                    raise
                inference = request.get("inferenceConfig", {})
                dropped = None
                for key, words in (("temperature", ("temperature",)), ("topP", ("top_p", "topp"))):
                    if key in inference and any(w in message.lower() for w in words):
                        dropped = key
                        break
                if dropped is None:
                    raise
                logger.warning(
                    f"Bedrock model {self.model} rejected `{dropped}`; retrying without it "
                    "(it will not be sent again for this model)."
                )
                self._unsupported.add(dropped)
                inference.pop(dropped, None)

    def _call(self, request: Dict) -> str:
        response = self._converse(request)
        stop_reason = response.get("stopReason")
        if stop_reason == "max_tokens":
            logger.warning(
                f"Response from {self.model} was cut off at the token limit "
                f"(stopReason=max_tokens, maxTokens={request['inferenceConfig'].get('maxTokens')}); "
                "raise llm.max_tokens."
            )
        usage = response.get("usage") or {}
        if usage:
            prompt_tokens = usage.get("inputTokens", 0) or 0
            completion_tokens = usage.get("outputTokens", 0) or 0
            self.last_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": usage.get("totalTokens") or (prompt_tokens + completion_tokens),
                "model": str(self.model),
                "finish_reason": stop_reason,
            }
            logger.info(
                f"LLM Token Usage ({self.model}): prompt_tokens={prompt_tokens}, "
                f"completion_tokens={completion_tokens}, total_tokens={self.last_usage['total_tokens']}"
            )
        else:
            self.last_usage = None
        blocks = response.get("output", {}).get("message", {}).get("content", []) or []
        # Keep only text blocks (skip reasoning / tool-use blocks)
        text = "".join(b["text"] for b in blocks if isinstance(b, dict) and "text" in b)
        if not text:
            raise RuntimeError(
                f"Empty response from Bedrock model {self.model} (stopReason={stop_reason})"
            )
        return text

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        request = self._request(system_message, messages, kwargs)
        timeout = kwargs.get("timeout", self.timeout)
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        loop = asyncio.get_event_loop()
        for attempt in range(retries + 1):
            try:
                return await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: self._call(request)), timeout=timeout + 30
                )
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(
                        f"Bedrock timeout on attempt {attempt + 1}/{retries + 1}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with timeout")
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Bedrock error on attempt {attempt + 1}/{retries + 1}: {e}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with error: {e}")
                    raise


def init_bedrock_client(model_cfg):
    """Factory function compatible with OpenEvolve's init_client config hook."""
    return BedrockLLM(model_cfg)
