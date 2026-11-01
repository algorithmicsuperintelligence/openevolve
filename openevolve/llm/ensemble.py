"""
Model ensemble for LLMs
"""

import asyncio
import hashlib
import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple

from openevolve.llm.base import LLMInterface
from openevolve.llm.openai import OpenAILLM
from openevolve.config import LLMModelConfig

logger = logging.getLogger(__name__)

_PROVIDER_REGISTRY = {
    "openai": lambda cfg: OpenAILLM(cfg),
}

try:
    from openevolve.llm.claude_code import ClaudeCodeLLM

    _PROVIDER_REGISTRY["claude_code"] = lambda cfg: ClaudeCodeLLM(cfg)
except ImportError:
    pass

try:
    from openevolve.llm.copilot_cli import CopilotCLILLM

    _PROVIDER_REGISTRY["copilot_cli"] = lambda cfg: CopilotCLILLM(cfg)
except ImportError:
    pass

try:
    from openevolve.llm.codex_app_server import CodexAppServerLLM

    _PROVIDER_REGISTRY["codex_app_server"] = lambda cfg: CodexAppServerLLM(cfg)
except ImportError:
    pass


def _create_model(model_cfg: LLMModelConfig) -> LLMInterface:
    if model_cfg.init_client:
        return model_cfg.init_client(model_cfg)
    provider = getattr(model_cfg, "provider", None)
    if provider and provider in _PROVIDER_REGISTRY:
        return _PROVIDER_REGISTRY[provider](model_cfg)
    return OpenAILLM(model_cfg)


class LLMEnsemble:
    """Ensemble of LLMs"""

    def __init__(self, models_cfg: List[LLMModelConfig]):
        self.models_cfg = models_cfg

        # Validate weights before initializing clients, which may have side effects.
        self.weights = [model.weight for model in models_cfg]
        if any(not math.isfinite(weight) or weight < 0 for weight in self.weights):
            raise ValueError("LLM ensemble weights must be finite and non-negative")
        total = sum(self.weights)
        if self.weights and total <= 0:
            raise ValueError("LLM ensemble must contain at least one model with a positive weight")
        if not math.isfinite(total):
            # Finite weights can still overflow when added. Rescale only this
            # case so ordinary normalization and seeded sampling stay unchanged.
            scale = max(self.weights)
            self.weights = [weight / scale for weight in self.weights]
            total = sum(self.weights)
        self.weights = [weight / total for weight in self.weights]

        self.models = [_create_model(model_cfg) for model_cfg in models_cfg]

        # Set up random state for deterministic model selection
        self.random_state = random.Random()
        # Initialize with seed from first model's config if available
        if (
            models_cfg
            and hasattr(models_cfg[0], "random_seed")
            and models_cfg[0].random_seed is not None
        ):
            self.random_state.seed(models_cfg[0].random_seed)
            logger.debug(
                f"LLMEnsemble: Set random seed to {models_cfg[0].random_seed} for deterministic model selection"
            )

        # Only log if we have multiple models or this is the first ensemble
        if len(models_cfg) > 1 or not hasattr(logger, "_ensemble_logged"):
            logger.info(
                f"Initialized LLM ensemble with models: "
                + ", ".join(
                    f"{model.name} (weight: {weight:.2f})"
                    for model, weight in zip(models_cfg, self.weights)
                )
            )
            logger._ensemble_logged = True

        self.last_usage: Optional[Dict[str, Any]] = None
        # Set by reseed(); gives each iteration's requests their own seed
        self._iteration: Optional[int] = None

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using a randomly selected model based on weights"""
        index = self._sample_index()
        model = self.models[index]
        res = await model.generate(prompt, **self._request_kwargs(index, kwargs))
        self.last_usage = getattr(model, "last_usage", None)
        return res

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        index = self._sample_index()
        model = self.models[index]
        res = await model.generate_with_context(
            system_message, messages, **self._request_kwargs(index, kwargs)
        )
        self.last_usage = getattr(model, "last_usage", None)
        return res

    def reseed(self, iteration: int) -> None:
        """Seed model sampling from the configured seed and `iteration`, so the pick depends
        on the iteration rather than on how many picks this worker process has already made"""
        seed = getattr(self.models_cfg[0], "random_seed", None) if self.models_cfg else None
        if seed is not None:
            self.random_state.seed(f"{seed}:{iteration}")
        self._iteration = iteration

    def _request_kwargs(self, index: int, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """After reseed(iteration), send a seed derived from the model's seed and the
        iteration. A fixed seed would make every identical prompt (common early on, when
        iterations share a parent) return the identical response on providers that honour it"""
        if self._iteration is None or "seed" in kwargs:
            return kwargs
        seed = getattr(self.models_cfg[index], "random_seed", None)
        if seed is None:
            return kwargs
        digest = hashlib.sha256(f"{seed}:{self._iteration}:request".encode()).digest()
        return {**kwargs, "seed": int.from_bytes(digest[:4], "big") & 0x7FFFFFFF}

    def _sample_index(self) -> int:
        """Index of a model sampled from the ensemble based on weights"""
        index = self.random_state.choices(range(len(self.models)), weights=self.weights, k=1)[0]
        logger.info(f"Sampled model: {self.models_cfg[index].name}")
        return index

    def _sample_model(self) -> LLMInterface:
        """Sample a model from the ensemble based on weights"""
        return self.models[self._sample_index()]

    async def generate_multiple(self, prompt: str, n: int, **kwargs) -> List[str]:
        """Generate multiple texts in parallel"""
        tasks = [self.generate(prompt, **kwargs) for _ in range(n)]
        return await asyncio.gather(*tasks)

    async def parallel_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts in parallel"""
        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def generate_all_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> List[Optional[str]]:
        """Request feedback from positive-weight models.

        Keep a None slot for each disabled model so response indices still match
        the ensemble weights when the evaluator aggregates and logs feedback.
        """
        responses: List[Optional[str]] = []
        for model, weight in zip(self.models, self.weights):
            if weight <= 0:
                responses.append(None)
                continue
            responses.append(await model.generate_with_context(system_message, messages, **kwargs))
        return responses
