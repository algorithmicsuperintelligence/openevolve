"""
Model ensemble for LLMs
"""

import asyncio
import logging
import random
from typing import Dict, List, Optional, Tuple

from openevolve.llm.base import LLMInterface
from openevolve.llm.openai import OpenAILLM
from openevolve.config import LLMModelConfig

logger = logging.getLogger(__name__)


class LLMEnsemble:
    """Ensemble of LLMs"""

    def __init__(self, models_cfg: List[LLMModelConfig]):
        self.models_cfg = models_cfg

        # Initialize models from the configuration
        self.models = [
            model_cfg.init_client(model_cfg) if model_cfg.init_client else OpenAILLM(model_cfg)
            for model_cfg in models_cfg
        ]

        # Extract and normalize model weights
        self.weights = [model.weight for model in models_cfg]
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

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

    async def generate(self, prompt: str, **kwargs) -> Tuple[str, int]:
        """Generate text using a randomly selected model based on weights

        Returns:
            Tuple of (generated_text, model_id) where model_id is the index
            of the selected model in the ensemble
        """
        model, model_id = self._sample_model()
        response = await model.generate(prompt, **kwargs)
        return response, model_id

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> Tuple[str, int]:
        """Generate text using a system message and conversational context

        Returns:
            Tuple of (generated_text, model_id) where model_id is the index
            of the selected model in the ensemble
        """
        model, model_id = self._sample_model()
        response = await model.generate_with_context(system_message, messages, **kwargs)
        return response, model_id

    def _sample_model(self) -> Tuple[LLMInterface, int]:
        """Sample a model from the ensemble based on weights

        Returns:
            Tuple of (model, model_id) where model_id is the index of the
            selected model in the ensemble
        """
        index = self.random_state.choices(range(len(self.models)), weights=self.weights, k=1)[0]
        sampled_model = self.models[index]
        logger.info(f"Sampled model: {vars(sampled_model)['model']}")
        return sampled_model, index

    async def generate_multiple(self, prompt: str, n: int, **kwargs) -> List[Tuple[str, int]]:
        """Generate multiple texts in parallel

        Returns:
            List of (generated_text, model_id) tuples where model_id is the
            index of the selected model in the ensemble
        """
        tasks = [self.generate(prompt, **kwargs) for _ in range(n)]
        return await asyncio.gather(*tasks)

    async def parallel_generate(self, prompts: List[str], **kwargs) -> List[Tuple[str, int]]:
        """Generate responses for multiple prompts in parallel

        Returns:
            List of (generated_text, model_id) tuples where model_id is the
            index of the selected model in the ensemble
        """
        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def generate_all_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> List[str]:
        """Generate text using all available models and average their returned metrics

        Returns:
            List of generated texts, one per model in the ensemble (order matches
            self.models). The model_id for each response is its index in the list.
        """
        responses = []
        for model in self.models:
            responses.append(await model.generate_with_context(system_message, messages, **kwargs))
        return responses
