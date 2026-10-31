"""Explicit, lossy configuration export for the AlphaEvolve experiment contract."""

import math
import re
import warnings
from collections.abc import Mapping
from dataclasses import fields
from numbers import Real
from typing import Optional

from openevolve.config import Config, LLMConfig, LLMModelConfig


def _positive_integer(value, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _gemini_name(value, path: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"gemini-[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError(f"{path} must explicitly name a Gemini model")
    return value


def _warn_omitted_settings(config: Config) -> None:
    # Inspect fixed dataclass field names, never values or keys from arbitrary
    # dictionaries (e.g. evaluator_args). Do not serialize or deepcopy clients.
    omitted = []
    for field in fields(Config):
        if field.name in {"max_iterations", "language", "llm"}:
            continue
        value = getattr(config, field.name)
        if field.name in {"prompt", "database", "evaluator", "evolution_trace"}:
            omitted.extend(
                f"{field.name}.{child.name}"
                for child in fields(value)
                if getattr(value, child.name) is not None
            )
        elif value is not None:
            omitted.append(field.name)
    for field in fields(LLMConfig):
        # Legacy model aliases have already been expanded into llm.models.
        if field.name in {
            "models",
            "primary_model",
            "primary_model_weight",
            "secondary_model",
            "secondary_model_weight",
        }:
            continue
        if getattr(config.llm, field.name) is not None:
            omitted.append(f"llm.{field.name}")
    for index, model in enumerate(config.llm.models):
        omitted.extend(
            f"llm.models[{index}].{field.name}"
            for field in fields(LLMModelConfig)
            if field.name not in {"name", "weight"} and getattr(model, field.name) is not None
        )
    warnings.warn(
        "OpenEvolve settings not exported (including defaults): " + ", ".join(omitted),
        UserWarning,
        stacklevel=3,
    )


def config_to_alphaevolve(
    config: Config,
    *,
    title: str,
    problem_description: str,
    model_mapping: Optional[Mapping[str, str]] = None,
    generation_concurrency: Optional[int] = None,
) -> dict:
    """Export a Python experiment dictionary without files, credentials or requests.

    Copy max_iterations as the generation-attempt budget, language='python', and
    model names/relative weights. Require literal title/problem_description text;
    OpenEvolve prompt template names and per-island prompts are not resolved.

    Non-Gemini models require explicit model_mapping entries; no model is chosen
    automatically. Merge duplicate target names by summing weights and allow at
    most two distinct targets. Service availability, permissions and model-tier
    mixture restrictions still require validation by the caller/service.

    generation_concurrency, if supplied, sets the service's generation parallelism.
    OpenEvolve evaluation worker counts have a different meaning and are omitted.
    The generation budget is not an evaluation, token, time or monetary limit.

    Warn by field path for omitted settings, including defaults, without printing
    values. Local endpoints/keys, evaluator configuration, MAP-Elites, stopping,
    LLM sampling/runtime controls and prompt construction are not transferred.
    The input is not mutated. This is not a runnable bundle or a cloud client.
    """
    if not isinstance(config, Config) or not isinstance(config.llm, LLMConfig):
        raise TypeError("config must be an OpenEvolve Config with an LLMConfig")
    for path, value in (("title", title), ("problem_description", problem_description)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path} must be non-empty literal text")
    if config.language != "python":
        raise ValueError("Set config.language='python' explicitly for this export")
    run_settings = {"max_programs": _positive_integer(config.max_iterations, "max_iterations")}
    if generation_concurrency is not None:
        run_settings["concurrency"] = _positive_integer(
            generation_concurrency, "generation_concurrency"
        )

    if model_mapping is None:
        model_mapping = {}
    if not isinstance(model_mapping, Mapping):
        raise TypeError("model_mapping must be a mapping")
    for source, target in model_mapping.items():
        if not isinstance(source, str) or not source.strip():
            raise ValueError("model_mapping keys must be non-empty model names")
        _gemini_name(target, "model_mapping target")
    if not isinstance(config.llm.models, (list, tuple)) or not config.llm.models:
        raise ValueError("llm.models must be a non-empty list")

    weights_by_name = {}
    source_names = set()
    remapped = []
    for index, model in enumerate(config.llm.models):
        path = f"llm.models[{index}]"
        if not isinstance(model, LLMModelConfig):
            raise TypeError(f"{path} must be an LLMModelConfig")
        if not isinstance(model.name, str) or not model.name.strip():
            raise ValueError(f"{path}.name must be a non-empty model name")
        source_names.add(model.name)
        name = _gemini_name(model_mapping.get(model.name, model.name), f"{path}.name")
        weight = model.weight
        if isinstance(weight, bool) or not isinstance(weight, Real):
            raise ValueError(f"{path}.weight must be finite and positive")
        try:
            weight = float(weight)
        except OverflowError:
            raise ValueError(f"{path}.weight must be finite and positive") from None
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"{path}.weight must be finite and positive")
        weights_by_name.setdefault(name, []).append(weight)
        if name != model.name:
            remapped.append(f"{path}.name")
    if set(model_mapping) - source_names:
        raise ValueError("model_mapping contains names absent from llm.models")
    if len(weights_by_name) > 2:
        raise ValueError("AlphaEvolve supports at most two distinct generation models")
    try:
        models = [
            {"name": name, "weight": math.fsum(weights)}
            for name, weights in weights_by_name.items()
        ]
        total_weight = math.fsum(model["weight"] for model in models)
    except OverflowError:
        raise ValueError("model weights must have a finite sum") from None
    if not math.isfinite(total_weight):
        raise ValueError("model weights must have a finite sum")

    _warn_omitted_settings(config)
    if remapped:
        warnings.warn(
            "Model names explicitly remapped: " + ", ".join(remapped),
            UserWarning,
            stacklevel=2,
        )
    return {
        "title": title,
        "problem_description": problem_description,
        "program_language": "python",
        "run_settings": run_settings,
        "generation_settings": {"models": models},
    }
