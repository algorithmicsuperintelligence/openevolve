"""Opt-in .env configuration import for local Python AlphaEvolve examples."""

import math
import re
import warnings
from pathlib import Path
from typing import Optional, Union

from openevolve.compat.alphaevolve import _warn_unmapped_fields, config_from_alphaevolve
from openevolve.config import Config


def _read_env(path: Union[str, Path]) -> dict:
    try:
        from dotenv.parser import parse_stream
    except ImportError:
        raise ImportError(
            'AlphaEvolve .env import requires: pip install "openevolve[alphaevolve-env]"'
        ) from None

    values = {}
    # Use dotenv's parser bindings so malformed lines and duplicate assignments
    # can fail explicitly, without interpolation, global environment changes,
    # implicit file discovery, or logging source lines that may contain secrets.
    with Path(path).open(encoding="utf-8-sig") as stream:
        for binding in parse_stream(stream):
            if binding.error:
                raise ValueError(f"Invalid .env syntax near line {binding.original.line}")
            key = binding.key
            if key is None:
                continue
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"Invalid .env variable name near line {binding.original.line}")
            if key in values:
                raise ValueError(f"Duplicate .env assignment near line {binding.original.line}")
            values[key] = binding.value
    return values


def _positive_integer(values: dict, key: str) -> int:
    value = values.get(key)
    if value is None or not re.fullmatch(r"[0-9]+", value) or int(value) < 1:
        raise ValueError(f"{key} must be an explicit positive integer")
    return int(value)


def config_from_alphaevolve_env(
    env_path: Union[str, Path],
    *,
    problem_description: str,
    api_base: str,
    api_key: Optional[str] = None,
) -> Config:
    """Import explicit .env model/budget settings using the dictionary converter.

    Requires the optional alphaevolve-env extra. Reads only env_path; does not
    search for other files, interpolate variables, modify os.environ, execute
    shell/script code, or construct clients. The caller provides the problem text
    (e.g. instructions.md) and OE endpoint/key. Cloud credentials are not copied.

    Accept either MODEL with optional MODEL_WEIGHT, or MODEL_1..N with optional
    MODEL_i_WEIGHT (positive indices, numeric order, gaps allowed). Mixing the
    forms, missing model names, malformed/duplicate assignments, and variable or
    command references in model names are rejected. Missing weights become 1.

    MAX_PROGRAMS_GENERATED is required and maps to OE generation attempts. A
    present MAX_PROGRAMS_EVALUATED must be equal, otherwise import fails rather
    than discarding a different limit. Even equal limits warn: OE also evaluates
    the seed and may retry, so this is not a hard evaluation/request/cost cap.
    Other variables, including concurrency and GCP settings, warn by name only.
    """
    values = _read_env(env_path)
    indices = set()
    for key in values:
        match = re.fullmatch(r"MODEL_([0-9]+)(?:_WEIGHT)?", key)
        if match:
            index = match.group(1)
            if index.startswith("0"):
                raise ValueError("MODEL indices must be positive integers without leading zeros")
            indices.add(int(index))
    if indices and ("MODEL" in values or "MODEL_WEIGHT" in values):
        raise ValueError("Use MODEL or numbered MODEL_i variables, not both")
    model_keys = [f"MODEL_{i}" for i in sorted(indices)] if indices else ["MODEL"]
    supported = {"MAX_PROGRAMS_GENERATED", "MAX_PROGRAMS_EVALUATED"}
    models = []
    for key in model_keys:
        name = values.get(key)
        if not name or not name.strip() or "$" in name or "`" in name:
            raise ValueError(f"{key} must contain a literal, non-empty model name")
        weight_key = f"{key}_WEIGHT"
        weight = 1.0
        if weight_key in values:
            try:
                weight = float(values[weight_key])
            except (TypeError, ValueError, OverflowError):
                raise ValueError(f"{weight_key} must be finite and positive") from None
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"{weight_key} must be finite and positive")
        models.append({"name": name, "weight": weight})
        supported.update({key, weight_key})

    iterations = _positive_integer(values, "MAX_PROGRAMS_GENERATED")
    if "MAX_PROGRAMS_EVALUATED" in values:
        evaluated = _positive_integer(values, "MAX_PROGRAMS_EVALUATED")
        if evaluated != iterations:
            raise ValueError(
                "MAX_PROGRAMS_EVALUATED differs from MAX_PROGRAMS_GENERATED; "
                "separate generation/evaluation limits cannot be enforced by this import"
            )

    config = config_from_alphaevolve(
        {
            "program_language": "python",
            "problem_description": problem_description,
            "run_settings": {"max_programs": iterations},
            "generation_settings": {"models": models},
        },
        api_base=api_base,
        api_key=api_key,
    )
    if "MAX_PROGRAMS_EVALUATED" in values:
        warnings.warn(
            "MAX_PROGRAMS_EVALUATED is not enforced as a hard cap: "
            "MAX_PROGRAMS_GENERATED maps to OE generation attempts, "
            "plus a seed evaluation and possible retries",
            UserWarning,
            stacklevel=2,
        )
    _warn_unmapped_fields(values, supported)
    return config
