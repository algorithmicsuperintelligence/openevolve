"""
OrcaRouter model catalog discovery and capability filtering.

The single source of truth for "which models can this account actually call" is
``GET {api_base}/models`` on the configured inference origin. The live response
is authoritative; a small, explicitly verified seed keeps a fresh installation
usable when the catalog endpoint is unreachable.

Every AI entry point filters the catalog for the capability it needs. A model
is never admitted on the strength of its name: the catalog metadata has to say
so, and unknown metadata fails closed.
"""

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from openevolve.llm.orcarouter_auth import (
    MAX_CATALOG_BYTES,
    MAX_CATALOG_ITEMS,
    MODELS_PATH,
    OrcaAuthError,
    redact,
    resolve_api_base,
)

logger = logging.getLogger(__name__)

#: Endpoint types the OpenAI-compatible text client can speak.
TEXT_ENDPOINT_TYPES = ("openai", "openai-response", "anthropic", "gemini")

#: Endpoint types that are text-generation-shaped.
NON_TEXT_ENDPOINT_TYPES = ("image-generation", "openai-video", "jina-rerank", "embeddings")

KNOWN_ENDPOINT_TYPES = TEXT_ENDPOINT_TYPES + NON_TEXT_ENDPOINT_TYPES

DEFAULT_TIMEOUT = 20
DEFAULT_CACHE_TTL = 300.0

CAPABILITY_CHAT = "chat"
CAPABILITY_EMBEDDING = "embedding"
CAPABILITY_IMAGE = "image"
CAPABILITY_VIDEO = "video"
CAPABILITY_RERANK = "rerank"

MODALITY_TEXT = "text"
MODALITY_IMAGE = "image"
MODALITY_AUDIO = "audio"
MODALITY_VIDEO = "video"

MODALITIES = (MODALITY_TEXT, MODALITY_IMAGE, MODALITY_AUDIO, MODALITY_VIDEO)

REASONING_EFFORT_LADDER = ("low", "medium", "high", "xhigh")

#: Verified cold-start catalog. Every entry was checked against the live
#: catalog and, where noted, carries context window, input modalities and the
#: reasoning-effort ladder. Used only when live discovery fails, and always
#: surfaced to the user as a degraded, explicitly-labelled list.
FALLBACK_CATALOG: Tuple["OrcaModel", ...] = ()


@dataclass(frozen=True)
class OrcaModel:
    """One catalog entry, in the provider's own naming."""

    id: str
    name: str = ""
    context_length: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    input_modalities: Tuple[str, ...] = ()
    output_modalities: Tuple[str, ...] = ()
    endpoint_types: Tuple[str, ...] = ()
    reasoning_efforts: Tuple[str, ...] = ()
    description: str = ""
    owned_by: str = ""
    #: True only for entries shipped in the verified fallback seed.
    verified: bool = False

    @property
    def supports_chat(self) -> bool:
        return any(t in TEXT_ENDPOINT_TYPES for t in self.endpoint_types)

    @property
    def supports_embedding(self) -> bool:
        return "embeddings" in self.endpoint_types

    @property
    def supports_image_generation(self) -> bool:
        return "image-generation" in self.endpoint_types

    @property
    def supports_video(self) -> bool:
        return "openai-video" in self.endpoint_types

    @property
    def supports_rerank(self) -> bool:
        return "jina-rerank" in self.endpoint_types

    @property
    def is_text_only(self) -> bool:
        return self.input_modalities in ((), (MODALITY_TEXT,))

    def accepts_modality(self, modality: str) -> bool:
        """Fail closed: a model with no declared modalities accepts text only."""
        if modality == MODALITY_TEXT:
            return True
        return modality in self.input_modalities

    def label(self) -> str:
        return self.name or self.id

    def to_option(self) -> Dict[str, Any]:
        """Minimal metadata for a UI model selector."""
        return {
            "id": self.id,
            "label": self.label(),
            "context_length": self.context_length,
            "input_modalities": list(self.input_modalities),
            "reasoning_efforts": list(self.reasoning_efforts),
            "verified": self.verified,
        }


def _verified_catalog() -> Tuple[OrcaModel, ...]:
    """The verified seed, built once."""
    return (
        OrcaModel(
            id="openai/gpt-5.5",
            name="OpenAI: GPT-5.5",
            context_length=400000,
            max_completion_tokens=128000,
            input_modalities=(MODALITY_TEXT,),
            output_modalities=(MODALITY_TEXT,),
            endpoint_types=TEXT_ENDPOINT_TYPES,
            reasoning_efforts=REASONING_EFFORT_LADDER,
            description="Verified fallback entry; reasoning effort low/medium/high/xhigh.",
            owned_by="openai",
            verified=True,
        ),
        OrcaModel(
            id="anthropic/claude-opus-4.8",
            name="Anthropic: Claude Opus 4.8",
            context_length=200000,
            max_completion_tokens=64000,
            input_modalities=(MODALITY_TEXT, MODALITY_IMAGE),
            output_modalities=(MODALITY_TEXT,),
            endpoint_types=("openai", "anthropic", "openai-response"),
            owned_by="anthropic",
            verified=True,
        ),
        OrcaModel(
            id="google/gemini-3.5-flash",
            name="Google: Gemini 3.5 Flash",
            context_length=1000000,
            max_completion_tokens=64000,
            input_modalities=(MODALITY_TEXT, MODALITY_IMAGE),
            output_modalities=(MODALITY_TEXT,),
            endpoint_types=TEXT_ENDPOINT_TYPES,
            owned_by="google",
            verified=True,
        ),
        OrcaModel(
            id="deepseek/deepseek-v4-pro",
            name="DeepSeek: DeepSeek V4 Pro",
            context_length=1048576,
            max_completion_tokens=384000,
            input_modalities=(MODALITY_TEXT,),
            output_modalities=(MODALITY_TEXT,),
            endpoint_types=("openai", "openai-response"),
            owned_by="deepseek",
            verified=True,
        ),
        OrcaModel(
            id="orcarouter/auto",
            name="OrcaRouter: Auto",
            input_modalities=(MODALITY_TEXT,),
            output_modalities=(MODALITY_TEXT,),
            endpoint_types=TEXT_ENDPOINT_TYPES,
            description="Verified fallback entry; router picks the backing model.",
            owned_by="orcarouter",
            verified=True,
        ),
    )


def fallback_catalog() -> List[OrcaModel]:
    """Return a copy of the verified cold-start catalog."""
    global FALLBACK_CATALOG
    if not FALLBACK_CATALOG:
        FALLBACK_CATALOG = _verified_catalog()
    return list(FALLBACK_CATALOG)


def _as_str_tuple(value: Any) -> Tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value if isinstance(v, (str, int)))
    return ()


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _parse_reasoning_efforts(raw: Dict[str, Any]) -> Tuple[str, ...]:
    """Read a declared reasoning-effort ladder, if the catalog publishes one.

    The public OrcaRouter catalog does not send reasoning metadata today, so
    this stays empty for live results and nothing is inferred from a model's
    name. A self-hosted catalog that does publish a ladder is honoured, bounded
    to the values this client knows how to send.
    """
    reasoning = raw.get("reasoning")
    candidates: Any = None
    if isinstance(reasoning, dict):
        candidates = reasoning.get("efforts") or reasoning.get("effort_levels")
    if candidates is None:
        candidates = raw.get("reasoning_efforts") or raw.get("supported_reasoning_efforts")
    return tuple(e for e in _as_str_tuple(candidates) if e in REASONING_EFFORT_LADDER)


def parse_model(raw: Dict[str, Any]) -> Optional[OrcaModel]:
    """Parse one catalog record, or return None when it is unusable.

    Bounding the accepted shape keeps a hostile or broken catalog response from
    advertising routes this client cannot speak.
    """
    if not isinstance(raw, dict):
        return None
    model_id = raw.get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        return None
    architecture = raw.get("architecture") if isinstance(raw.get("architecture"), dict) else {}
    endpoints = tuple(
        t for t in _as_str_tuple(raw.get("supported_endpoint_types")) if t in KNOWN_ENDPOINT_TYPES
    )
    if not endpoints:
        # A record that advertises only routes this client cannot speak is not
        # usable; do not guess a capability from its name.
        return None
    return OrcaModel(
        id=model_id.strip(),
        name=str(raw.get("name") or ""),
        context_length=_as_int(raw.get("context_length"))
        or _as_int(
            (raw.get("top_provider") or {}).get("context_length")
            if isinstance(raw.get("top_provider"), dict)
            else None
        ),
        max_completion_tokens=_as_int(raw.get("max_completion_tokens")),
        input_modalities=tuple(
            m for m in _as_str_tuple(architecture.get("input_modalities")) if m in MODALITIES
        ),
        output_modalities=tuple(
            m for m in _as_str_tuple(architecture.get("output_modalities")) if m in MODALITIES
        ),
        endpoint_types=endpoints,
        reasoning_efforts=_parse_reasoning_efforts(raw),
        description=str(raw.get("description") or "")[:2000],
        owned_by=str(raw.get("owned_by") or ""),
    )


def parse_catalog(payload: Any) -> List[OrcaModel]:
    """Parse a full catalog response into bounded, deduplicated models."""
    if isinstance(payload, dict):
        data = payload.get("data", [])
    elif isinstance(payload, list):
        data = payload
    else:
        return []
    if not isinstance(data, list):
        return []
    models: List[OrcaModel] = []
    seen = set()
    for raw in data[:MAX_CATALOG_ITEMS]:
        model = parse_model(raw)
        if model is None or model.id in seen:
            continue
        seen.add(model.id)
        models.append(model)
    return models


def filter_by_capability(models: Iterable[OrcaModel], capability: str) -> List[OrcaModel]:
    """Keep only the models this client can actually drive for a capability."""
    if capability == CAPABILITY_CHAT:
        return [m for m in models if m.supports_chat]
    if capability == CAPABILITY_EMBEDDING:
        return [m for m in models if m.supports_embedding]
    if capability == CAPABILITY_IMAGE:
        return [m for m in models if m.supports_image_generation]
    if capability == CAPABILITY_VIDEO:
        return [m for m in models if m.supports_video]
    if capability == CAPABILITY_RERANK:
        return [m for m in models if m.supports_rerank]
    raise ValueError(f"unknown capability: {capability}")


def filter_for_entry_point(
    models: Iterable[OrcaModel], capability: str, modality: str = MODALITY_TEXT
) -> List[OrcaModel]:
    """Select the models an entry point may offer.

    Chat models are always filtered for text first; a non-text modality is then
    required to be *declared* by ``architecture.input_modalities``. A model that
    declares nothing about modalities cannot serve an image/audio/video intake.
    """
    candidates = filter_by_capability(models, capability)
    if capability != CAPABILITY_CHAT or modality == MODALITY_TEXT:
        return candidates
    return [m for m in candidates if m.accepts_modality(modality)]


@dataclass(frozen=True)
class CatalogResult:
    """Outcome of a discovery attempt."""

    models: Tuple[OrcaModel, ...]
    source: str  # "live" | "cache" | "seed"
    api_base: str
    degraded: bool = False
    error: Optional[str] = None
    fetched_at: Optional[float] = None

    def options(self, capability: str = CAPABILITY_CHAT, modality: str = MODALITY_TEXT):
        return [m.to_option() for m in filter_for_entry_point(self.models, capability, modality)]

    def ids(self, capability: str = CAPABILITY_CHAT, modality: str = MODALITY_TEXT):
        return [m.id for m in filter_for_entry_point(self.models, capability, modality)]

    def status(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "degraded": self.degraded,
            "error": self.error,
            "api_base": self.api_base,
            "model_count": len(self.models),
            "fetched_at": self.fetched_at,
        }


class OrcaCatalogClient:
    """Fetches and caches the OrcaRouter catalog for one API base + key."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        cache_ttl: float = DEFAULT_CACHE_TTL,
        opener: Optional[Callable[[urllib.request.Request, int], Any]] = None,
    ):
        self.api_key = api_key
        self.api_base = api_base or resolve_api_base()
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self._opener = opener
        self._cache: Dict[str, Tuple[float, List[OrcaModel]]] = {}

    # -- transport ---------------------------------------------------------

    def _open(self, request: urllib.request.Request, timeout: int):
        if self._opener is not None:
            return self._opener(request, timeout)
        return urllib.request.urlopen(request, timeout=timeout)

    def _request(self, capability: Optional[str] = None) -> List[OrcaModel]:
        url = f"{self.api_base}{MODELS_PATH}"
        if capability:
            url = f"{url}?{urllib.parse.urlencode({'capability': capability})}"
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(url, headers=headers, method="GET")
        with self._open(request, self.timeout) as response:
            raw = response.read(MAX_CATALOG_BYTES + 1)
        if len(raw) > MAX_CATALOG_BYTES:
            raise OrcaAuthError("OrcaRouter catalog response exceeded the size limit")
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except (ValueError, UnicodeDecodeError):
            raise OrcaAuthError("OrcaRouter catalog response was not valid JSON") from None
        return parse_catalog(payload)

    # -- discovery ---------------------------------------------------------

    def discover(
        self,
        capability: Optional[str] = None,
        allow_seed: bool = True,
        use_cache: bool = True,
    ) -> CatalogResult:
        """Return the authoritative catalog, or a bounded fallback."""
        cache_key = capability or "*"
        now = time.monotonic()
        cached = self._cache.get(cache_key)
        if use_cache and cached and (now - cached[0]) < self.cache_ttl:
            return CatalogResult(
                models=tuple(cached[1]),
                source="cache",
                api_base=self.api_base,
                fetched_at=cached[0],
            )

        try:
            models = (
                filter_by_capability(self._request(capability), capability)
                if capability
                else self._request(None)
            )
        except OrcaAuthError as exc:
            return self._degraded(cache_key, str(exc), allow_seed)
        except urllib.error.HTTPError as exc:
            return self._degraded(
                cache_key, f"HTTP {exc.code}", allow_seed, auth_error=exc.code in (401, 403)
            )
        except urllib.error.URLError as exc:
            return self._degraded(
                cache_key, f"network error ({exc.reason.__class__.__name__})", allow_seed
            )
        except (TimeoutError, OSError) as exc:
            return self._degraded(cache_key, f"{exc.__class__.__name__}", allow_seed)

        if not models:
            return self._degraded(cache_key, "catalog returned no usable models", allow_seed)

        self._cache[cache_key] = (now, models)
        return CatalogResult(
            models=tuple(models), source="live", api_base=self.api_base, fetched_at=now
        )

    def _degraded(
        self,
        cache_key: str,
        error: str,
        allow_seed: bool,
        auth_error: bool = False,
    ) -> CatalogResult:
        if auth_error:
            error = f"{error} (credential rejected; sign in again or paste a new key)"
        logger.warning("OrcaRouter catalog discovery failed: %s", error)
        last_good = self._cache.get(cache_key)
        if last_good:
            return CatalogResult(
                models=tuple(last_good[1]),
                source="cache",
                api_base=self.api_base,
                degraded=True,
                error=error,
                fetched_at=last_good[0],
            )
        if not allow_seed:
            return CatalogResult(
                models=(), source="seed", api_base=self.api_base, degraded=True, error=error
            )
        return CatalogResult(
            models=tuple(fallback_catalog()),
            source="seed",
            api_base=self.api_base,
            degraded=True,
            error=error,
        )

    def invalidate(self) -> None:
        self._cache.clear()


def make_model_configs(
    result: CatalogResult,
    capability: str = CAPABILITY_CHAT,
    modality: str = MODALITY_TEXT,
    weight: float = 1.0,
) -> List[Any]:
    """Build ``LLMModelConfig`` entries from a catalog result.

    This is the seam the CLI uses: the selectable models come from the catalog
    and its capability filter, not from a free-form string.
    """
    from openevolve.config import LLMModelConfig

    configs = []
    for model in filter_for_entry_point(result.models, capability, modality):
        cfg = LLMModelConfig(name=model.id, weight=weight)
        cfg.provider = "orcarouter"
        if model.reasoning_efforts:
            cfg.reasoning_effort = "medium"
        configs.append(cfg)
    return configs
