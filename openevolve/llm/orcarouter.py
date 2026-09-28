"""
OrcaRouter provider adapter for OpenEvolve.

OrcaRouter is an OpenAI-compatible AI gateway that routes many providers behind
one endpoint (``https://api.orcarouter.ai/v1``). This module registers it as a
first-class named provider with two explicit credential choices:

``orcarouter``
    An existing ``sk-orca-...`` API key (config, CLI flag, or the
    ``ORCAROUTER_API_KEY`` environment variable).

``orcarouter_oauth``
    Browser sign-in with an OrcaRouter account (OAuth 2.0 + PKCE); the exchange
    mints an ordinary API key that belongs to the user.

Both end up at the same credential seam, so the request path and model
discovery below never need to know which one was used.
"""

import asyncio
import logging
import multiprocessing
import os
from typing import Any, Dict, List, Optional

import openai

from openevolve.llm.base import LLMInterface
from openevolve.llm.openai import OpenAILLM
from openevolve.llm.orcarouter_auth import (
    KEY_ENV,
    KEY_DASHBOARD_URL,
    OrcaAuthError,
    OrcaCredential,
    OrcaCredentialStore,
    acquire_credential,
    looks_like_orcarouter_key,
    mask_key,
    resolve_api_base,
)
from openevolve.llm.orcarouter_catalog import (
    CAPABILITY_CHAT,
    CAPABILITY_EMBEDDING,
    MODALITY_TEXT,
    CatalogResult,
    OrcaCatalogClient,
)

logger = logging.getLogger(__name__)

#: The two first-class provider ids. ``PROVIDER_ID_KEY`` names the entry that
#: uses a pasted API key; ``PROVIDER_ID_OAUTH`` names the PKCE sign-in entry.
#: Neither constant holds a credential.
PROVIDER_ID_KEY = "orcarouter"
PROVIDER_ID_OAUTH = "orcarouter_oauth"
ORCAROUTER_PROVIDERS = (PROVIDER_ID_KEY, PROVIDER_ID_OAUTH)

#: ``LLMConfig.api_base`` defaults to OpenAI's endpoint and is propagated to
#: every model config. Seeing that exact stock value on an OrcaRouter provider
#: means "not configured", so the OrcaRouter base (env override or the public
#: default) applies instead.
OPENAI_STOCK_BASE = "https://api.openai.com/v1"


class OrcaReauthRequired(OrcaAuthError):
    """Terminal: the credential was rejected and no refresh grant exists.

    The exact account/generation that made the rejected request is marked for
    re-authentication and is not retried.
    """


def effective_api_base(configured: Optional[str]) -> str:
    """Resolve the inference base for an OrcaRouter provider."""
    if configured:
        candidate = configured.strip().rstrip("/")
        if candidate and candidate != OPENAI_STOCK_BASE:
            return candidate
    return resolve_api_base()


def is_worker_process() -> bool:
    """True inside a spawned worker, where prompting the user is impossible."""
    try:
        return multiprocessing.current_process().name != "MainProcess"
    except Exception:  # pragma: no cover - defensive
        return False


class OrcaRouterLLM(OpenAILLM):
    """OpenAI-compatible client bound to the OrcaRouter gateway."""

    provider_id = PROVIDER_ID_KEY

    def __init__(
        self,
        model_cfg: Optional[dict] = None,
        provider_id: Optional[str] = None,
        credential: Optional[OrcaCredential] = None,
        store: Optional[OrcaCredentialStore] = None,
        oob: bool = False,
    ):
        self.provider_id = provider_id or getattr(model_cfg, "provider", None) or PROVIDER_ID_KEY
        self.store = store or OrcaCredentialStore()
        if oob is False:
            oob = bool(getattr(model_cfg, "orcarouter_oob", False))
        self.credential = credential or self._resolve_credential(model_cfg, oob=oob)
        self.api_base = effective_api_base(getattr(model_cfg, "api_base", None))
        super().__init__(self._with_credentials(model_cfg))
        self.catalog = OrcaCatalogClient(api_key=self.credential.api_key, api_base=self.api_base)

    def _is_interactive(self) -> bool:
        return not is_worker_process()

    def _resolve_credential(self, model_cfg: Optional[dict], oob: bool = False) -> OrcaCredential:
        configured = getattr(model_cfg, "api_key", None)
        api_key = configured if looks_like_orcarouter_key(configured) else None

        if self.provider_id != PROVIDER_ID_OAUTH:
            # API-key adapter: config value or ORCAROUTER_API_KEY. Never opens a
            # browser, so it is safe in every process.
            return acquire_credential(PROVIDER_ID_KEY, api_key=api_key, store=self.store)

        stored = self.store.load()
        if stored is not None and not stored.needs_reauth:
            # Reuse the durable key instead of minting another one: OrcaRouter
            # caps PKCE-issued keys at 10 per user per 24 hours.
            return stored
        if not self._is_interactive():
            raise OrcaAuthError(self._reauth_hint())
        return acquire_credential(PROVIDER_ID_OAUTH, store=self.store, oob=oob)

    @staticmethod
    def _reauth_hint() -> str:
        return (
            "No usable OrcaRouter account credential is available and this process "
            "cannot open a browser. Run `openevolve-run.py connect orcarouter` (or set "
            f"{KEY_ENV}) once, then start the run again."
        )

    def _with_credentials(self, model_cfg):
        """Hand the base OpenAILLM the OrcaRouter key and base URL.

        The credential is injected as the client's key, which is what the
        OpenAI SDK turns into ``Authorization: Bearer <key>``.
        """
        from dataclasses import replace

        if model_cfg is None:
            return model_cfg
        try:
            return replace(
                model_cfg,
                api_key=self.credential.api_key,
                api_base=self.api_base,
            )
        except TypeError:  # pragma: no cover - duck-typed config
            model_cfg.api_key = self.credential.api_key
            model_cfg.api_base = self.api_base
            return model_cfg

    # -- status + lifecycle ------------------------------------------------

    @property
    def needs_reauth(self) -> bool:
        return self.credential.needs_reauth

    def masked_key(self) -> str:
        return mask_key(self.credential.api_key)

    def status(self) -> Dict[str, Any]:
        """Redacted status for the CLI and the settings UI."""
        return {
            "provider": self.provider_id,
            "authenticated": bool(self.credential) and not self.credential.needs_reauth,
            "auth_source": self.credential.source,
            "account_id": self.credential.account_id,
            "generation": self.credential.generation,
            "scope": self.credential.scope,
            "secret_masked": self.masked_key(),
            "api_base": self.api_base,
            "needs_reauth": self.credential.needs_reauth,
            "key_dashboard_url": KEY_DASHBOARD_URL,
        }

    def model_options(
        self,
        capability: str = CAPABILITY_CHAT,
        modality: str = MODALITY_TEXT,
        refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        """Capability-filtered model options from the live catalog."""
        if refresh:
            self.catalog.invalidate()
        return self.discover(refresh=refresh).options(capability, modality)

    def discover(self, capability: Optional[str] = None, refresh: bool = False) -> CatalogResult:
        if refresh:
            self.catalog.invalidate()
        return self.catalog.discover(capability=capability)

    def embedding_catalog(self, refresh: bool = False) -> CatalogResult:
        return self.discover(capability=CAPABILITY_EMBEDDING, refresh=refresh)

    # -- request path ------------------------------------------------------

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Retry like the OpenAI client, but treat a rejected key as terminal.

        A revoked OrcaRouter key cannot be refreshed -- there is no refresh
        grant -- so it must not be retried or silently rotated.
        """
        retries = kwargs.pop("retries", self.retries) or 0
        retry_delay = kwargs.pop("retry_delay", self.retry_delay) or 0
        for attempt in range(retries + 1):
            try:
                return await super().generate_with_context(
                    system_message, messages, retries=0, **kwargs
                )
            except OrcaReauthRequired:
                raise
            except Exception as exc:
                if attempt < retries:
                    logger.warning(
                        "OrcaRouter request failed on attempt %s/%s: %s. Retrying...",
                        attempt + 1,
                        retries + 1,
                        redact(str(exc)),
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    raise
        raise AssertionError("unreachable")  # pragma: no cover

    async def _call_api(self, params: Dict[str, Any]) -> str:
        try:
            return await super()._call_api(params)
        except openai.AuthenticationError as exc:
            raise self._reject_credential(exc) from None
        except openai.PermissionDeniedError as exc:
            raise OrcaAuthError(
                "OrcaRouter refused the request: this key is not permitted to use "
                f"'{self.model}'. Pick another model from the live catalog."
            ) from None
        except openai.APIStatusError as exc:
            if getattr(exc, "status_code", None) == 401:
                raise self._reject_credential(exc) from None
            raise

    def _reject_credential(self, exc: Exception) -> OrcaReauthRequired:
        """Mark exactly this credential generation for re-authentication."""
        self.store.mark_needs_reauth(self.credential.generation)
        return OrcaReauthRequired(
            "OrcaRouter rejected this credential (401). It may have been revoked at "
            f"{KEY_DASHBOARD_URL}. Sign in again or paste a new key; the stored key is "
            "kept until a new one succeeds."
        )


def redact(text: str) -> str:
    from openevolve.llm.orcarouter_auth import redact as _redact

    return _redact(text)


def init_orcarouter_client(model_cfg):
    """Factory compatible with OpenEvolve's ``init_client`` config hook."""
    return OrcaRouterLLM(model_cfg, provider_id=PROVIDER_ID_KEY)


def init_orcarouter_oauth_client(model_cfg):
    """Factory for the PKCE (account sign-in) entry."""
    return OrcaRouterLLM(model_cfg, provider_id=PROVIDER_ID_OAUTH)


def orcarouter_credential_status(
    provider_id: str = PROVIDER_ID_KEY, store: Optional[OrcaCredentialStore] = None
) -> Dict[str, Any]:
    """Redacted credential status without constructing an LLM client."""
    store = store or OrcaCredentialStore()
    credential = store.load()
    if credential is None:
        return {
            "provider": provider_id,
            "authenticated": False,
            "auth_source": None,
            "account_id": None,
            "generation": None,
            "scope": None,
            "secret_masked": "",
            "api_base": resolve_api_base(),
            "needs_reauth": False,
            "key_dashboard_url": KEY_DASHBOARD_URL,
        }
    return {
        "provider": provider_id,
        "authenticated": not credential.needs_reauth,
        "auth_source": credential.source,
        "account_id": credential.account_id,
        "generation": credential.generation,
        "scope": credential.scope,
        "secret_masked": credential.masked,
        "api_base": resolve_api_base(),
        "needs_reauth": credential.needs_reauth,
        "key_dashboard_url": KEY_DASHBOARD_URL,
    }


def resolve_credential_for_discovery(
    store: Optional[OrcaCredentialStore] = None,
) -> Optional[OrcaCredential]:
    """Return the stored credential a discovery call should authenticate with.

    Discovery is read-only and never opens a browser, so it reuses whatever the
    user already stored instead of triggering a new sign-in. A credential that
    was flagged ``needs_reauth`` is deliberately not returned: putting a rejected
    key back on the wire would produce a 401 the UI cannot act on.
    """
    credential = (store or OrcaCredentialStore()).load()
    if credential is None or credential.needs_reauth:
        return None
    return credential


def logout(store: Optional[OrcaCredentialStore] = None) -> bool:
    """Forget the stored OrcaRouter credential. Returns True if one existed."""
    return (store or OrcaCredentialStore()).clear()


__all__ = [
    "PROVIDER_ID_KEY",
    "PROVIDER_ID_OAUTH",
    "ORCAROUTER_PROVIDERS",
    "OrcaRouterLLM",
    "OrcaReauthRequired",
    "init_orcarouter_client",
    "init_orcarouter_oauth_client",
    "orcarouter_credential_status",
    "resolve_credential_for_discovery",
    "effective_api_base",
    "logout",
]
