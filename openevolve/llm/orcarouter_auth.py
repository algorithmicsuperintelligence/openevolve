"""
OrcaRouter credential acquisition.

Two explicit ways to obtain the *same* credential are provided behind one small
interface (`CredentialProvider`):

``ApiKeyCredentialProvider``
    The user pastes an existing ``sk-orca-...`` key (or exports it through the
    project's usual environment/secret mechanism).

``PkceCredentialProvider``
    The user signs in with their OrcaRouter account through OAuth 2.0 + PKCE.
    The browser flow mints an ordinary API key belonging to that user.

Both adapters return an :class:`OrcaCredential`. Everything downstream (the
provider adapter, model discovery, every AI entry point) consumes only that
type, so nothing below this seam knows which adapter produced the credential.

The PKCE exchange returns a *durable API key*, not a refresh token. There is no
refresh grant: a revoked key means re-authentication.
"""

import base64
import hashlib
import hmac
import http.server
import json
import logging
import os
import re
import secrets
import socket
import stat
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Origins
#
# Authentication and inference live on *different* public origins. Neither is
# ever derived from the other by rewriting a hostname or appending "/v1" -- a
# single shared base is only honoured when the operator sets ORCA_BASE_URL.
# ---------------------------------------------------------------------------

DEFAULT_AUTH_BASE = "https://www.orcarouter.ai"
DEFAULT_API_BASE = "https://api.orcarouter.ai/v1"

AUTHORIZE_PATH = "/auth"
EXCHANGE_PATH = "/api/v1/auth/keys"
DEVICE_CODE_PATH = "/api/v1/auth/device/code"
DEVICE_POLL_PATH = "/api/v1/auth/device/token"
MODELS_PATH = "/models"

KEY_DASHBOARD_URL = "https://www.orcarouter.ai/console/authorized-apps"

#: Environment variables are read by *name*; these constants hold the name,
#: never a credential value.
KEY_ENV = "ORCAROUTER_API_KEY"
AUTH_BASE_ENV = "ORCA_AUTH_BASE_URL"
API_BASE_ENV = "ORCA_API_BASE_URL"
SHARED_BASE_ENV = "ORCA_BASE_URL"
SECRETS_FILE_ENV = "OPENEVOLVE_SECRETS_FILE"

DEFAULT_SECRETS_FILE = "secrets.yaml"
SECRETS_NAMESPACE = "orcarouter"

KEY_PREFIX = "sk-orca-"
KEY_PATTERN = re.compile(r"^sk-orca-[A-Za-z0-9_\-]{8,}$")

#: Smallest scope that satisfies OrcaRouter inference. A granted scope below
#: this means the workspace role did not permit the request.
MINIMUM_SCOPE = "api"

DEFAULT_TIMEOUT = 30
DEFAULT_CONNECT_TIMEOUT = 300
MAX_CATALOG_BYTES = 4 * 1024 * 1024
MAX_CATALOG_ITEMS = 5000


class OrcaAuthError(RuntimeError):
    """Authentication failure with a message that is safe to show a user.

    Messages never embed a verifier, an authorization code, or a key.
    """


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _normalize_base(url: str, *, allow_loopback_http: bool = True) -> str:
    """Validate an origin/base URL and strip a trailing slash.

    Remote origins must be HTTPS. Plain HTTP is accepted only for loopback
    development hosts.
    """
    if not url:
        raise OrcaAuthError("empty OrcaRouter base URL")
    parsed = urllib.parse.urlsplit(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise OrcaAuthError("OrcaRouter base URL must use http or https")
    host = (parsed.hostname or "").lower()
    if not host:
        raise OrcaAuthError("OrcaRouter base URL has no host")
    if parsed.scheme == "http":
        loopback = host in ("localhost", "127.0.0.1", "::1") or host.endswith(".localhost")
        if not (loopback and allow_loopback_http):
            raise OrcaAuthError("refusing plain HTTP for non-loopback OrcaRouter origin; use https")
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise OrcaAuthError("OrcaRouter base URL must not carry userinfo, query or fragment")
    return url.strip().rstrip("/")


def resolve_auth_base(env: Optional[Dict[str, str]] = None) -> str:
    """Resolve the authentication origin.

    Precedence: explicit ``ORCA_AUTH_BASE_URL``, then a shared
    ``ORCA_BASE_URL`` self-hosted deployment, then the public default.
    """
    env = os.environ if env is None else env
    raw = env.get(AUTH_BASE_ENV) or env.get(SHARED_BASE_ENV) or DEFAULT_AUTH_BASE
    return _normalize_base(raw)


def resolve_api_base(env: Optional[Dict[str, str]] = None) -> str:
    """Resolve the inference base URL.

    Precedence: explicit ``ORCA_API_BASE_URL``, then ``ORCA_BASE_URL`` (with the
    OpenAI-compatible ``/v1`` path appended for a shared self-hosted origin),
    then the public default.
    """
    env = os.environ if env is None else env
    explicit = env.get(API_BASE_ENV)
    if explicit:
        return _normalize_base(explicit)
    shared = env.get(SHARED_BASE_ENV)
    if shared:
        base = _normalize_base(shared)
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return base
    return DEFAULT_API_BASE


# ---------------------------------------------------------------------------
# Credential value + storage
# ---------------------------------------------------------------------------


def mask_key(key: Optional[str]) -> str:
    """Render a key for display. Never returns the full key.

    Only the fixed format prefix and the last few characters are shown, which
    is enough for a user to tell two keys apart without exposing material that
    could be used to authenticate.
    """
    if not key:
        return ""
    if len(key) <= len(KEY_PREFIX) + 4:
        return f"{KEY_PREFIX}…"
    return f"{KEY_PREFIX}…{key[-4:]}"


def looks_like_orcarouter_key(key: Optional[str]) -> bool:
    """Cheap format check only -- never proof that a credential is valid."""
    return bool(key) and bool(KEY_PATTERN.match(key.strip()))


@dataclass(frozen=True)
class OrcaCredential:
    """A usable OrcaRouter credential plus its provenance.

    ``generation`` increments every time a credential is written for an
    account. It lets a late failure be attributed to the exact credential that
    made the rejected request instead of a newer replacement.
    """

    api_key: str
    source: str
    key_id: str = ""
    account_id: Optional[str] = None
    generation: int = 1
    scope: Optional[str] = None
    needs_reauth: bool = False
    updated_at: Optional[float] = None

    @property
    def masked(self) -> str:
        return mask_key(self.api_key)

    def to_record(self) -> Dict[str, Any]:
        return {
            "api_key": self.api_key,
            "source": self.source,
            "key_id": self.key_id,
            "account_id": self.account_id,
            "generation": self.generation,
            "scope": self.scope,
            "needs_reauth": self.needs_reauth,
            "updated_at": self.updated_at if self.updated_at is not None else time.time(),
        }


def redact(text: str) -> str:
    """Redact anything that looks like an OrcaRouter credential."""
    if not text:
        return text
    return re.sub(r"sk-orca-[A-Za-z0-9_\-]+", "sk-orca-<REDACTED>", text)


def resolve_secrets_path(path: Optional[str] = None) -> Path:
    """Resolve the secrets file location.

    An explicit path wins, then ``OPENEVOLVE_SECRETS_FILE``, then the project
    default. The environment-provided path is normalized (expanduser +
    normpath) so a relative or ``~``-prefixed value still lands in a single,
    well-defined location.
    """
    if path:
        return Path(path).expanduser()
    env_path = os.environ.get(SECRETS_FILE_ENV)
    if env_path:
        return Path(os.path.normpath(os.path.expanduser(env_path)))
    return Path(DEFAULT_SECRETS_FILE).expanduser()


class OrcaCredentialStore:
    """Persist the OrcaRouter credential in the project's secrets file.

    The repository already establishes ``secrets.yaml`` (gitignored) as its
    secret-holding file. This store reuses it under a dedicated namespace and
    writes it atomically with owner-only permissions; it does not introduce a
    second credential database.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = resolve_secrets_path(path)

    # -- low level ---------------------------------------------------------

    def _read_all(self) -> Dict[str, Any]:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError as exc:  # unreadable file must not crash a run
            logger.warning(f"Could not read secrets file: {exc.__class__.__name__}")
            return {}
        if not raw.strip():
            return {}
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError:
            logger.warning("Secrets file is not valid YAML; ignoring its contents")
            return {}
        return data if isinstance(data, dict) else {}

    def _write_all(self, data: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.tmp")
        payload = yaml.safe_dump(data, default_flow_style=False, sort_keys=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        fd = os.open(str(tmp), flags, stat.S_IRUSR | stat.S_IWUSR)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.replace(str(tmp), str(self.path))
        try:
            os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass

    # -- credential API ----------------------------------------------------

    def load(self) -> Optional[OrcaCredential]:
        record = self._read_all().get(SECRETS_NAMESPACE)
        if not isinstance(record, dict):
            return None
        key = record.get("api_key")
        if not isinstance(key, str) or not key.strip():
            return None
        return OrcaCredential(
            api_key=key.strip(),
            source=str(record.get("source") or "stored"),
            key_id=str(record.get("key_id") or ""),
            account_id=record.get("account_id"),
            generation=int(record.get("generation") or 1),
            scope=record.get("scope"),
            needs_reauth=bool(record.get("needs_reauth")),
            updated_at=record.get("updated_at"),
        )

    def save(self, credential: OrcaCredential) -> None:
        data = self._read_all()
        data[SECRETS_NAMESPACE] = credential.to_record()
        self._write_all(data)

    def clear(self) -> bool:
        data = self._read_all()
        if SECRETS_NAMESPACE not in data:
            return False
        del data[SECRETS_NAMESPACE]
        self._write_all(data)
        return True

    def mark_needs_reauth(self, generation: int) -> bool:
        """Flag the credential for re-authentication.

        Only the exact generation that made the rejected request is touched, so
        a late failure from an old request cannot poison a freshly issued
        credential. The stored secret is *not* deleted here -- a transient or
        misclassified failure must stay recoverable.
        """
        current = self.load()
        if current is None:
            return False
        if current.generation != generation:
            logger.debug("Ignoring stale 401 for credential generation %s", generation)
            return False
        if current.needs_reauth:
            return True
        self.save(replace(current, needs_reauth=True))
        return True


def _next_generation(current: Optional[OrcaCredential], account_id: Optional[str]) -> int:
    if current is None:
        return 1
    if account_id and current.account_id and account_id != current.account_id:
        return 1
    return current.generation + 1


# ---------------------------------------------------------------------------
# Credential interface + adapters
# ---------------------------------------------------------------------------


class CredentialProvider(ABC):
    """One way of obtaining an OrcaRouter credential."""

    #: Stable identifier of the adapter, recorded with the credential.
    source: str = "unknown"

    @abstractmethod
    def acquire(self) -> OrcaCredential:
        """Return a usable credential, or raise :class:`OrcaAuthError`."""


class ApiKeyCredentialProvider(CredentialProvider):
    """Adapter for a user-supplied ``sk-orca-...`` key.

    The key comes from the injected value (config/UI) or, failing that, the
    ``ORCAROUTER_API_KEY`` environment variable. A stored credential is reused
    when it is still usable and matches the injected key.
    """

    source = "api_key"

    def __init__(
        self,
        api_key: Optional[str] = None,
        store: Optional[OrcaCredentialStore] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        self._explicit = api_key
        self.store = store or OrcaCredentialStore()
        self._env = os.environ if env is None else env

    def resolve_key(self) -> Optional[str]:
        key = self._explicit or self._env.get(KEY_ENV)
        if key is None:
            return None
        key = key.strip()
        return key or None

    def acquire(self) -> OrcaCredential:
        key = self.resolve_key()
        if not key:
            raise OrcaAuthError(
                f"No OrcaRouter API key available. Set {KEY_ENV} or enter an "
                f"{KEY_PREFIX}... key (get one at {KEY_DASHBOARD_URL})."
            )
        if not looks_like_orcarouter_key(key):
            raise OrcaAuthError(
                f"That does not look like an OrcaRouter API key (expected "
                f"'{KEY_PREFIX}...'). Copy one from {KEY_DASHBOARD_URL}."
            )
        current = self.store.load()
        if current is not None and current.api_key == key and not current.needs_reauth:
            return current
        # A different key is a different credential, even for the same account:
        # bump the generation so a late failure from the previous key cannot mark
        # the new one as rejected.
        credential = OrcaCredential(
            api_key=key,
            source=self.source,
            key_id=mask_key(key),
            account_id=current.account_id if current and current.api_key == key else None,
            generation=_next_generation(current, None),
            updated_at=time.time(),
        )
        self.store.save(credential)
        return credential


class PkceCredentialProvider(CredentialProvider):
    """Adapter for browser sign-in (OAuth 2.0 + PKCE, S256).

    Flow A (loopback redirect) is preferred; Flow B (out-of-band code) is used
    when the caller cannot receive a redirect. Both always send S256 and always
    compare ``state`` before redeeming a code.
    """

    source = "oauth_pkce"

    def __init__(
        self,
        auth_base: Optional[str] = None,
        store: Optional[OrcaCredentialStore] = None,
        app_name: str = "OpenEvolve",
        callback_url: Optional[str] = None,
        oob: bool = False,
        timeout: int = DEFAULT_CONNECT_TIMEOUT,
        open_browser: bool = True,
        code_prompt: Optional[Callable[[str], str]] = None,
        url_sink: Optional[Callable[[str], None]] = None,
        listen_host: str = "127.0.0.1",
    ):
        self.auth_base = _normalize_base(auth_base) if auth_base else resolve_auth_base()
        self.store = store or OrcaCredentialStore()
        self.app_name = app_name
        self.callback_url = callback_url
        self.oob = oob
        self.timeout = timeout
        self.open_browser = open_browser
        self.code_prompt = code_prompt or (lambda _url: input("Code: "))
        self.url_sink = url_sink
        self.listen_host = listen_host

    # -- authorize / exchange (pure helpers, directly unit-testable) --------

    def build_authorize_url(self, challenge: str, state: str, callback_url: str) -> str:
        params = {
            "callback_url": callback_url,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "app_name": self.app_name,
            "scope": "api",
        }
        return f"{self.auth_base}{AUTHORIZE_PATH}?" + urllib.parse.urlencode(params)

    def exchange(self, code: str, verifier: str, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
        """Redeem an authorization code for an API key.

        Posts to the *auth* origin's ``/api/v1/auth/keys`` -- never to the
        inference origin -- and never puts the verifier in the URL.
        """
        if not code:
            raise OrcaAuthError("No authorization code was received.")
        body = json.dumps(
            {
                "code": code,
                "code_verifier": verifier,
                "code_challenge_method": "S256",
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.auth_base}{EXCHANGE_PATH}",
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            detail = _safe_http_error(exc)
            if exc.code == 403:
                raise OrcaAuthError(
                    "OrcaRouter rejected the authorization code: it is unknown, expired, "
                    "already used, or does not match this sign-in attempt. Start again."
                ) from None
            if exc.code == 400:
                raise OrcaAuthError(
                    f"OrcaRouter refused the exchange request ({detail}). "
                    "Start again from a fresh sign-in."
                ) from None
            if exc.code == 429:
                raise OrcaAuthError(
                    "OrcaRouter is rate limiting sign-ins (429). Wait a few minutes; "
                    "there is a cap of 10 issued keys per user per 24 hours."
                ) from None
            raise OrcaAuthError(f"OrcaRouter sign-in failed ({exc.code} {detail}).") from None
        except urllib.error.URLError as exc:
            raise OrcaAuthError(
                f"Could not reach the OrcaRouter authorization service "
                f"({exc.reason.__class__.__name__}). Check your network and try again."
            ) from None
        except (TimeoutError, socket.timeout):
            raise OrcaAuthError(
                "Timed out talking to the OrcaRouter authorization service."
            ) from None

        key = payload.get("key")
        if not isinstance(key, str) or not key.strip():
            raise OrcaAuthError("OrcaRouter returned no API key for this authorization.")
        granted_scope = payload.get("scope")
        if granted_scope and granted_scope != MINIMUM_SCOPE:
            raise OrcaAuthError(
                f"OrcaRouter granted scope '{granted_scope}', which does not cover "
                f"inference (needs '{MINIMUM_SCOPE}'). Ask an administrator for a role "
                "that permits the 'api' scope."
            )
        return {"key": key.strip(), "scope": granted_scope, "user_id": payload.get("user_id")}

    # -- flows -------------------------------------------------------------

    def _new_attempt(self) -> Tuple[str, str, str]:
        """Fresh verifier/challenge/state for one attempt, from a CSPRNG."""
        verifier = _b64url(secrets.token_bytes(32))
        challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
        state = _b64url(secrets.token_bytes(16))
        return verifier, challenge, state

    def _announce(self, url: str) -> None:
        if self.url_sink is not None:
            self.url_sink(url)
        else:
            logger.info("Open this URL to authorize OpenEvolve with OrcaRouter:\n%s", url)
        if self.open_browser:
            try:
                webbrowser.open(url)
            except Exception:  # pragma: no cover - depends on desktop environment
                logger.debug("Could not open a browser automatically")

    def acquire(self) -> OrcaCredential:
        verifier, challenge, state = self._new_attempt()

        if self.oob or self.callback_url == "oob":
            code = self._run_oob(challenge, state)
        elif self.callback_url:
            code = self._run_callback(challenge, state, self.callback_url)
        else:
            code = self._run_loopback(challenge, state)

        result = self.exchange(code, verifier)
        del verifier

        current = self.store.load()
        credential = OrcaCredential(
            api_key=result["key"],
            source=self.source,
            key_id=mask_key(result["key"]),
            account_id=str(result["user_id"]) if result.get("user_id") else None,
            generation=_next_generation(
                current,
                str(result["user_id"]) if result.get("user_id") else None,
            ),
            scope=result.get("scope"),
            updated_at=time.time(),
        )
        self.store.save(credential)
        return credential

    def _run_loopback(self, challenge: str, state: str) -> str:
        """Flow A: listen on loopback first, then open the browser."""
        result: Dict[str, Any] = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # silence the default log
                pass

            def do_GET(self) -> None:  # noqa: N802 - stdlib naming
                parsed = urllib.parse.urlsplit(self.path)
                if parsed.path != "/cb":
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    b"<html><body><p>OrcaRouter sign-in complete. "
                    b"You can close this tab and return to OpenEvolve.</p></body></html>"
                )
                if result.get("done"):
                    return
                params = urllib.parse.parse_qs(parsed.query)
                returned_state = (params.get("state") or [""])[0]
                # Compare state before looking at anything else.
                if not hmac.compare_digest(returned_state, state):
                    result.update(
                        done=True,
                        error=OrcaAuthError(
                            "OrcaRouter sign-in returned a mismatched state value; "
                            "the response was ignored. Start again."
                        ),
                    )
                    return
                error = (params.get("error") or [""])[0]
                if error:
                    result.update(
                        done=True,
                        error=OrcaAuthError(
                            "OrcaRouter authorization was denied."
                            if error == "access_denied"
                            else f"OrcaRouter authorization failed ({error})."
                        ),
                    )
                    return
                code = (params.get("code") or [""])[0]
                if not code:
                    result.update(
                        done=True, error=OrcaAuthError("OrcaRouter returned no authorization code.")
                    )
                    return
                result.update(done=True, code=code)

        server = http.server.ThreadingHTTPServer((self.listen_host, 0), Handler)
        server.timeout = 1.0
        port = server.server_address[1]
        callback_url = f"http://{self.listen_host}:{port}/cb"
        url = self.build_authorize_url(challenge, state, callback_url)
        self._announce(url)

        deadline = time.monotonic() + self.timeout
        try:
            while not result.get("done"):
                server.handle_request()
                if not result.get("done") and time.monotonic() > deadline:
                    raise OrcaAuthError(
                        "Timed out waiting for OrcaRouter authorization. Nothing was saved; "
                        "run the sign-in again when you are ready."
                    )
        finally:
            server.server_close()

        if result.get("error"):
            raise result["error"]
        return result["code"]

    def _run_callback(self, challenge: str, state: str, callback_url: str) -> str:
        """Flow B (hosted callback): the code is pasted back by the user."""
        url = self.build_authorize_url(challenge, state, callback_url)
        self._announce(url)
        return self._read_code(state)

    def _run_oob(self, challenge: str, state: str) -> str:
        """Flow B: ``callback_url=oob``; S256 is mandatory here."""
        url = self.build_authorize_url(challenge, state, "oob")
        self._announce(url)
        return self._read_code(state)

    def _read_code(self, state: str) -> str:
        raw = self.code_prompt(self.auth_base).strip()
        if not raw:
            raise OrcaAuthError("No authorization code was entered. Nothing was saved.")
        # A pasted redirect URL is accepted, and its state is checked the same
        # way the loopback listener checks it.
        if raw.lower().startswith(("http://", "https://")):
            params = urllib.parse.parse_qs(urllib.parse.urlsplit(raw).query)
            returned = (params.get("state") or [""])[0]
            if not returned or not hmac.compare_digest(returned, state):
                raise OrcaAuthError(
                    "That redirect URL carries a mismatched state value; refusing it."
                )
            if (params.get("error") or [""])[0]:
                raise OrcaAuthError("OrcaRouter authorization was denied.")
            raw = (params.get("code") or [""])[0]
            if not raw:
                raise OrcaAuthError("That redirect URL carries no authorization code.")
        return raw


def _safe_http_error(exc: urllib.error.HTTPError) -> str:
    """Describe an HTTP error without echoing credentials."""
    try:
        body = exc.read().decode("utf-8", errors="replace")[:200]
    except Exception:  # pragma: no cover - body may already be consumed
        body = ""
    return redact(body.strip()) or exc.reason or ""


# ---------------------------------------------------------------------------
# Resolution used by the provider adapter
# ---------------------------------------------------------------------------


def acquire_credential(
    provider_id: str,
    api_key: Optional[str] = None,
    store: Optional[OrcaCredentialStore] = None,
    oob: bool = False,
    allow_stored_fallback: bool = True,
) -> OrcaCredential:
    """Return a credential for an OrcaRouter provider id.

    ``orcarouter`` uses the API-key adapter; ``orcarouter_oauth`` uses PKCE.
    Either adapter may reuse an already stored credential, so a successful
    sign-in is not repeated on every launch (OrcaRouter caps PKCE key issuance
    at 10 per user per 24 hours).
    """
    store = store or OrcaCredentialStore()

    if provider_id == "orcarouter_oauth":
        stored = store.load() if allow_stored_fallback else None
        if stored is not None and not stored.needs_reauth:
            return stored
        return PkceCredentialProvider(store=store, oob=oob).acquire()

    if provider_id == "orcarouter":
        resolved = api_key or os.environ.get(KEY_ENV)
        if resolved:
            return ApiKeyCredentialProvider(resolved, store=store).acquire()
        if allow_stored_fallback:
            stored = store.load()
            if stored is not None and not stored.needs_reauth:
                return stored
        return ApiKeyCredentialProvider(None, store=store).acquire()

    raise OrcaAuthError(f"Unknown OrcaRouter credential provider: {provider_id}")
