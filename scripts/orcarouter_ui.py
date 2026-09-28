"""
OrcaRouter provider settings UI for the OpenEvolve visualizer.

This is a first-class provider page, not a special-case panel: both credential
choices (paste an API key, or sign in with an OrcaRouter account) live side by
side, and the model selector is populated from the live capability-filtered
catalog rather than free text.

The page runs in the same local process as the run, so the browser never holds
the key. Every response the browser receives is redacted; the key itself only
ever travels to ``api.orcarouter.ai``.
"""

import json
import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from flask import Blueprint, jsonify, render_template, request

logger = logging.getLogger(__name__)

#: A login attempt becomes collectable after this many seconds without a poll.
ATTEMPT_TTL_SECONDS = 900


def _blocked_origins(auth_base: str, api_base: str) -> List[str]:
    return [auth_base, api_base]


class OrcaLoginManager:
    """Runs PKCE sign-ins in a background thread and hands the URL to the UI.

    Every attempt carries a monotonically increasing generation. A late result
    from an old attempt is dropped instead of overwriting a newer login, and
    every terminal path (success, denial, error, timeout, explicit cancel,
    pagehide, provider switch, unmount) releases the attempt.
    """

    def __init__(
        self,
        store=None,
        provider_factory: Optional[Callable[..., Any]] = None,
        timeout: int = 300,
    ):
        self._lock = threading.Lock()
        self._attempts: Dict[str, Dict[str, Any]] = {}
        self._generation = 0
        self._store = store
        self._provider_factory = provider_factory
        self._timeout = timeout

    # -- helpers -----------------------------------------------------------

    def _store_obj(self):
        if self._store is not None:
            return self._store
        from openevolve.llm.orcarouter_auth import OrcaCredentialStore

        self._store = OrcaCredentialStore()
        return self._store

    def _provider(self, oob: bool):
        if self._provider_factory is not None:
            return self._provider_factory(oob=oob)
        from openevolve.llm.orcarouter_auth import PkceCredentialProvider

        return PkceCredentialProvider(
            store=self._store_obj(),
            oob=oob,
            timeout=self._timeout,
            open_browser=True,
        )

    def _reap(self) -> None:
        now = time.time()
        stale = [
            key
            for key, attempt in self._attempts.items()
            if now - attempt["updated_at"] > ATTEMPT_TTL_SECONDS
            and attempt["state"] not in ("pending",)
        ]
        for key in stale:
            self._attempts.pop(key, None)

    # -- API ---------------------------------------------------------------

    def start(self, oob: bool = False) -> Dict[str, Any]:
        with self._lock:
            self._reap()
            self._generation += 1
            generation = self._generation
            attempt_id = uuid.uuid4().hex
            attempt = {
                "id": attempt_id,
                "generation": generation,
                "state": "pending",
                "url": None,
                "error": None,
                "credential": None,
                "updated_at": time.time(),
            }
            self._attempts[attempt_id] = attempt

        def run() -> None:
            try:
                provider = self._provider(oob)
                url = provider.build_authorize_url(
                    "<challenge>", "<state>", "oob" if oob else "http://127.0.0.1:0/cb"
                )
                credential = self._start_provider(provider, attempt_id, generation, oob, url)
                self._finish(attempt_id, generation, credential=credential)
            except Exception as exc:  # every failure must release the attempt
                self._finish(attempt_id, generation, error=self._message(exc))

        thread = threading.Thread(target=run, name="orca-login", daemon=True)
        thread.start()
        return {"attempt_id": attempt_id, "generation": generation}

    def _start_provider(self, provider, attempt_id: str, generation: int, oob: bool, url: str):
        """Run the provider, publishing the authorize URL when it is known."""
        original_sink = getattr(provider, "url_sink", None)

        def sink(authorize_url: str) -> None:
            with self._lock:
                attempt = self._attempts.get(attempt_id)
                if attempt is None or attempt["generation"] != generation:
                    return
                attempt["url"] = authorize_url
                attempt["updated_at"] = time.time()
            if original_sink is not None:
                original_sink(authorize_url)

        provider.url_sink = sink
        return provider.acquire()

    def _finish(
        self,
        attempt_id: str,
        generation: int,
        credential: Any = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            attempt = self._attempts.get(attempt_id)
            if attempt is None or attempt["generation"] != generation:
                # A newer attempt owns the state; drop this result.
                return
            attempt["updated_at"] = time.time()
            if credential is not None:
                attempt["state"] = "success"
                attempt["credential"] = credential
            elif error is not None:
                attempt["state"] = "error"
                attempt["error"] = error
            else:
                attempt["state"] = "cancelled"

    def _message(self, exc: Exception) -> str:
        from openevolve.llm.orcarouter_auth import redact

        text = redact(str(exc)) or exc.__class__.__name__
        return text

    def poll(self, attempt_id: str) -> Dict[str, Any]:
        with self._lock:
            attempt = self._attempts.get(attempt_id)
            if attempt is None:
                return {"state": "unknown"}
            payload = {
                "attempt_id": attempt_id,
                "generation": attempt["generation"],
                "state": attempt["state"],
                "url": attempt["url"],
                "error": attempt["error"],
            }
            if attempt["state"] == "success" and attempt["credential"] is not None:
                payload["secret_masked"] = attempt["credential"].masked
                payload["account_id"] = attempt["credential"].account_id
                payload["scope"] = attempt["credential"].scope
                payload["source"] = attempt["credential"].source
        return payload

    def cancel(self, attempt_id: str) -> Dict[str, Any]:
        """Explicit cancel, pagehide, unmount, provider switch or modal close."""
        with self._lock:
            attempt = self._attempts.get(attempt_id)
            if attempt is None:
                return {"cancelled": False}
            if attempt["state"] == "pending":
                # Invalidate the generation so the in-flight thread's result is
                # dropped even if its own cleanup runs later.
                self._generation += 1
                attempt["generation"] = self._generation
                attempt["state"] = "cancelled"
            attempt["updated_at"] = time.time()
        return {"cancelled": True}


def _catalog_client():
    """Build the catalog client used by the settings routes.

    The stored credential is attached when one is present, so the dropdown lists
    the models this account can actually call. Without a credential the gateway
    still answers with its public catalog; the page labels that case explicitly
    rather than passing it off as the workspace's own list.
    """
    from openevolve.llm.orcarouter import resolve_credential_for_discovery
    from openevolve.llm.orcarouter_catalog import OrcaCatalogClient

    credential = resolve_credential_for_discovery()
    return OrcaCatalogClient(api_key=credential.api_key if credential else None)


def create_orcarouter_blueprint(
    login_manager: Optional[OrcaLoginManager] = None,
    catalog_factory: Optional[Callable[..., Any]] = None,
) -> Blueprint:
    bp = Blueprint("orcarouter", __name__, url_prefix="/orcarouter")
    manager = login_manager or OrcaLoginManager()
    make_catalog = catalog_factory or _catalog_client

    def _status() -> Dict[str, Any]:
        from openevolve.llm.orcarouter import orcarouter_credential_status

        return orcarouter_credential_status()

    @bp.route("", methods=["GET"], strict_slashes=False)
    @bp.route("/", methods=["GET"], strict_slashes=False)
    def provider_page():
        from openevolve.llm.orcarouter import PROVIDER_API_KEY, PROVIDER_OAUTH
        from openevolve.llm.orcarouter_auth import KEY_DASHBOARD_URL

        return render_template(
            "orcarouter_page.html",
            provider_api_key=PROVIDER_API_KEY,
            provider_oauth=PROVIDER_OAUTH,
            key_dashboard_url=KEY_DASHBOARD_URL,
        )

    @bp.get("/api/status")
    def api_status():
        return jsonify(_status())

    @bp.get("/api/models")
    def api_models():
        capability = request.args.get("capability", "chat")
        modality = request.args.get("modality", "text")
        refresh = request.args.get("refresh") in ("1", "true", "yes")
        client = make_catalog()
        if refresh:
            client.invalidate()
        status = _status()
        result = client.discover(capability=capability)
        # Without a credential the gateway still answers, with the public
        # catalog rather than this workspace's. Say which one the user got.
        public_catalog = not status["authenticated"]
        return jsonify(
            {
                **result.status(),
                "capability": capability,
                "modality": modality,
                "authenticated": status["authenticated"],
                "needs_reauth": status["needs_reauth"],
                "public_catalog": public_catalog,
                "catalog_source_url": f"{result.api_base}/models",
                "models": result.options(capability, modality),
            }
        )

    @bp.post("/api/key")
    def api_key():
        from openevolve.llm.orcarouter_auth import ApiKeyCredentialProvider, OrcaAuthError

        body = request.get_json(silent=True) or {}
        key = str(body.get("api_key") or "").strip()
        try:
            credential = ApiKeyCredentialProvider(key).acquire()
        except OrcaAuthError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify(
            {
                "ok": True,
                "secret_masked": credential.masked,
                "source": credential.source,
                "generation": credential.generation,
            }
        )

    @bp.post("/api/key/clear")
    def api_key_clear():
        from openevolve.llm.orcarouter import logout

        return jsonify({"ok": True, "removed": logout()})

    @bp.post("/api/login")
    def api_login():
        body = request.get_json(silent=True) or {}
        return jsonify(manager.start(oob=bool(body.get("oob"))))

    @bp.get("/api/login/<attempt_id>")
    def api_login_poll(attempt_id: str):
        return jsonify(manager.poll(attempt_id))

    @bp.post("/api/login/<attempt_id>/cancel")
    def api_login_cancel(attempt_id: str):
        return jsonify(manager.cancel(attempt_id))

    @bp.post("/api/logout")
    def api_logout():
        from openevolve.llm.orcarouter import logout

        return jsonify({"ok": True, "removed": logout()})

    return bp, manager


def register_orcarouter(app, **kwargs) -> OrcaLoginManager:
    """Attach the OrcaRouter blueprint to a Flask app."""
    blueprint, manager = create_orcarouter_blueprint(**kwargs)
    app.register_blueprint(blueprint)
    return manager
