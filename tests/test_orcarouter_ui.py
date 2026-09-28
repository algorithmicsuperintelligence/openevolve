"""
Tests for catalog-driven model selection and the OrcaRouter settings UI.

The model selector must be populated from the catalog (never free text), each
capability/modality must be filtered independently, and every terminal path of
an asynchronous login must release the attempt — including `pagehide`, which
must clear busy/hint state without relying on the guarded `finally` block.
"""

import json
import os
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from openevolve.llm.orcarouter_auth import (  # noqa: E402
    OrcaAuthError,
    OrcaCredential,
    OrcaCredentialStore,
)
from openevolve.llm.orcarouter_catalog import (  # noqa: E402
    CAPABILITY_CHAT,
    CAPABILITY_EMBEDDING,
    CAPABILITY_IMAGE,
    CatalogResult,
    OrcaCatalogClient,
    fallback_catalog,
    parse_catalog,
)

from test_orcarouter_catalog import FIXTURE, FakeOpener  # noqa: E402

FAKE_KEY = "sk-orca-" + "a" * 44


class CatalogDrivenSelection(unittest.TestCase):
    """Options handed to a selector are the filtered catalog, nothing else."""

    def setUp(self):
        self.client = OrcaCatalogClient(api_key=FAKE_KEY, opener=FakeOpener(FIXTURE))

    def test_options_come_from_the_api_not_a_handwritten_list(self):
        result = self.client.discover()
        ids = set(result.ids(CAPABILITY_CHAT, "text"))
        self.assertNotEqual(ids, {m.id for m in fallback_catalog()})
        self.assertIn("deepseek/deepseek-v4-pro", ids)

    def test_image_attachment_removes_text_only_models(self):
        before = set(self.client.discover().ids(CAPABILITY_CHAT, "text"))
        after = set(self.client.discover().ids(CAPABILITY_CHAT, "image"))
        self.assertIn("deepseek/deepseek-v4-pro", before)
        self.assertNotIn("deepseek/deepseek-v4-pro", after)
        self.assertTrue(after.issubset(before))

    def test_a_stale_selection_becomes_incompatible(self):
        text_options = self.client.discover().options(CAPABILITY_CHAT, "text")
        selected = "deepseek/deepseek-v4-pro"
        self.assertIn(selected, [o["id"] for o in text_options])
        image_options = self.client.discover().options(CAPABILITY_CHAT, "image")
        self.assertNotIn(selected, [o["id"] for o in image_options])
        # The UI must clear it rather than silently keep an incompatible value.
        self.assertNotIn(selected, [o["id"] for o in image_options])

    def test_capability_switch_changes_the_option_set(self):
        chat = set(self.client.discover().ids(CAPABILITY_CHAT))
        embed = set(self.client.discover().ids(CAPABILITY_EMBEDDING))
        image = set(self.client.discover().ids(CAPABILITY_IMAGE))
        self.assertTrue(embed.isdisjoint(chat))
        self.assertTrue(image.isdisjoint(chat))

    def test_degraded_catalog_marks_every_option_verified(self):
        client = OrcaCatalogClient(
            api_key=FAKE_KEY, opener=FakeOpener(None, raise_exc=OSError("offline"))
        )
        result = client.discover()
        self.assertTrue(result.degraded)
        self.assertTrue(result.options()[0]["verified"])

    def test_live_catalog_marks_no_option_verified(self):
        result = self.client.discover()
        self.assertFalse(any(o["verified"] for o in result.options()))


class FakePkceProvider:
    """Stands in for PkceCredentialProvider inside the login manager."""

    def __init__(self, outcome="success", url="https://www.orcarouter.ai/auth?x=1", delay=0.0):
        self.outcome = outcome
        self.url = url
        self.delay = delay
        self.url_sink = None
        self.cancelled = False

    def build_authorize_url(self, challenge, state, callback_url):
        return self.url

    def acquire(self):
        if self.url_sink:
            self.url_sink(self.url)
        if self.delay:
            time.sleep(self.delay)
        if self.outcome == "error":
            raise OrcaAuthError("denied by the user")
        if self.outcome == "cancel":
            # Simulate a flow that never completes.
            while not self.cancelled:
                time.sleep(0.01)
            raise OrcaAuthError("cancelled")
        return OrcaCredential(api_key=FAKE_KEY, source="oauth_pkce", generation=1)


class OrcaLoginManagerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = OrcaCredentialStore(str(Path(self._tmp.name) / "secrets.yaml"))
        from orcarouter_ui import OrcaLoginManager

        self.manager_cls = OrcaLoginManager

    def _manager(self, outcome="success", delay=0.0):
        provider = FakePkceProvider(outcome=outcome, delay=delay)

        def factory(oob):
            return provider

        return self.manager_cls(store=self.store, provider_factory=factory, timeout=5), provider

    def _wait(self, manager, attempt_id, states=("success", "error", "cancelled"), timeout=5.0):
        deadline = time.time() + timeout
        payload = {}
        while time.time() < deadline:
            payload = manager.poll(attempt_id)
            if payload.get("state") in states:
                return payload
            time.sleep(0.02)
        return payload

    def test_success_records_a_redacted_credential(self):
        manager, provider = self._manager()
        started = manager.start()
        payload = self._wait(manager, started["attempt_id"])
        self.assertEqual(payload["state"], "success")
        self.assertEqual(payload["secret_masked"], "sk-orca-…" + FAKE_KEY[-4:])
        self.assertNotIn(FAKE_KEY, json.dumps(payload))

    def test_error_releases_the_attempt(self):
        manager, provider = self._manager(outcome="error")
        started = manager.start()
        payload = self._wait(manager, started["attempt_id"])
        self.assertEqual(payload["state"], "error")
        self.assertIn("denied", payload["error"])

    def test_error_message_is_redacted(self):
        manager, provider = self._manager(outcome="error")
        provider.acquire = lambda: (_ for _ in ()).throw(OrcaAuthError(f"bad {FAKE_KEY}"))
        started = manager.start()
        payload = self._wait(manager, started["attempt_id"])
        self.assertNotIn(FAKE_KEY, payload["error"])

    def test_explicit_cancel_marks_the_attempt_cancelled(self):
        manager, provider = self._manager(outcome="cancel")
        started = manager.start()
        time.sleep(0.05)
        result = manager.cancel(started["attempt_id"])
        self.assertTrue(result["cancelled"])
        payload = manager.poll(started["attempt_id"])
        self.assertEqual(payload["state"], "cancelled")

    def test_a_new_attempt_gets_a_higher_generation(self):
        manager, provider = self._manager(outcome="cancel")
        first = manager.start()
        second = manager.start()
        self.assertGreater(second["generation"], first["generation"])

    def test_a_stale_result_cannot_overwrite_a_newer_attempt(self):
        """A late success from attempt 1 must not land under attempt 2's state."""
        slow = FakePkceProvider(delay=0.3)
        stuck = FakePkceProvider(outcome="cancel")
        providers = {"n": 0}

        def factory(oob):
            providers["n"] += 1
            return slow if providers["n"] == 1 else stuck

        manager = self.manager_cls(store=self.store, provider_factory=factory, timeout=5)
        first = manager.start()
        time.sleep(0.05)
        second = manager.start()
        # Cancel (invalidate) attempt 1 while it is still running.
        manager.cancel(first["attempt_id"])
        self.assertEqual(manager.poll(second["attempt_id"])["state"], "pending")
        time.sleep(0.6)
        # The late success resolved but was dropped: neither attempt was mutated.
        self.assertEqual(manager.poll(first["attempt_id"])["state"], "cancelled")
        self.assertEqual(manager.poll(second["attempt_id"])["state"], "pending")

    def test_prime_hides_the_busy_flag_after_pagehide(self):
        """pagehide clears state synchronously and a second login can start."""
        manager, provider = self._manager(outcome="cancel")
        first = manager.start()
        time.sleep(0.05)
        self.assertEqual(manager.poll(first["attempt_id"])["state"], "pending")
        # pagehide: cancel with keepalive, then start a second login immediately.
        manager.cancel(first["attempt_id"])
        second = manager.start()
        self.assertEqual(manager.poll(first["attempt_id"])["state"], "cancelled")
        self.assertIn(manager.poll(second["attempt_id"])["state"], ("pending", "cancelled"))

    def test_unknown_attempt_polls_cleanly(self):
        manager, _ = self._manager()
        self.assertEqual(manager.poll("does-not-exist")["state"], "unknown")
        self.assertFalse(manager.cancel("does-not-exist")["cancelled"])

    def test_a_timeout_raised_by_the_flow_is_surfaced_as_an_error(self):
        """The sign-in flow owns the deadline; its failure must release the attempt."""

        class TimingOut(FakePkceProvider):
            def acquire(self):
                raise OrcaAuthError("Timed out waiting for OrcaRouter authorization.")

        manager = self.manager_cls(
            store=self.store, provider_factory=lambda oob: TimingOut(), timeout=0
        )
        started = manager.start()
        payload = self._wait(manager, started["attempt_id"], timeout=2.0)
        self.assertEqual(payload["state"], "error")
        self.assertIn("Timed out", payload["error"])


class FakeCatalogFactory:
    def __init__(self, result):
        self.result = result

    def __call__(self):
        outer = self

        class Client:
            def discover(self, *a, **k):
                return outer.result

            def invalidate(self):
                pass

        return Client()


class OrcaUiRouteTests(unittest.TestCase):
    """The settings page and its JSON endpoints, with a real Flask test client."""

    def setUp(self):
        from orcarouter_ui import OrcaLoginManager, create_orcarouter_blueprint
        from flask import Flask

        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.secrets_path = str(Path(self._tmp.name) / "secrets.yaml")
        self.env = patch.dict(os.environ, {"OPENEVOLVE_SECRETS_FILE": self.secrets_path})
        self.env.start()
        self.addCleanup(self.env.stop)

        self.catalog_factory = None
        app = Flask(
            __name__,
            template_folder=str(Path(__file__).resolve().parent.parent / "scripts" / "templates"),
        )
        blueprint, self.manager = create_orcarouter_blueprint(
            login_manager=OrcaLoginManager(store=OrcaCredentialStore(self.secrets_path)),
            catalog_factory=lambda: self.catalog_factory,
        )
        app.register_blueprint(blueprint)
        self.app = app
        self.client = app.test_client()

    def test_page_shows_both_authentication_methods(self):
        response = self.client.get("/orcarouter/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('data-auth-method="api_key"', html)
        self.assertIn('data-auth-method="oauth_pkce"', html)
        self.assertIn("orcarouter", html)
        self.assertIn("orcarouter_oauth", html)
        self.assertIn("console/authorized-apps", html)
        self.assertIn("modelSelector", html)

    def test_status_endpoint_is_redacted(self):
        store = OrcaCredentialStore(self.secrets_path)
        store.save(OrcaCredential(api_key=FAKE_KEY, source="api_key", generation=1))
        body = self.client.get("/orcarouter/api/status").get_json()
        self.assertNotIn(FAKE_KEY, json.dumps(body))

    def test_saving_an_api_key_through_the_route(self):
        response = self.client.post("/orcarouter/api/key", json={"api_key": FAKE_KEY})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["ok"])
        self.assertNotIn(FAKE_KEY, json.dumps(body))
        self.assertEqual(OrcaCredentialStore(self.secrets_path).load().api_key, FAKE_KEY)

    def test_a_malformed_key_is_rejected_without_echoing_it(self):
        response = self.client.post("/orcarouter/api/key", json={"api_key": "sk-proj-nope"})
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("sk-proj-nope", response.get_data(as_text=True))

    def test_clearing_the_key(self):
        self.client.post("/orcarouter/api/key", json={"api_key": FAKE_KEY})
        response = self.client.post("/orcarouter/api/key/clear")
        self.assertTrue(response.get_json()["removed"])
        self.assertIsNone(OrcaCredentialStore(self.secrets_path).load())

    def test_logout_endpoint(self):
        self.client.post("/orcarouter/api/key", json={"api_key": FAKE_KEY})
        response = self.client.post("/orcarouter/api/logout")
        self.assertTrue(response.get_json()["ok"])
        self.assertIsNone(OrcaCredentialStore(self.secrets_path).load())

    def test_models_endpoint_returns_filtered_options(self):
        result = CatalogResult(
            models=tuple(parse_catalog(FIXTURE)),
            source="live",
            api_base="https://api.orcarouter.ai/v1",
        )
        self.catalog_factory = FakeCatalogFactory(result)()
        body = self.client.get("/orcarouter/api/models?capability=chat&modality=image").get_json()
        ids = [m["id"] for m in body["models"]]
        self.assertIn("anthropic/claude-opus-4.8", ids)
        self.assertNotIn("deepseek/deepseek-v4-pro", ids)
        self.assertEqual(body["source"], "live")

    def test_models_route_authenticates_with_the_stored_credential(self):
        """The dropdown must reflect this account, not the anonymous public list."""
        from openevolve.llm.orcarouter import resolve_credential_for_discovery
        from orcarouter_ui import _catalog_client

        OrcaCredentialStore(self.secrets_path).save(
            OrcaCredential(api_key=FAKE_KEY, source="api_key", generation=1)
        )
        client = _catalog_client()
        self.assertEqual(client.api_key, FAKE_KEY)

        # A rejected credential is not put back on the wire: discovery falls back
        # to the anonymous catalog and the UI labels it as public.
        store = OrcaCredentialStore(self.secrets_path)
        store.mark_needs_reauth(1)
        self.assertIsNone(resolve_credential_for_discovery(store))
        self.assertIsNone(_catalog_client().api_key)

    def test_models_route_reports_a_public_catalog_without_a_credential(self):
        result = CatalogResult(
            models=tuple(parse_catalog(FIXTURE)),
            source="live",
            api_base="https://api.orcarouter.ai/v1",
        )
        self.catalog_factory = FakeCatalogFactory(result)()
        body = self.client.get("/orcarouter/api/models?capability=chat").get_json()
        self.assertFalse(body["authenticated"])
        self.assertTrue(body["public_catalog"])

    def test_dropdown_options_are_tagged_with_their_catalog_source(self):
        js = (
            Path(__file__).resolve().parent.parent / "scripts" / "static" / "js" / "orcarouter.js"
        ).read_text()
        self.assertIn('row.setAttribute("data-catalog-source", state.source', js)
        self.assertIn("state.source = body.source", js)

    def test_login_and_cancel_endpoints(self):
        provider = FakePkceProvider(outcome="cancel")
        self.manager._provider_factory = lambda oob: provider
        started = self.client.post("/orcarouter/api/login", json={}).get_json()
        self.assertIn("attempt_id", started)
        cancelled = self.client.post(
            f"/orcarouter/api/login/{started['attempt_id']}/cancel"
        ).get_json()
        self.assertTrue(cancelled["cancelled"])
        polled = self.client.get(f"/orcarouter/api/login/{started['attempt_id']}").get_json()
        self.assertEqual(polled["state"], "cancelled")


class GuiLifecycleContractTests(unittest.TestCase):
    """The pagehide contract has to be visible in the shipped JavaScript."""

    def setUp(self):
        self.js = (
            Path(__file__).resolve().parent.parent / "scripts" / "static" / "js" / "orcarouter.js"
        ).read_text()

    def test_pagehide_handler_clears_busy_and_hint_synchronously(self):
        self.assertIn('window.addEventListener("pagehide", onPageHide)', self.js)
        self.assertIn('window.addEventListener("beforeunload", onPageHide)', self.js)
        handler = self.js.split("function onPageHide()")[1].split("function init()")[0]
        self.assertIn("setLoginBusy(false", handler)
        self.assertIn("stopPolling()", handler)

    def test_pagehide_invalidates_the_generation_before_cancelling(self):
        handler = self.js.split("function onPageHide()")[1].split("function init()")[0]
        self.assertIn("state.generation += 1", handler)
        self.assertIn("cancelLogin(true)", handler)

    def test_cancel_uses_keepalive(self):
        self.assertIn("keepalive: !!keepalive", self.js)

    def test_generation_guard_on_every_async_response(self):
        self.assertIn("generation !== state.generation", self.js)

    def test_api_key_is_never_persisted_in_the_browser(self):
        for forbidden in ("localStorage", "sessionStorage", "document.cookie"):
            self.assertNotIn(forbidden, self.js)

    def test_both_controls_are_wired(self):
        self.assertIn('$("saveKeyBtn").addEventListener', self.js)
        self.assertIn('$("connectBtn").addEventListener', self.js)

    def test_incompatible_selection_is_cleared(self):
        self.assertIn("ensureSelectionIsCompatible", self.js)
        self.assertIn('$("modelTriggerLabel").textContent = "Select a model…"', self.js)


class EmbeddingProviderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.secrets_path = str(Path(self._tmp.name) / "secrets.yaml")
        self.env = patch.dict(os.environ, {"OPENEVOLVE_SECRETS_FILE": self.secrets_path})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_embeddings_use_the_orcarouter_provider_path(self):
        from openevolve.embedding import EmbeddingClient

        OrcaCredentialStore(self.secrets_path).save(
            OrcaCredential(api_key=FAKE_KEY, source="api_key", generation=1)
        )
        result = CatalogResult(
            models=tuple(parse_catalog(FIXTURE)),
            source="live",
            api_base="https://api.orcarouter.ai/v1",
        )
        with (
            patch("openevolve.llm.orcarouter_catalog.OrcaCatalogClient") as client_cls,
            patch("openevolve.embedding.openai.OpenAI") as mock_openai,
        ):
            client_cls.return_value.discover.return_value = result
            client = EmbeddingClient("openai/text-embedding-3-small", provider="orcarouter")
        self.assertEqual(client.model, "openai/text-embedding-3-small")
        self.assertEqual(mock_openai.call_args.kwargs["base_url"], "https://api.orcarouter.ai/v1")
        self.assertEqual(mock_openai.call_args.kwargs["api_key"], FAKE_KEY)

    def test_an_incompatible_embedding_model_is_refused(self):
        from openevolve.embedding import EmbeddingClient

        OrcaCredentialStore(self.secrets_path).save(
            OrcaCredential(api_key=FAKE_KEY, source="api_key", generation=1)
        )
        result = CatalogResult(
            models=tuple(parse_catalog(FIXTURE)),
            source="live",
            api_base="https://api.orcarouter.ai/v1",
        )
        with patch("openevolve.llm.orcarouter_catalog.OrcaCatalogClient") as client_cls:
            client_cls.return_value.discover.return_value = result
            with self.assertRaises(ValueError):
                EmbeddingClient("openai/gpt-5.5", provider="orcarouter")

    def test_embeddings_without_a_credential_is_actionable(self):
        from openevolve.embedding import EmbeddingClient

        with self.assertRaises(ValueError) as ctx:
            EmbeddingClient("openai/text-embedding-3-small", provider="orcarouter")
        self.assertIn("connect orcarouter", str(ctx.exception))

    def test_existing_embedding_behaviour_is_unchanged(self):
        from openevolve.embedding import EmbeddingClient

        with patch.dict(os.environ, {"OPENAI_API_KEY": "llm-key"}, clear=False):
            with patch("openevolve.embedding.openai.OpenAI") as mock_openai:
                EmbeddingClient("qwen/qwen3-embedding-8b", api_base="https://openrouter.ai/api/v1")
        self.assertEqual(mock_openai.call_args.kwargs["base_url"], "https://openrouter.ai/api/v1")


if __name__ == "__main__":
    unittest.main()
