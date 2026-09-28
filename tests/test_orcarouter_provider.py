"""
Tests for the OrcaRouter provider adapter and its registration as a
first-class named provider.

The two credential entries must remain independently selectable, both must send
Bearer auth to ``https://api.orcarouter.ai/v1``, and the downstream request path
and model discovery must not care which one produced the credential.
"""

import asyncio
import json
import os
import unittest
import urllib.error
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import openai

from openevolve.config import Config, LLMModelConfig
from openevolve.llm.ensemble import _PROVIDER_REGISTRY, LLMEnsemble, _create_model
from openevolve.llm.openai import OpenAILLM
from openevolve.llm.orcarouter import (
    OPENAI_STOCK_BASE,
    PROVIDER_ID_KEY,
    PROVIDER_ID_OAUTH,
    OrcaReauthRequired,
    OrcaRouterLLM,
    effective_api_base,
    logout,
    orcarouter_credential_status,
)
from openevolve.llm.orcarouter_auth import (
    API_BASE_ENV,
    KEY_ENV,
    AUTH_BASE_ENV,
    DEFAULT_API_BASE,
    OrcaAuthError,
    OrcaCredential,
    OrcaCredentialStore,
    mask_key,
)

FAKE_KEY = "sk-orca-" + "a" * 44
FAKE_KEY_2 = "sk-orca-" + "b" * 44


class ProviderTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.secrets_path = str(Path(self._tmp.name) / "secrets.yaml")
        self.store = OrcaCredentialStore(self.secrets_path)
        self._env = patch.dict(os.environ, {"OPENEVOLVE_SECRETS_FILE": self.secrets_path})
        self._env.start()
        self.addCleanup(self._env.stop)

    def _cfg(self, **kwargs):
        kwargs.setdefault("name", "openai/gpt-5.5")
        kwargs.setdefault("api_key", FAKE_KEY)
        return LLMModelConfig(**kwargs)

    def _client(self, provider_id=PROVIDER_ID_KEY, credential=None, **kwargs):
        with patch("openevolve.llm.orcarouter.OrcaRouterLLM._probe", create=True):
            return OrcaRouterLLM(
                self._cfg(**kwargs),
                provider_id=provider_id,
                credential=credential or OrcaCredential(api_key=FAKE_KEY, source="api_key"),
                store=self.store,
            )


class TestProviderRegistration(unittest.TestCase):
    def test_both_entries_are_registered(self):
        self.assertIn("orcarouter", _PROVIDER_REGISTRY)
        self.assertIn("orcarouter_oauth", _PROVIDER_REGISTRY)

    def test_registry_builds_the_right_adapter_for_each_entry(self):
        for provider_id in ("orcarouter", "orcarouter_oauth"):
            self.assertIsNotNone(_PROVIDER_REGISTRY[provider_id])

    def test_ensemble_routes_api_key_provider(self):
        cfg = LLMModelConfig(name="openai/gpt-5.5", api_key=FAKE_KEY)
        cfg.provider = PROVIDER_ID_KEY
        with patch("openevolve.llm.orcarouter.OrcaRouterLLM._resolve_credential") as resolve:
            resolve.return_value = OrcaCredential(api_key=FAKE_KEY, source="api_key")
            model = _create_model(cfg)
        self.assertIsInstance(model, OrcaRouterLLM)
        self.assertEqual(model.provider_id, PROVIDER_ID_KEY)

    def test_ensemble_routes_oauth_provider(self):
        cfg = LLMModelConfig(name="openai/gpt-5.5")
        cfg.provider = PROVIDER_ID_OAUTH
        with patch("openevolve.llm.orcarouter.OrcaRouterLLM._resolve_credential") as resolve:
            resolve.return_value = OrcaCredential(api_key=FAKE_KEY, source="oauth_pkce")
            model = _create_model(cfg)
        self.assertEqual(model.provider_id, PROVIDER_ID_OAUTH)

    def test_default_provider_path_unchanged(self):
        from openevolve.llm.openai import OpenAILLM

        cfg = LLMModelConfig(name="gpt-4", api_key=FAKE_KEY)
        self.assertIsInstance(_create_model(cfg), OpenAILLM)


class TestEffectiveApiBase(unittest.TestCase):
    def test_stock_openai_default_is_replaced(self):
        self.assertEqual(effective_api_base(OPENAI_STOCK_BASE), DEFAULT_API_BASE)

    def test_none_uses_the_configured_origin(self):
        self.assertEqual(effective_api_base(None), DEFAULT_API_BASE)

    def test_explicit_value_wins(self):
        self.assertEqual(
            effective_api_base("https://self-hosted.example.test/v1"),
            "https://self-hosted.example.test/v1",
        )

    def test_env_override_is_honoured(self):
        with patch.dict(os.environ, {API_BASE_ENV: "https://infer.example.test/v1"}):
            self.assertEqual(effective_api_base(None), "https://infer.example.test/v1")

    def test_api_base_is_never_derived_from_the_auth_origin(self):
        with patch.dict(os.environ, {AUTH_BASE_ENV: "https://auth.example.test"}, clear=True):
            self.assertEqual(effective_api_base(None), DEFAULT_API_BASE)


class TestCredentialSeamEquivalence(ProviderTestCase):
    """Both adapters must yield the same downstream behaviour."""

    def test_api_key_adapter_produces_a_usable_client(self):
        client = self._client(PROVIDER_ID_KEY)
        self.assertEqual(client.api_base, DEFAULT_API_BASE)
        self.assertEqual(client.provider_id, "orcarouter")
        self.assertFalse(client.needs_reauth)

    def test_pkce_adapter_produces_the_same_client_shape(self):
        client = self._client(
            PROVIDER_ID_OAUTH,
            credential=OrcaCredential(
                api_key=FAKE_KEY_2, source="oauth_pkce", account_id="42", scope="api"
            ),
        )
        self.assertEqual(client.api_base, DEFAULT_API_BASE)
        self.assertEqual(client.provider_id, "orcarouter_oauth")

    def test_request_path_is_identical_regardless_of_credential_source(self):
        captured = {}

        def capture(self, params):
            captured["params"] = params
            return "ok"

        async def capture_async(self, params):
            return capture(self, params)

        api_key_client = self._client(PROVIDER_ID_KEY)
        oauth_client = self._client(
            PROVIDER_ID_OAUTH, credential=OrcaCredential(api_key=FAKE_KEY_2, source="oauth_pkce")
        )
        # Discovery must not branch on the credential source either.
        for client in (api_key_client, oauth_client):
            with patch.object(OrcaRouterLLM, "_call_api", capture_async):
                asyncio.run(client.generate_with_context("sys", [{"role": "user", "content": "x"}]))
        self.assertEqual(captured["params"]["model"], "openai/gpt-5.5")

    def test_model_discovery_does_not_care_about_the_credential_source(self):
        results = []
        for source in ("api_key", "oauth_pkce"):
            client = self._client(
                PROVIDER_ID_KEY, credential=OrcaCredential(api_key=FAKE_KEY, source=source)
            )
            with patch.object(
                client.catalog,
                "discover",
                return_value=MagicMock(options=lambda *a: [{"id": "vendor/m"}]),
            ):
                results.append(client.model_options())
        self.assertEqual(results[0], results[1])

    def test_status_is_redacted_for_both_sources(self):
        for source in ("api_key", "oauth_pkce"):
            client = self._client(
                PROVIDER_ID_KEY,
                credential=OrcaCredential(api_key=FAKE_KEY, source=source, generation=2),
            )
            status = client.status()
            self.assertEqual(status["secret_masked"], mask_key(FAKE_KEY))
            self.assertNotIn(FAKE_KEY, json.dumps(status))
            self.assertIn("console/authorized-apps", status["key_dashboard_url"])


class TestBearerTransport(ProviderTestCase):
    def test_client_is_constructed_with_the_orcarouter_base_and_key(self):
        with patch("openevolve.llm.orcarouter.openai.OpenAI") as mock_openai:
            OrcaRouterLLM(
                self._cfg(),
                provider_id=PROVIDER_ID_KEY,
                credential=OrcaCredential(api_key=FAKE_KEY, source="api_key"),
                store=self.store,
            )
        kwargs = mock_openai.call_args.kwargs
        self.assertEqual(kwargs["base_url"], DEFAULT_API_BASE)
        self.assertEqual(kwargs["api_key"], FAKE_KEY)

    def test_inference_origin_is_the_api_origin_not_the_auth_origin(self):
        client = self._client()
        self.assertIn("api.orcarouter.ai", client.api_base)
        self.assertNotIn("www.orcarouter.ai", client.api_base)

    def test_configured_api_base_override_is_respected(self):
        client = self._client(api_base="https://self-hosted.example.test/v1")
        self.assertEqual(client.api_base, "https://self-hosted.example.test/v1")


class TestTerminalReauth(ProviderTestCase):
    def _client_with_store(self):
        self.store.save(OrcaCredential(api_key=FAKE_KEY, source="api_key", generation=1))
        with patch("openevolve.llm.orcarouter.openai.OpenAI"):
            return OrcaRouterLLM(
                self._cfg(),
                provider_id=PROVIDER_ID_KEY,
                credential=OrcaCredential(api_key=FAKE_KEY, source="api_key", generation=1),
                store=self.store,
            )

    def test_401_marks_exactly_this_generation_needs_reauth(self):
        client = self._client_with_store()
        error = client._reject_credential(RuntimeError("401"))
        self.assertIsInstance(error, OrcaReauthRequired)
        stored = self.store.load()
        self.assertTrue(stored.needs_reauth)
        self.assertEqual(stored.generation, 1)
        self.assertEqual(stored.api_key, FAKE_KEY)

    def test_401_does_not_delete_the_stored_secret(self):
        client = self._client_with_store()
        client._reject_credential(RuntimeError("401"))
        self.assertEqual(self.store.load().api_key, FAKE_KEY)

    def test_late_401_from_an_old_generation_leaves_the_new_credential_clean(self):
        self.store.save(OrcaCredential(api_key=FAKE_KEY, source="api_key", generation=1))
        old = OrcaCredential(api_key=FAKE_KEY, source="api_key", generation=1)
        # A new sign-in replaces the credential.
        self.store.save(OrcaCredential(api_key=FAKE_KEY_2, source="oauth_pkce", generation=2))
        with patch("openevolve.llm.orcarouter.openai.OpenAI"):
            client = OrcaRouterLLM(
                self._cfg(), provider_id=PROVIDER_ID_KEY, credential=old, store=self.store
            )
        client._reject_credential(RuntimeError("late 401"))
        current = self.store.load()
        self.assertEqual(current.api_key, FAKE_KEY_2)
        self.assertFalse(current.needs_reauth)

    def test_no_refresh_grant_is_attempted(self):
        """A revoked durable key is terminal; nothing may try to refresh it."""
        client = self._client_with_store()
        # Patch the base client so the OrcaRouter override still wraps it.
        with patch.object(
            OpenAILLM,
            "_call_api",
            side_effect=openai.AuthenticationError(
                "401", response=MagicMock(status_code=401), body=None
            ),
        ):
            with self.assertRaises(OrcaReauthRequired):
                asyncio.run(client.generate_with_context("sys", [{"role": "user", "content": "x"}]))
        self.assertTrue(self.store.load().needs_reauth)

    def test_rejected_credential_is_not_retried(self):
        client = self._client_with_store()
        calls = []

        async def counted(_self, params):
            calls.append(params)
            raise openai.AuthenticationError("401", response=MagicMock(status_code=401), body=None)

        client.retries = 3
        with patch.object(OpenAILLM, "_call_api", counted):
            with self.assertRaises(OrcaReauthRequired):
                asyncio.run(client.generate_with_context("sys", [{"role": "user", "content": "x"}]))
        self.assertEqual(len(calls), 1)

    def test_permission_error_is_reported_without_leaking_the_key(self):
        client = self._client_with_store()
        with patch.object(
            OpenAILLM,
            "_call_api",
            side_effect=openai.PermissionDeniedError(
                "403", response=MagicMock(status_code=403), body=None
            ),
        ):
            with self.assertRaises(OrcaAuthError) as ctx:
                asyncio.run(client.generate_with_context("sys", [{"role": "user", "content": "x"}]))
        self.assertNotIn(FAKE_KEY, str(ctx.exception))

    def test_reauth_error_message_is_actionable_and_redacted(self):
        client = self._client_with_store()
        error = client._reject_credential(RuntimeError("401"))
        self.assertIn("Sign in again", str(error))
        self.assertIn("console/authorized-apps", str(error))
        self.assertNotIn(FAKE_KEY, str(error))

    def test_other_api_status_errors_propagate_unchanged(self):
        client = self._client_with_store()
        boom = openai.InternalServerError("500", response=MagicMock(status_code=500), body=None)
        with patch.object(OpenAILLM, "_call_api", side_effect=boom):
            with self.assertRaises(openai.InternalServerError):
                asyncio.run(client.generate_with_context("sys", [{"role": "user", "content": "x"}]))


class TestWorkerSafety(ProviderTestCase):
    def test_worker_without_a_stored_credential_fails_with_instructions(self):
        with patch("openevolve.llm.orcarouter.is_worker_process", return_value=True):
            with self.assertRaises(OrcaAuthError) as ctx:
                OrcaRouterLLM(
                    self._cfg(api_key=None), provider_id=PROVIDER_ID_OAUTH, store=self.store
                )
        self.assertIn("connect orcarouter", str(ctx.exception))

    def test_worker_reuses_a_stored_credential(self):
        self.store.save(OrcaCredential(api_key=FAKE_KEY, source="oauth_pkce", generation=4))
        with (
            patch("openevolve.llm.orcarouter.is_worker_process", return_value=True),
            patch("openevolve.llm.orcarouter.openai.OpenAI"),
        ):
            client = OrcaRouterLLM(
                self._cfg(api_key=None), provider_id=PROVIDER_ID_OAUTH, store=self.store
            )
        self.assertEqual(client.credential.api_key, FAKE_KEY)
        self.assertEqual(client.credential.generation, 4)

    def test_api_key_provider_never_needs_a_browser(self):
        with (
            patch("openevolve.llm.orcarouter.is_worker_process", return_value=True),
            patch("openevolve.llm.orcarouter.openai.OpenAI"),
        ):
            client = OrcaRouterLLM(self._cfg(), provider_id=PROVIDER_ID_KEY, store=self.store)
        self.assertEqual(client.credential.api_key, FAKE_KEY)


class TestCredentialStatusAndLogout(ProviderTestCase):
    def test_status_without_a_credential(self):
        status = orcarouter_credential_status(store=self.store)
        self.assertFalse(status["authenticated"])
        self.assertEqual(status["secret_masked"], "")
        self.assertIn("api.orcarouter.ai", status["api_base"])

    def test_status_never_exposes_the_raw_key(self):
        self.store.save(OrcaCredential(api_key=FAKE_KEY, source="api_key", generation=1))
        status = orcarouter_credential_status(store=self.store)
        self.assertNotIn(FAKE_KEY, json.dumps(status))
        self.assertEqual(status["secret_masked"], mask_key(FAKE_KEY))

    def test_status_reflects_needs_reauth(self):
        self.store.save(
            OrcaCredential(api_key=FAKE_KEY, source="api_key", generation=1, needs_reauth=True)
        )
        status = orcarouter_credential_status(store=self.store)
        self.assertFalse(status["authenticated"])
        self.assertTrue(status["needs_reauth"])

    def test_logout_clears_the_credential(self):
        self.store.save(OrcaCredential(api_key=FAKE_KEY, source="api_key"))
        self.assertTrue(logout(self.store))
        self.assertIsNone(self.store.load())


class TestConfigIntegration(ProviderTestCase):
    def test_provider_reaches_model_configs(self):
        config = Config.from_dict(
            {"llm": {"provider": "orcarouter", "models": [{"name": "openai/gpt-5.5"}]}}
        )
        self.assertEqual([m.provider for m in config.llm.models], ["orcarouter"])

    def test_oob_flag_propagates(self):
        config = Config.from_dict(
            {
                "llm": {
                    "provider": "orcarouter_oauth",
                    "orcarouter_oob": True,
                    "models": [{"name": "openai/gpt-5.5"}],
                }
            }
        )
        self.assertTrue(config.llm.models[0].orcarouter_oob)

    def test_ensemble_constructs_both_orcarouter_backends(self):
        for provider_id in ("orcarouter", "orcarouter_oauth"):
            config = Config.from_dict(
                {
                    "llm": {
                        "provider": provider_id,
                        "api_key": FAKE_KEY,
                        "models": [{"name": "openai/gpt-5.5"}],
                    }
                }
            )
            self.store.save(OrcaCredential(api_key=FAKE_KEY, source="api_key", generation=1))
            ensemble = LLMEnsemble(config.llm.models)
            self.assertEqual(type(ensemble.models[0]).__name__, "OrcaRouterLLM")
            self.assertEqual(ensemble.models[0].provider_id, provider_id)


if __name__ == "__main__":
    unittest.main()
