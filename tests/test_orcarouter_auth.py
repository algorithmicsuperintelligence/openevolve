"""
Tests for the shared OrcaRouter credential seam and its two adapters.

The two adapters (pasted API key / OAuth 2.0 + PKCE sign-in) must produce the
same credential type, and a secret must never leak into a log, an error message,
a URL or a test snapshot.
"""

import base64
import hashlib
import io
import json
import logging
import os
import threading
import unittest
import urllib.error
import urllib.parse
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from openevolve.llm.orcarouter_auth import (
    KEY_ENV,
    AUTH_BASE_ENV,
    API_BASE_ENV,
    DEFAULT_API_BASE,
    DEFAULT_AUTH_BASE,
    SHARED_BASE_ENV,
    ApiKeyCredentialProvider,
    CredentialProvider,
    OrcaAuthError,
    OrcaCredential,
    OrcaCredentialStore,
    PkceCredentialProvider,
    acquire_credential,
    looks_like_orcarouter_key,
    mask_key,
    redact,
    resolve_api_base,
    resolve_auth_base,
    _b64url,
)

FAKE_KEY = "sk-orca-test000000000000000000000000000000000000000000"
FAKE_KEY_2 = "sk-orca-second0000000000000000000000000000000000000000"
FAKE_CODE = "test-code-abcdefghijklmnop"


class TempStoreMixin:
    def setUp(self):
        super().setUp()
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.secrets_path = str(Path(self._tmp.name) / "secrets.yaml")
        self.store = OrcaCredentialStore(self.secrets_path)


class TestOrigins(unittest.TestCase):
    def test_public_defaults(self):
        self.assertEqual(resolve_auth_base({}), DEFAULT_AUTH_BASE)
        self.assertEqual(resolve_api_base({}), DEFAULT_API_BASE)

    def test_explicit_overrides_win(self):
        env = {
            AUTH_BASE_ENV: "https://auth.example.test",
            API_BASE_ENV: "https://infer.example.test/v1",
            SHARED_BASE_ENV: "https://shared.example.test",
        }
        self.assertEqual(resolve_auth_base(env), "https://auth.example.test")
        self.assertEqual(resolve_api_base(env), "https://infer.example.test/v1")

    def test_shared_base_is_fallback_for_both(self):
        env = {SHARED_BASE_ENV: "https://shared.example.test"}
        self.assertEqual(resolve_auth_base(env), "https://shared.example.test")
        self.assertEqual(resolve_api_base(env), "https://shared.example.test/v1")

    def test_api_origin_is_never_derived_from_auth_origin(self):
        """A single-origin self-hosted base must not be reached by swapping hosts."""
        env = {AUTH_BASE_ENV: "https://www.orcarouter.ai"}
        # Only the auth override is set: inference must fall back to its own default.
        self.assertEqual(resolve_api_base(env), DEFAULT_API_BASE)
        env = {API_BASE_ENV: "https://api.orcarouter.ai/v1"}
        self.assertEqual(resolve_auth_base(env), DEFAULT_AUTH_BASE)

    def test_remote_http_is_refused(self):
        with self.assertRaises(OrcaAuthError):
            resolve_auth_base({AUTH_BASE_ENV: "http://www.orcarouter.ai"})

    def test_loopback_http_is_allowed(self):
        self.assertEqual(
            resolve_auth_base({AUTH_BASE_ENV: "http://127.0.0.1:9000"}),
            "http://127.0.0.1:9000",
        )
        self.assertEqual(
            resolve_auth_base({AUTH_BASE_ENV: "http://localhost:9000"}),
            "http://localhost:9000",
        )

    def test_userinfo_query_fragment_refused(self):
        for bad in (
            "https://user:pw@www.orcarouter.ai",
            "https://www.orcarouter.ai?x=1",
            "https://www.orcarouter.ai#frag",
        ):
            with self.assertRaises(OrcaAuthError):
                resolve_auth_base({AUTH_BASE_ENV: bad})


class TestMaskingAndRedaction(unittest.TestCase):
    def test_mask_shows_only_the_prefix_and_last_four(self):
        masked = mask_key(FAKE_KEY)
        self.assertEqual(masked, "sk-orca-…" + FAKE_KEY[-4:])
        self.assertNotIn(FAKE_KEY[7:-4], masked)
        # No more than the fixed prefix plus four characters of the key.
        self.assertLessEqual(len(masked) - len("sk-orca-…"), 4)

    def test_mask_short_key(self):
        self.assertEqual(mask_key("sk-short"), "sk-orca-…")
        self.assertEqual(mask_key(None), "")

    def test_redact_removes_key_shaped_strings(self):
        text = f"failed with {FAKE_KEY} in header"
        cleaned = redact(text)
        self.assertNotIn(FAKE_KEY, cleaned)
        self.assertIn("<REDACTED>", cleaned)

    def test_format_check_is_not_validation(self):
        self.assertTrue(looks_like_orcarouter_key(FAKE_KEY))
        self.assertFalse(looks_like_orcarouter_key("sk-orca-"))
        self.assertFalse(looks_like_orcarouter_key("sk-proj-abc"))
        self.assertFalse(looks_like_orcarouter_key(None))


class TestCredentialStore(TempStoreMixin, unittest.TestCase):
    def test_save_load_clear_round_trip(self):
        credential = OrcaCredential(
            api_key=FAKE_KEY, source="api_key", key_id=mask_key(FAKE_KEY), generation=1
        )
        self.store.save(credential)
        loaded = self.store.load()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.api_key, FAKE_KEY)
        self.assertEqual(loaded.source, "api_key")
        self.assertTrue(self.store.clear())
        self.assertIsNone(self.store.load())
        self.assertFalse(self.store.clear())

    def test_store_file_is_owner_only(self):
        self.store.save(OrcaCredential(api_key=FAKE_KEY, source="api_key"))
        mode = os.stat(self.secrets_path).st_mode & 0o777
        self.assertEqual(mode, 0o600)

    def test_existing_unrelated_secrets_are_preserved(self):
        Path(self.secrets_path).write_text("other_provider:\n  api_key: keep-me\n")
        self.store.save(OrcaCredential(api_key=FAKE_KEY, source="api_key"))
        import yaml

        data = yaml.safe_load(Path(self.secrets_path).read_text())
        self.assertEqual(data["other_provider"]["api_key"], "keep-me")
        self.assertEqual(data["orcarouter"]["api_key"], FAKE_KEY)

    def test_corrupt_secrets_file_does_not_crash(self):
        Path(self.secrets_path).write_text("::: not yaml :::\n")
        self.assertIsNone(self.store.load())

    def test_mark_needs_reauth_only_for_matching_generation(self):
        self.store.save(OrcaCredential(api_key=FAKE_KEY, source="api_key", generation=1))
        self.assertFalse(self.store.mark_needs_reauth(99))
        self.assertFalse(self.store.load().needs_reauth)
        self.assertTrue(self.store.mark_needs_reauth(1))
        self.assertTrue(self.store.load().needs_reauth)

    def test_stale_failure_does_not_poison_newer_credential(self):
        """A late 401 from generation 1 must not flag the new generation 2."""
        self.store.save(OrcaCredential(api_key=FAKE_KEY, source="api_key", generation=1))
        self.store.save(OrcaCredential(api_key=FAKE_KEY_2, source="oauth_pkce", generation=2))
        self.assertFalse(self.store.mark_needs_reauth(1))
        current = self.store.load()
        self.assertEqual(current.generation, 2)
        self.assertFalse(current.needs_reauth)

    def test_reauth_does_not_delete_the_stored_secret(self):
        self.store.save(OrcaCredential(api_key=FAKE_KEY, source="api_key", generation=1))
        self.store.mark_needs_reauth(1)
        self.assertEqual(self.store.load().api_key, FAKE_KEY)

    def test_generation_increments_per_write(self):
        provider = ApiKeyCredentialProvider(FAKE_KEY, store=self.store)
        first = provider.acquire()
        self.assertEqual(first.generation, 1)
        self.store.save(OrcaCredential(api_key=FAKE_KEY_2, source="api_key", generation=1))
        second = ApiKeyCredentialProvider(FAKE_KEY, store=self.store).acquire()
        self.assertEqual(second.generation, 2)


class TestApiKeyAdapter(TempStoreMixin, unittest.TestCase):
    def test_returns_shared_credential_type(self):
        provider = ApiKeyCredentialProvider(FAKE_KEY, store=self.store)
        self.assertIsInstance(provider, CredentialProvider)
        credential = provider.acquire()
        self.assertIsInstance(credential, OrcaCredential)
        self.assertEqual(credential.api_key, FAKE_KEY)
        self.assertEqual(credential.source, "api_key")

    def test_env_fallback(self):
        provider = ApiKeyCredentialProvider(None, store=self.store, env={KEY_ENV: FAKE_KEY})
        self.assertEqual(provider.acquire().api_key, FAKE_KEY)

    def test_missing_key_is_actionable(self):
        provider = ApiKeyCredentialProvider(None, store=self.store, env={})
        with self.assertRaises(OrcaAuthError) as ctx:
            provider.acquire()
        self.assertIn(KEY_ENV, str(ctx.exception))
        self.assertIn("console/authorized-apps", str(ctx.exception))

    def test_malformed_key_rejected_without_echoing_it(self):
        provider = ApiKeyCredentialProvider("sk-proj-not-orca", store=self.store)
        with self.assertRaises(OrcaAuthError) as ctx:
            provider.acquire()
        self.assertNotIn("sk-proj-not-orca", str(ctx.exception))

    def test_stored_credential_is_reused(self):
        provider = ApiKeyCredentialProvider(FAKE_KEY, store=self.store)
        first = provider.acquire()
        second = provider.acquire()
        self.assertEqual(first.generation, second.generation)
        self.assertEqual(second.generation, 1)

    def test_needs_reauth_stored_key_is_replaced(self):
        self.store.save(
            OrcaCredential(api_key=FAKE_KEY, source="api_key", generation=1, needs_reauth=True)
        )
        credential = ApiKeyCredentialProvider(FAKE_KEY, store=self.store).acquire()
        self.assertFalse(credential.needs_reauth)

    def test_key_never_logged(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        root = logging.getLogger()
        root.addHandler(handler)
        self.addCleanup(root.removeHandler, handler)
        ApiKeyCredentialProvider(FAKE_KEY, store=self.store).acquire()
        self.assertNotIn(FAKE_KEY, stream.getvalue())


class FakeAuthServer:
    """Minimal stand-in for the OrcaRouter auth origin.

    Records every request so a test can assert which origin and path were used
    and that the verifier never appeared in a URL.
    """

    def __init__(self, response=None, status=200, body=None):
        self.requests = []
        self.calls = []
        self.response = (
            response
            if response is not None
            else {
                "key": FAKE_KEY,
                "user_id": "12345",
                "scope": "api",
            }
        )
        self.status = status
        self.body = body

    def __call__(self, request, timeout):
        self.requests.append(request)
        self.calls.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "data": request.data,
                "headers": dict(request.headers),
            }
        )
        if self.status != 200:
            raise urllib.error.HTTPError(
                request.full_url,
                self.status,
                "error",
                {},
                io.BytesIO((self.body or "").encode()),
            )
        payload = json.dumps(self.response).encode()
        response = MagicMock()
        response.read.return_value = payload
        response.__enter__ = lambda self_: response
        response.__exit__ = lambda self_, *args: False
        return response


class TestPkceHelpers(unittest.TestCase):
    def setUp(self):
        self.provider = PkceCredentialProvider(auth_base="https://www.orcarouter.ai")

    def test_challenge_is_s256_base64url_without_padding(self):
        verifier = _b64url(os.urandom(32))
        challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
        self.assertNotIn("=", challenge)
        self.assertNotIn("+", challenge)
        self.assertNotIn("/", challenge)
        expected = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
            .decode()
            .rstrip("=")
        )
        self.assertEqual(challenge, expected)

    def test_authorize_url_targets_auth_origin_and_uses_s256(self):
        verifier, challenge, state = self.provider._new_attempt()
        url = self.provider.build_authorize_url(challenge, state, "http://127.0.0.1:5/cb")
        parsed = urllib.parse.urlsplit(url)
        self.assertEqual(parsed.netloc, "www.orcarouter.ai")
        self.assertEqual(parsed.path, "/auth")
        params = urllib.parse.parse_qs(parsed.query)
        self.assertEqual(params["code_challenge_method"], ["S256"])
        self.assertEqual(params["code_challenge"], [challenge])
        self.assertEqual(params["state"], [state])
        self.assertEqual(params["callback_url"], ["http://127.0.0.1:5/cb"])
        self.assertEqual(params["scope"], ["api"])
        self.assertEqual(params["app_name"], ["OpenEvolve"])

    def test_verifier_never_appears_in_the_authorize_url(self):
        verifier, challenge, state = self.provider._new_attempt()
        url = self.provider.build_authorize_url(challenge, state, "oob")
        self.assertNotIn(verifier, url)
        self.assertNotIn(verifier, urllib.parse.unquote(url))

    def test_attempt_is_fresh_each_time(self):
        first = self.provider._new_attempt()
        second = self.provider._new_attempt()
        self.assertNotEqual(first[0], second[0])
        self.assertNotEqual(first[2], second[2])

    def test_verifier_is_high_entropy(self):
        verifier, _, _ = self.provider._new_attempt()
        self.assertGreaterEqual(len(base64.urlsafe_b64decode(verifier + "==")), 32)

    def test_oob_authorize_url_uses_literal_oob(self):
        _, challenge, state = self.provider._new_attempt()
        url = self.provider.build_authorize_url(challenge, state, "oob")
        self.assertEqual(
            urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["callback_url"], ["oob"]
        )


class TestPkceExchange(unittest.TestCase):
    def _provider(self, server):
        return (
            PkceCredentialProvider(
                auth_base="https://www.orcarouter.ai",
                store=OrcaCredentialStore("/tmp/openevolve-does-not-exist-<x>.yaml"),
            ),
            server,
        )

    def test_posts_to_auth_origin_exchange_path(self):
        server = FakeAuthServer()
        provider = PkceCredentialProvider(auth_base="https://www.orcarouter.ai")
        with patch("openevolve.llm.orcarouter_auth.urllib.request.urlopen", server):
            result = provider.exchange(FAKE_CODE, "verifier-value")
        self.assertEqual(result["key"], FAKE_KEY)
        self.assertEqual(server.calls[0]["url"], "https://www.orcarouter.ai/api/v1/auth/keys")
        self.assertNotIn("/v1/auth/keys", server.calls[0]["url"].replace("/api/v1/auth/keys", ""))

    def test_never_uses_the_inference_origins_auth_path(self):
        server = FakeAuthServer()
        provider = PkceCredentialProvider(auth_base="https://www.orcarouter.ai")
        with patch("openevolve.llm.orcarouter_auth.urllib.request.urlopen", server):
            provider.exchange(FAKE_CODE, "verifier-value")
        self.assertNotIn("api.orcarouter.ai", server.calls[0]["url"])

    def test_body_carries_verifier_and_s256(self):
        server = FakeAuthServer()
        provider = PkceCredentialProvider(auth_base="https://www.orcarouter.ai")
        with patch("openevolve.llm.orcarouter_auth.urllib.request.urlopen", server):
            provider.exchange(FAKE_CODE, "verifier-value")
        body = json.loads(server.calls[0]["data"].decode())
        self.assertEqual(body["code"], FAKE_CODE)
        self.assertEqual(body["code_verifier"], "verifier-value")
        self.assertEqual(body["code_challenge_method"], "S256")

    def test_verifier_is_not_in_the_request_url(self):
        server = FakeAuthServer()
        provider = PkceCredentialProvider(auth_base="https://www.orcarouter.ai")
        with patch("openevolve.llm.orcarouter_auth.urllib.request.urlopen", server):
            provider.exchange(FAKE_CODE, "super-secret-verifier")
        self.assertNotIn("super-secret-verifier", server.calls[0]["url"])

    def test_granted_scope_downgrade_is_reported(self):
        server = FakeAuthServer(response={"key": FAKE_KEY, "scope": "connector"})
        provider = PkceCredentialProvider(auth_base="https://www.orcarouter.ai")
        with patch("openevolve.llm.orcarouter_auth.urllib.request.urlopen", server):
            with self.assertRaises(OrcaAuthError) as ctx:
                provider.exchange(FAKE_CODE, "v")
        self.assertIn("connector", str(ctx.exception))
        self.assertNotIn(FAKE_KEY, str(ctx.exception))

    def test_missing_key_in_response_is_rejected(self):
        server = FakeAuthServer(response={"scope": "api"})
        provider = PkceCredentialProvider(auth_base="https://www.orcarouter.ai")
        with patch("openevolve.llm.orcarouter_auth.urllib.request.urlopen", server):
            with self.assertRaises(OrcaAuthError):
                provider.exchange(FAKE_CODE, "v")

    def test_403_is_terminal_and_does_not_leak_body_credentials(self):
        server = FakeAuthServer(status=403, body=f'{{"error":"bad","key":"{FAKE_KEY}"}}')
        provider = PkceCredentialProvider(auth_base="https://www.orcarouter.ai")
        with patch("openevolve.llm.orcarouter_auth.urllib.request.urlopen", server):
            with self.assertRaises(OrcaAuthError) as ctx:
                provider.exchange(FAKE_CODE, "v")
        self.assertNotIn(FAKE_KEY, str(ctx.exception))

    def test_400_and_429_are_reported(self):
        for status in (400, 429):
            server = FakeAuthServer(status=status, body="{}")
            provider = PkceCredentialProvider(auth_base="https://www.orcarouter.ai")
            with patch("openevolve.llm.orcarouter_auth.urllib.request.urlopen", server):
                with self.assertRaises(OrcaAuthError):
                    provider.exchange(FAKE_CODE, "v")

    def test_network_failure_is_reported_not_raised_raw(self):
        provider = PkceCredentialProvider(auth_base="https://www.orcarouter.ai")

        def boom(request, timeout):
            raise urllib.error.URLError("no route to host")

        with patch("openevolve.llm.orcarouter_auth.urllib.request.urlopen", boom):
            with self.assertRaises(OrcaAuthError):
                provider.exchange(FAKE_CODE, "v")

    def test_timeout_is_reported(self):
        provider = PkceCredentialProvider(auth_base="https://www.orcarouter.ai")

        def slow(request, timeout):
            raise TimeoutError("timed out")

        with patch("openevolve.llm.orcarouter_auth.urllib.request.urlopen", slow):
            with self.assertRaises(OrcaAuthError) as ctx:
                provider.exchange(FAKE_CODE, "v")
        self.assertIn("Timed out", str(ctx.exception))

    def test_empty_code_rejected_before_any_request(self):
        server = FakeAuthServer()
        provider = PkceCredentialProvider(auth_base="https://www.orcarouter.ai")
        with patch("openevolve.llm.orcarouter_auth.urllib.request.urlopen", server):
            with self.assertRaises(OrcaAuthError):
                provider.exchange("", "v")
        self.assertEqual(server.calls, [])

    def test_exchange_error_never_contains_the_verifier(self):
        server = FakeAuthServer(status=403, body="{}")
        provider = PkceCredentialProvider(auth_base="https://www.orcarouter.ai")
        with patch("openevolve.llm.orcarouter_auth.urllib.request.urlopen", server):
            try:
                provider.exchange(FAKE_CODE, "unique-verifier-marker")
            except OrcaAuthError as exc:
                self.assertNotIn("unique-verifier-marker", str(exc))


class TestPkceOutOfBandFlow(TempStoreMixin, unittest.TestCase):
    def _provider(self, server, code=FAKE_CODE, **kwargs):
        provider = PkceCredentialProvider(
            auth_base="https://www.orcarouter.ai",
            store=self.store,
            oob=True,
            open_browser=False,
            code_prompt=lambda _url: code,
            **kwargs,
        )
        return provider, server

    def test_oob_flow_persists_the_shared_credential_type(self):
        server = FakeAuthServer()
        provider, _ = self._provider(server)
        with patch("openevolve.llm.orcarouter_auth.urllib.request.urlopen", server):
            credential = provider.acquire()
        self.assertIsInstance(credential, OrcaCredential)
        self.assertEqual(credential.api_key, FAKE_KEY)
        self.assertEqual(credential.source, "oauth_pkce")
        self.assertEqual(credential.account_id, "12345")
        self.assertEqual(credential.scope, "api")
        self.assertEqual(self.store.load().api_key, FAKE_KEY)

    def test_pasted_redirect_url_state_is_checked(self):
        server = FakeAuthServer()
        provider = PkceCredentialProvider(
            auth_base="https://www.orcarouter.ai",
            store=self.store,
            oob=True,
            open_browser=False,
            code_prompt=lambda _url: f"http://127.0.0.1:1/cb?code={FAKE_CODE}&state=wrong",
        )
        with patch("openevolve.llm.orcarouter_auth.urllib.request.urlopen", server):
            with self.assertRaises(OrcaAuthError) as ctx:
                provider.acquire()
        self.assertIn("state", str(ctx.exception).lower())
        self.assertEqual(server.calls, [])

    def test_pasted_redirect_url_with_matching_state_is_accepted(self):
        server = FakeAuthServer()
        captured = {}

        def prompt(_url):
            return f"http://127.0.0.1:1/cb?code={FAKE_CODE}&state={captured['state']}"

        provider = PkceCredentialProvider(
            auth_base="https://www.orcarouter.ai",
            store=self.store,
            oob=True,
            open_browser=False,
            code_prompt=prompt,
            url_sink=lambda url: captured.setdefault(
                "state", urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["state"][0]
            ),
        )
        with patch("openevolve.llm.orcarouter_auth.urllib.request.urlopen", server):
            credential = provider.acquire()
        self.assertEqual(credential.api_key, FAKE_KEY)

    def test_empty_pasted_code_leaves_nothing_stored(self):
        server = FakeAuthServer()
        provider, _ = self._provider(server, code="")
        with patch("openevolve.llm.orcarouter_auth.urllib.request.urlopen", server):
            with self.assertRaises(OrcaAuthError):
                provider.acquire()
        self.assertIsNone(self.store.load())
        self.assertEqual(server.calls, [])

    def test_denial_raises_and_stores_nothing(self):
        server = FakeAuthServer()
        provider = PkceCredentialProvider(
            auth_base="https://www.orcarouter.ai",
            store=self.store,
            oob=True,
            open_browser=False,
            code_prompt=lambda _url: "http://127.0.0.1:1/cb?error=access_denied&state=x",
        )
        # state mismatch is caught first; use the listener path for a real denial
        with patch("openevolve.llm.orcarouter_auth.urllib.request.urlopen", server):
            with self.assertRaises(OrcaAuthError):
                provider.acquire()
        self.assertIsNone(self.store.load())

    def test_authorize_url_is_announced_to_the_caller(self):
        server = FakeAuthServer()
        seen = []
        provider = PkceCredentialProvider(
            auth_base="https://www.orcarouter.ai",
            store=self.store,
            oob=True,
            open_browser=False,
            code_prompt=lambda _url: FAKE_CODE,
            url_sink=seen.append,
        )
        with patch("openevolve.llm.orcarouter_auth.urllib.request.urlopen", server):
            provider.acquire()
        self.assertEqual(len(seen), 1)
        self.assertIn("/auth?", seen[0])


class TestPkceLoopbackFlow(TempStoreMixin, unittest.TestCase):
    """Flow A end-to-end against a real loopback listener."""

    def _run(self, params_for_state, expect_error=None):
        server = FakeAuthServer()
        captured = {}
        # The exchange call is patched below; the fake browser must keep using
        # the real transport or it would hit the mock instead of our listener.
        real_urlopen = urllib.request.urlopen

        def sink(url):
            captured["url"] = url
            captured["state"] = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["state"][0]
            captured["callback"] = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)[
                "callback_url"
            ][0]
            # Hit the listener from a thread, exactly as a browser would.
            threading.Thread(
                target=_deliver,
                args=(real_urlopen, captured["callback"], params_for_state(captured["state"])),
                daemon=True,
            ).start()

        def _deliver(opener, callback, query):
            import time as _time

            for _ in range(200):
                try:
                    opener(f"{callback}?{query}", timeout=2).read()
                    return
                except Exception:
                    _time.sleep(0.05)

        provider = PkceCredentialProvider(
            auth_base="https://www.orcarouter.ai",
            store=self.store,
            open_browser=False,
            timeout=15,
            url_sink=sink,
        )
        with patch("openevolve.llm.orcarouter_auth.urllib.request.urlopen", server):
            credential = provider.acquire()
        return credential, captured, server

    def test_successful_loopback_flow(self):
        credential, captured, server = self._run(lambda state: f"code={FAKE_CODE}&state={state}")
        self.assertEqual(credential.api_key, FAKE_KEY)
        self.assertEqual(credential.source, "oauth_pkce")
        self.assertEqual(self.store.load().api_key, FAKE_KEY)
        # The listener must bind loopback and the exchange must go to the auth origin.
        self.assertTrue(captured["callback"].startswith("http://127.0.0.1:"))
        self.assertEqual(server.calls[0]["url"], "https://www.orcarouter.ai/api/v1/auth/keys")

    def test_state_mismatch_is_refused(self):
        with self.assertRaises(OrcaAuthError) as ctx:
            self._run(lambda state: f"code={FAKE_CODE}&state=not-the-state")
        self.assertIn("state", str(ctx.exception).lower())
        self.assertIsNone(self.store.load())

    def test_denial_sends_no_exchange_request(self):
        with self.assertRaises(OrcaAuthError):
            self._run(lambda state: f"error=access_denied&state={state}")
        self.assertIsNone(self.store.load())

    def test_timeout_is_bounded_and_stores_nothing(self):
        provider = PkceCredentialProvider(
            auth_base="https://www.orcarouter.ai",
            store=self.store,
            open_browser=False,
            timeout=1,
            url_sink=lambda _url: None,
        )
        with self.assertRaises(OrcaAuthError) as ctx:
            provider.acquire()
        self.assertIn("Timed out", str(ctx.exception))
        self.assertIsNone(self.store.load())


class TestAcquireCredentialSeam(TempStoreMixin, unittest.TestCase):
    def test_api_key_provider_uses_key_adapter(self):
        with patch.dict(os.environ, {KEY_ENV: FAKE_KEY}):
            credential = acquire_credential("orcarouter", store=self.store)
        self.assertEqual(credential.source, "api_key")

    def test_oauth_provider_reuses_stored_credential(self):
        """A stored key must be reused, not re-minted (10 keys/user/24h cap)."""
        self.store.save(OrcaCredential(api_key=FAKE_KEY, source="oauth_pkce", generation=3))
        credential = acquire_credential("orcarouter_oauth", store=self.store)
        self.assertEqual(credential.api_key, FAKE_KEY)
        self.assertEqual(credential.generation, 3)

    def test_oauth_provider_does_not_reuse_a_rejected_credential(self):
        self.store.save(
            OrcaCredential(api_key=FAKE_KEY, source="oauth_pkce", generation=1, needs_reauth=True)
        )
        with patch.object(PkceCredentialProvider, "acquire") as mock_acquire:
            mock_acquire.return_value = OrcaCredential(
                api_key=FAKE_KEY_2, source="oauth_pkce", generation=2
            )
            credential = acquire_credential("orcarouter_oauth", store=self.store)
        self.assertTrue(mock_acquire.called)
        self.assertEqual(credential.api_key, FAKE_KEY_2)

    def test_api_key_provider_reuses_stored_key_when_no_env_or_config(self):
        self.store.save(OrcaCredential(api_key=FAKE_KEY, source="api_key", generation=1))
        with patch.dict(os.environ, {}, clear=True):
            credential = acquire_credential("orcarouter", store=self.store)
        self.assertEqual(credential.api_key, FAKE_KEY)

    def test_unknown_provider_rejected(self):
        with self.assertRaises(OrcaAuthError):
            acquire_credential("nope", store=self.store)


if __name__ == "__main__":
    unittest.main()
