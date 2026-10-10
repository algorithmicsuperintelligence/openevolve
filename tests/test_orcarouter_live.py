"""
Live checks against the real OrcaRouter gateway, through the shipped code paths.

These exercise the provider this branch adds -- ``OrcaRouterLLM`` for inference
and ``OrcaCatalogClient`` for model discovery -- rather than a hand-rolled HTTP
call, so a green run means the integration itself works.

The suite is skipped unless ``ORCAROUTER_API_KEY`` is present, which keeps the
default ``unittest discover`` run offline and hermetic. When a key *is* present
these must pass: a live failure is a real failure.
"""

import asyncio
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from openevolve.config import LLMModelConfig  # noqa: E402
from openevolve.llm.orcarouter import (  # noqa: E402
    PROVIDER_ID_KEY,
    PROVIDER_ID_OAUTH,
    OrcaRouterLLM,
    orcarouter_credential_status,
)
from openevolve.llm.orcarouter_auth import (  # noqa: E402
    DEFAULT_API_BASE,
    DEFAULT_AUTH_BASE,
    OrcaCredential,
    OrcaCredentialStore,
    mask_key,
)
from openevolve.llm.orcarouter_catalog import (  # noqa: E402
    CAPABILITY_CHAT,
    CAPABILITY_EMBEDDING,
    MODALITY_IMAGE,
    MODALITY_TEXT,
    OrcaCatalogClient,
    filter_for_entry_point,
)

API_KEY = os.environ.get("ORCAROUTER_API_KEY")
LIVE = unittest.skipUnless(API_KEY, "ORCAROUTER_API_KEY is required for live checks")

#: Smoke target. The live catalog below proves it is offered to this account
#: before the request is sent, so the test never assumes an entitlement.
SMOKE_MODEL = "deepseek/deepseek-v4-pro"


class LiveOriginContractTests(unittest.TestCase):
    """Auth and inference must stay on their own public origins."""

    @LIVE
    def test_the_two_origins_are_distinct_and_https(self):
        self.assertEqual(DEFAULT_AUTH_BASE, "https://www.orcarouter.ai")
        self.assertEqual(DEFAULT_API_BASE, "https://api.orcarouter.ai/v1")
        self.assertNotEqual(DEFAULT_AUTH_BASE, DEFAULT_API_BASE)
        self.assertFalse(DEFAULT_API_BASE.startswith("https://www.orcarouter"))
        self.assertNotIn("/v1/auth", DEFAULT_API_BASE)


class LiveProviderRequestTests(unittest.TestCase):
    """A real completion through the provider this branch adds."""

    def setUp(self):
        # Keep every write inside a scratch secrets file: a live run must never
        # touch the developer's real credential store.
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.secrets_path = str(Path(self._tmp.name) / "secrets.yaml")
        self.env = patch.dict(os.environ, {"OPENEVOLVE_SECRETS_FILE": self.secrets_path})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.store = OrcaCredentialStore(self.secrets_path)
        self.credential = OrcaCredential(api_key=API_KEY, source="api_key")

    def _client(self, provider_id=PROVIDER_ID_KEY, model=SMOKE_MODEL):
        cfg = LLMModelConfig(
            name=model,
            provider=provider_id,
            system_message="You are a terse assistant.",
        )
        with patch("openevolve.llm.orcarouter.OrcaRouterLLM._probe", create=True):
            return OrcaRouterLLM(
                cfg,
                provider_id=provider_id,
                credential=self.credential,
                store=self.store,
            )

    @LIVE
    def test_inference_hits_the_api_origin_and_answers(self):
        client = self._client()
        self.assertEqual(client.api_base, DEFAULT_API_BASE)
        self.assertTrue(client.api_base.startswith("https://"))

        text = asyncio.run(
            client.generate_with_context(
                system_message="You are a terse assistant.",
                messages=[{"role": "user", "content": "Reply with exactly: pong"}],
                max_tokens=16,
            )
        )
        self.assertIsInstance(text, str)
        self.assertTrue(text.strip(), "the gateway returned an empty completion")
        self.assertNotIn(API_KEY, text)

    @LIVE
    def test_status_reports_the_credential_without_disclosing_it(self):
        client = self._client()
        status = client.status()
        self.assertTrue(status["authenticated"])
        self.assertEqual(status["api_base"], DEFAULT_API_BASE)
        self.assertFalse(status["needs_reauth"])
        rendered = repr(status)
        self.assertNotIn(API_KEY, rendered)
        self.assertEqual(status["secret_masked"], mask_key(API_KEY))
        self.assertNotIn(API_KEY[6:-4], rendered)

    @LIVE
    def test_module_level_status_uses_the_same_redaction(self):
        # The on-disk round trip is covered with a fake key in
        # tests/test_orcarouter_auth.py. Repeating it here would write the real
        # key to disk for no extra signal, so this only checks the rendering.
        with patch.object(
            OrcaCredentialStore,
            "load",
            return_value=OrcaCredential(api_key=API_KEY, source="api_key", generation=1),
        ):
            status = orcarouter_credential_status(store=self.store)
        rendered = repr(status)
        self.assertNotIn(API_KEY, rendered)
        self.assertNotIn(API_KEY[6:-4], rendered)
        self.assertEqual(status["secret_masked"], mask_key(API_KEY))
        self.assertTrue(status["authenticated"])

    @LIVE
    def test_both_provider_ids_resolve_to_the_same_gateway(self):
        for provider_id in (PROVIDER_ID_KEY, PROVIDER_ID_OAUTH):
            with self.subTest(provider=provider_id):
                client = self._client(provider_id=provider_id)
                self.assertEqual(client.api_base, DEFAULT_API_BASE)


class LiveCatalogDiscoveryTests(unittest.TestCase):
    """The dropdown is built from the live catalog, filtered by capability."""

    @classmethod
    def setUpClass(cls):
        cls.client = OrcaCatalogClient(api_key=API_KEY)
        cls.result = cls.client.discover()

    @LIVE
    def test_discovery_is_live_and_authoritative(self):
        self.assertEqual(self.result.source, "live")
        self.assertFalse(self.result.degraded)
        self.assertTrue(self.result.models, "the live catalog returned no models")
        self.assertEqual(self.result.api_base, DEFAULT_API_BASE)

    @LIVE
    def test_no_seed_entry_survives_a_successful_live_discovery(self):
        live_ids = {m.id for m in self.result.models}
        for option in self.result.options(CAPABILITY_CHAT, MODALITY_TEXT):
            self.assertIn(option["id"], live_ids)
            self.assertFalse(option.get("verified"), "a seed entry leaked into a live result")

    @LIVE
    def test_model_ids_keep_their_vendor_namespace(self):
        for model in self.result.models:
            self.assertIn("/", model.id, model.id)
            self.assertTrue(model.endpoint_types, model.id)

    @LIVE
    def test_chat_options_are_text_capable_only(self):
        text_ids = self.result.ids(CAPABILITY_CHAT, MODALITY_TEXT)
        self.assertTrue(text_ids)
        by_id = {m.id: m for m in self.result.models}
        for model_id in text_ids:
            model = by_id[model_id]
            self.assertTrue(model.supports_chat, model_id)
            self.assertTrue(model.is_text_only or MODALITY_TEXT in model.input_modalities)
            for forbidden in ("image-generation", "openai-video", "jina-rerank"):
                self.assertNotIn(forbidden, model.endpoint_types, model_id)

    @LIVE
    def test_image_options_declare_image_input(self):
        image_ids = self.result.ids(CAPABILITY_CHAT, MODALITY_IMAGE)
        text_ids = self.result.ids(CAPABILITY_CHAT, MODALITY_TEXT)
        by_id = {m.id: m for m in self.result.models}
        for model_id in image_ids:
            self.assertTrue(by_id[model_id].accepts_modality(MODALITY_IMAGE), model_id)
        # Multimodal is a strict subset of text, never a superset.
        self.assertTrue(set(image_ids).issubset(set(text_ids)))

    @LIVE
    def test_embedding_options_match_the_embeddings_endpoint(self):
        embedding_ids = self.result.ids(CAPABILITY_EMBEDDING)
        by_id = {m.id: m for m in self.result.models}
        for model_id in embedding_ids:
            self.assertIn("embeddings", by_id[model_id].endpoint_types, model_id)

    @LIVE
    def test_every_option_carries_the_metadata_the_filters_need(self):
        for option in self.result.options(CAPABILITY_CHAT, MODALITY_TEXT):
            self.assertIn("input_modalities", option)
            self.assertIn("context_length", option)
            self.assertIn("id", option)
            self.assertNotIn("api_key", option)

    @LIVE
    def test_filters_only_ever_return_catalog_members(self):
        known = {m.id for m in self.result.models}
        for capability in (CAPABILITY_CHAT, CAPABILITY_EMBEDDING):
            for modality in (MODALITY_TEXT, MODALITY_IMAGE):
                selected = filter_for_entry_point(self.result.models, capability, modality)
                self.assertTrue({m.id for m in selected}.issubset(known))

    @LIVE
    def test_an_unknown_capability_fails_closed(self):
        with self.assertRaises(ValueError):
            filter_for_entry_point(self.result.models, "telepathy")


if __name__ == "__main__":
    unittest.main()
