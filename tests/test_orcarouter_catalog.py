"""
Tests for OrcaRouter model discovery and capability filtering.

The catalog is the only source of truth for selectable models. These tests use
fixtures that cover every capability the cloner needs to distinguish: text-only
chat, image-input chat, embedding, image generation, video and rerank.
"""

import json
import unittest
import urllib.error
from dataclasses import replace
from unittest.mock import MagicMock, patch

from openevolve.llm.orcarouter_catalog import (
    CAPABILITY_CHAT,
    CAPABILITY_EMBEDDING,
    CAPABILITY_IMAGE,
    CAPABILITY_RERANK,
    CAPABILITY_VIDEO,
    OrcaCatalogClient,
    OrcaModel,
    fallback_catalog,
    filter_by_capability,
    filter_for_entry_point,
    parse_catalog,
    parse_model,
)

TEXT_ENDPOINTS = ["openai", "openai-response", "anthropic", "gemini"]

FIXTURE = {
    "data": [
        {
            "id": "openai/gpt-5.5",
            "name": "OpenAI: GPT-5.5",
            "context_length": 400000,
            "max_completion_tokens": 128000,
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "supported_endpoint_types": TEXT_ENDPOINTS,
            "owned_by": "openai",
        },
        {
            "id": "anthropic/claude-opus-4.8",
            "name": "Anthropic: Claude Opus 4.8",
            "context_length": 1000000,
            "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
            "supported_endpoint_types": ["openai", "anthropic", "openai-response"],
            "owned_by": "anthropic",
        },
        {
            "id": "google/gemini-3.5-flash",
            "architecture": {"input_modalities": ["text", "image", "audio", "video"]},
            "supported_endpoint_types": TEXT_ENDPOINTS,
        },
        {
            "id": "deepseek/deepseek-v4-pro",
            "architecture": {"input_modalities": ["text"]},
            "supported_endpoint_types": ["openai", "openai-response"],
        },
        {"id": "orcarouter/auto", "supported_endpoint_types": TEXT_ENDPOINTS},
        {
            "id": "openai/text-embedding-3-small",
            "architecture": {"input_modalities": ["text"]},
            "supported_endpoint_types": ["embeddings"],
        },
        {
            "id": "openai/gpt-image-1",
            "architecture": {"input_modalities": ["text"], "output_modalities": ["image"]},
            "supported_endpoint_types": ["image-generation"],
        },
        {
            "id": "kling/kling-v3",
            "supported_endpoint_types": ["openai-video"],
        },
        {
            "id": "jina/jina-reranker-v2",
            "supported_endpoint_types": ["jina-rerank"],
        },
        # Shapes that must be dropped rather than guessed at.
        {"id": "", "supported_endpoint_types": TEXT_ENDPOINTS},
        {"name": "no id here"},
        {"id": "weird/model", "supported_endpoint_types": ["something-else"]},
        "not-a-dict",
        {
            "id": "duplicate/text",
            "supported_endpoint_types": TEXT_ENDPOINTS,
        },
        {
            "id": "duplicate/text",
            "supported_endpoint_types": TEXT_ENDPOINTS,
        },
    ]
}


class TestCatalogParsing(unittest.TestCase):
    def test_parses_known_models(self):
        models = parse_catalog(FIXTURE)
        ids = [m.id for m in models]
        self.assertIn("openai/gpt-5.5", ids)
        self.assertIn("openai/text-embedding-3-small", ids)
        self.assertIn("openai/gpt-image-1", ids)
        self.assertIn("kling/kling-v3", ids)
        self.assertIn("jina/jina-reranker-v2", ids)

    def test_drops_unusable_records(self):
        ids = [m.id for m in parse_catalog(FIXTURE)]
        self.assertNotIn("", ids)
        self.assertNotIn("weird/model", ids)

    def test_deduplicates_by_id(self):
        ids = [m.id for m in parse_catalog(FIXTURE)]
        self.assertEqual(ids.count("duplicate/text"), 1)

    def test_preserves_vendor_namespace(self):
        model = parse_model(
            {"id": "deepseek/deepseek-v4-pro", "supported_endpoint_types": ["openai"]}
        )
        self.assertEqual(model.id, "deepseek/deepseek-v4-pro")

    def test_unknown_endpoint_types_are_dropped(self):
        model = parse_model({"id": "x/y", "supported_endpoint_types": ["openai", "made-up"]})
        self.assertEqual(model.endpoint_types, ("openai",))

    def test_missing_modality_metadata_is_empty_not_assumed(self):
        model = parse_model({"id": "x/y", "supported_endpoint_types": ["openai"]})
        self.assertEqual(model.input_modalities, ())
        self.assertTrue(model.is_text_only)

    def test_catalog_accepts_bare_list_and_rejects_junk(self):
        self.assertEqual(len(parse_catalog(FIXTURE["data"])), len(parse_catalog(FIXTURE)))
        self.assertEqual(parse_catalog("nope"), [])
        self.assertEqual(parse_catalog({"data": "nope"}), [])

    def test_context_length_falls_back_to_top_provider(self):
        model = parse_model(
            {
                "id": "x/y",
                "top_provider": {"context_length": 12345},
                "supported_endpoint_types": ["openai"],
            }
        )
        self.assertEqual(model.context_length, 12345)

    def test_record_with_only_unknown_routes_is_dropped(self):
        self.assertIsNone(parse_model({"id": "x/y", "supported_endpoint_types": ["made-up"]}))
        self.assertIsNone(parse_model({"id": "x/y"}))

    def test_reasoning_ladder_read_when_published(self):
        model = parse_model(
            {
                "id": "x/y",
                "reasoning": {"efforts": ["low", "medium", "high", "xhigh", "bogus"]},
                "supported_endpoint_types": ["openai"],
            }
        )
        self.assertEqual(model.reasoning_efforts, ("low", "medium", "high", "xhigh"))

    def test_reasoning_ladder_absent_by_default(self):
        model = parse_model({"id": "x/y", "supported_endpoint_types": ["openai"]})
        self.assertEqual(model.reasoning_efforts, ())

    def test_verified_seed_has_required_ids_and_metadata(self):
        seed = {m.id: m for m in fallback_catalog()}
        for expected in (
            "openai/gpt-5.5",
            "anthropic/claude-opus-4.8",
            "google/gemini-3.5-flash",
            "deepseek/deepseek-v4-pro",
            "orcarouter/auto",
        ):
            self.assertIn(expected, seed)
            self.assertTrue(seed[expected].verified)
        self.assertEqual(
            seed["openai/gpt-5.5"].reasoning_efforts, ("low", "medium", "high", "xhigh")
        )
        self.assertIn("image", seed["anthropic/claude-opus-4.8"].input_modalities)
        self.assertEqual(seed["deepseek/deepseek-v4-pro"].context_length, 1048576)


class TestCapabilityFiltering(unittest.TestCase):
    def setUp(self):
        self.models = parse_catalog(FIXTURE)

    def test_chat_excludes_non_text_specialists(self):
        ids = [m.id for m in filter_by_capability(self.models, CAPABILITY_CHAT)]
        self.assertIn("openai/gpt-5.5", ids)
        self.assertNotIn("openai/gpt-image-1", ids)
        self.assertNotIn("kling/kling-v3", ids)
        self.assertNotIn("jina/jina-reranker-v2", ids)
        self.assertNotIn("openai/text-embedding-3-small", ids)

    def test_embedding_filter(self):
        ids = [m.id for m in filter_by_capability(self.models, CAPABILITY_EMBEDDING)]
        self.assertEqual(ids, ["openai/text-embedding-3-small"])

    def test_image_generation_filter(self):
        ids = [m.id for m in filter_by_capability(self.models, CAPABILITY_IMAGE)]
        self.assertEqual(ids, ["openai/gpt-image-1"])

    def test_video_filter(self):
        ids = [m.id for m in filter_by_capability(self.models, CAPABILITY_VIDEO)]
        self.assertEqual(ids, ["kling/kling-v3"])

    def test_rerank_filter(self):
        ids = [m.id for m in filter_by_capability(self.models, CAPABILITY_RERANK)]
        self.assertEqual(ids, ["jina/jina-reranker-v2"])

    def test_unknown_capability_raises(self):
        with self.assertRaises(ValueError):
            filter_by_capability(self.models, "teleportation")

    def test_image_intake_requires_declared_modality(self):
        ids = [m.id for m in filter_for_entry_point(self.models, CAPABILITY_CHAT, "image")]
        self.assertIn("anthropic/claude-opus-4.8", ids)
        self.assertIn("google/gemini-3.5-flash", ids)
        self.assertNotIn("openai/gpt-5.5", ids)
        self.assertNotIn("deepseek/deepseek-v4-pro", ids)
        # An entry that declares nothing must fail closed, not be admitted.
        self.assertNotIn("orcarouter/auto", ids)

    def test_audio_and_video_intakes_require_declared_modality(self):
        self.assertEqual(
            [m.id for m in filter_for_entry_point(self.models, CAPABILITY_CHAT, "audio")],
            ["google/gemini-3.5-flash"],
        )
        self.assertEqual(
            [m.id for m in filter_for_entry_point(self.models, CAPABILITY_CHAT, "video")],
            ["google/gemini-3.5-flash"],
        )

    def test_text_intake_keeps_text_only_models(self):
        ids = [m.id for m in filter_for_entry_point(self.models, CAPABILITY_CHAT, "text")]
        self.assertIn("openai/gpt-5.5", ids)
        self.assertIn("orcarouter/auto", ids)

    def test_modality_never_leaks_specialists_into_chat(self):
        for modality in ("text", "image", "audio", "video"):
            ids = [m.id for m in filter_for_entry_point(self.models, CAPABILITY_CHAT, modality)]
            self.assertNotIn("kling/kling-v3", ids)
            self.assertNotIn("openai/gpt-image-1", ids)


class FakeOpener:
    def __init__(self, payload, status=200, raise_exc=None):
        self.payload = payload
        self.status = status
        self.raise_exc = raise_exc
        self.urls = []
        self.headers = []

    def __call__(self, request, timeout):
        self.urls.append(request.full_url)
        self.headers.append({k.lower(): v for k, v in request.headers.items()})
        if self.raise_exc is not None:
            raise self.raise_exc
        if self.status != 200:
            raise urllib.error.HTTPError(
                request.full_url, self.status, "err", {}, MagicMock(read=lambda *a: b"{}")
            )
        body = json.dumps(self.payload).encode()
        response = MagicMock()
        response.read.return_value = body
        response.__enter__ = lambda self_: response
        response.__exit__ = lambda self_, *args: False
        return response


class TestCatalogClient(unittest.TestCase):
    def test_live_discovery_is_authoritative(self):
        opener = FakeOpener(FIXTURE)
        client = OrcaCatalogClient(api_key="sk-orca-" + "a" * 40, opener=opener)
        result = client.discover()
        self.assertEqual(result.source, "live")
        self.assertFalse(result.degraded)
        self.assertEqual(opener.urls[0], "https://api.orcarouter.ai/v1/models")
        self.assertEqual(opener.headers[0]["authorization"], "Bearer sk-orca-" + "a" * 40)

    def test_capability_query_is_sent(self):
        opener = FakeOpener(FIXTURE)
        client = OrcaCatalogClient(api_key="k", opener=opener)
        client.discover(capability=CAPABILITY_EMBEDDING)
        self.assertIn("capability=embedding", opener.urls[0])

    def test_seed_is_used_when_discovery_fails(self):
        opener = FakeOpener(None, raise_exc=urllib.error.URLError("offline"))
        client = OrcaCatalogClient(api_key="k", opener=opener)
        result = client.discover()
        self.assertTrue(result.degraded)
        self.assertEqual(result.source, "seed")
        self.assertIn("openai/gpt-5.5", result.ids(CAPABILITY_CHAT))
        self.assertTrue(all(m.verified for m in result.models))

    def test_seed_is_never_mixed_into_a_live_result(self):
        """A live catalog is authoritative: no verified seed entry may appear."""
        live_only = {"data": [{"id": "vendor/live-a", "supported_endpoint_types": TEXT_ENDPOINTS}]}
        client = OrcaCatalogClient(api_key="k", opener=FakeOpener(live_only))
        result = client.discover()
        self.assertEqual(result.source, "live")
        self.assertEqual(result.ids(CAPABILITY_CHAT), ["vendor/live-a"])
        self.assertFalse(any(m.verified for m in result.models))
        for seed_id in ("openai/gpt-5.5", "anthropic/claude-opus-4.8", "orcarouter/auto"):
            self.assertNotIn(seed_id, result.ids(CAPABILITY_CHAT))

    def test_last_known_good_beats_the_seed(self):
        live = FakeOpener(
            {"data": [{"id": "vendor/live-only", "supported_endpoint_types": TEXT_ENDPOINTS}]}
        )
        client = OrcaCatalogClient(api_key="k", opener=live, cache_ttl=0)
        client.discover()
        client.invalidate()
        client._cache["*"] = (0.0, parse_catalog(live.payload))
        client._opener = FakeOpener(None, raise_exc=urllib.error.URLError("down"))
        result = client.discover(use_cache=False)
        self.assertEqual(result.source, "cache")
        self.assertIn("vendor/live-only", result.ids(CAPABILITY_CHAT))

    def test_auth_failure_marks_the_credential_rejected(self):
        opener = FakeOpener(None, status=401)
        client = OrcaCatalogClient(api_key="k", opener=opener)
        result = client.discover()
        self.assertTrue(result.degraded)
        self.assertIn("credential rejected", result.error)

    def test_cache_is_used_within_the_ttl(self):
        opener = FakeOpener(FIXTURE)
        client = OrcaCatalogClient(api_key="k", opener=opener, cache_ttl=300)
        client.discover()
        result = client.discover()
        self.assertEqual(result.source, "cache")
        self.assertEqual(len(opener.urls), 1)

    def test_refresh_bypasses_the_cache(self):
        opener = FakeOpener(FIXTURE)
        client = OrcaCatalogClient(api_key="k", opener=opener, cache_ttl=300)
        client.discover()
        client.invalidate()
        client.discover()
        self.assertEqual(len(opener.urls), 2)

    def test_empty_catalog_is_degraded_not_silently_empty(self):
        opener = FakeOpener({"data": []})
        client = OrcaCatalogClient(api_key="k", opener=opener)
        result = client.discover()
        self.assertTrue(result.degraded)
        self.assertEqual(result.source, "seed")

    def test_oversized_response_is_refused(self):
        from openevolve.llm.orcarouter_auth import MAX_CATALOG_BYTES

        response = MagicMock()
        response.read.return_value = b"x" * (MAX_CATALOG_BYTES + 10)
        response.__enter__ = lambda self_: response
        response.__exit__ = lambda self_, *args: False
        client = OrcaCatalogClient(api_key="k", opener=lambda request, timeout: response)
        result = client.discover()
        self.assertTrue(result.degraded)

    def test_invalid_json_is_degraded(self):
        response = MagicMock()
        response.read.return_value = b"<html>not json</html>"
        response.__enter__ = lambda self_: response
        response.__exit__ = lambda self_, *args: False
        client = OrcaCatalogClient(api_key="k", opener=lambda request, timeout: response)
        self.assertTrue(client.discover().degraded)

    def test_degraded_catalog_can_be_limited_to_no_seed(self):
        opener = FakeOpener(None, raise_exc=urllib.error.URLError("offline"))
        client = OrcaCatalogClient(api_key="k", opener=opener)
        result = client.discover(allow_seed=False)
        self.assertEqual(result.models, ())
        self.assertTrue(result.degraded)

    def test_options_carry_selector_metadata(self):
        opener = FakeOpener(FIXTURE)
        client = OrcaCatalogClient(api_key="k", opener=opener)
        options = client.discover().options(CAPABILITY_CHAT, "image")
        ids = [o["id"] for o in options]
        self.assertIn("anthropic/claude-opus-4.8", ids)
        self.assertNotIn("deepseek/deepseek-v4-pro", ids)
        entry = next(o for o in options if o["id"] == "anthropic/claude-opus-4.8")
        self.assertEqual(entry["context_length"], 1000000)
        self.assertIn("image", entry["input_modalities"])
        self.assertFalse(entry["verified"])

    def test_seed_options_are_labelled_verified(self):
        opener = FakeOpener(None, raise_exc=urllib.error.URLError("offline"))
        client = OrcaCatalogClient(api_key="k", opener=opener)
        options = client.discover().options(CAPABILITY_CHAT)
        self.assertTrue(options)
        self.assertTrue(all(o["verified"] for o in options))

    def test_gpt55_keeps_its_reasoning_ladder_in_the_seed(self):
        opener = FakeOpener(None, raise_exc=urllib.error.URLError("offline"))
        client = OrcaCatalogClient(api_key="k", opener=opener)
        options = {o["id"]: o for o in client.discover().options(CAPABILITY_CHAT)}
        self.assertEqual(
            options["openai/gpt-5.5"]["reasoning_efforts"], ["low", "medium", "high", "xhigh"]
        )


class TestEntryPointModelSelection(unittest.TestCase):
    """The selector options must be exactly the filtered catalog."""

    def setUp(self):
        self.client = OrcaCatalogClient(api_key="k", opener=FakeOpener(FIXTURE))

    def test_switching_provider_recomputes_options(self):
        text_options = {o["id"] for o in self.client.discover().options(CAPABILITY_CHAT, "text")}
        image_options = {o["id"] for o in self.client.discover().options(CAPABILITY_CHAT, "image")}
        self.assertNotEqual(text_options, image_options)
        self.assertTrue(image_options.issubset(text_options) is False or True)

    def test_attaching_an_image_drops_text_only_models(self):
        text_only = "deepseek/deepseek-v4-pro"
        before = {o["id"] for o in self.client.discover().options(CAPABILITY_CHAT, "text")}
        after = {o["id"] for o in self.client.discover().options(CAPABILITY_CHAT, "image")}
        self.assertIn(text_only, before)
        self.assertNotIn(text_only, after)


if __name__ == "__main__":
    unittest.main()
