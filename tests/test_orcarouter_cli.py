"""
Tests for the OrcaRouter CLI surface.

Both credential choices must be discoverable from the command line, and the
model list must come from the live catalog rather than a free-form string.
"""

import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from openevolve.cli import (
    _resolve_orcarouter_models,
    main,
    parse_orcarouter_command,
    run_orcarouter_command,
)
from openevolve.llm.orcarouter_auth import OrcaCredential, OrcaCredentialStore
from openevolve.llm.orcarouter_catalog import CatalogResult, parse_catalog

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_orcarouter_catalog import FIXTURE  # noqa: E402

FAKE_KEY = "sk-orca-" + "a" * 44


class CommandRecognitionTests(unittest.TestCase):
    def test_connect_is_recognised(self):
        self.assertEqual(parse_orcarouter_command(["connect"]), ["connect"])
        self.assertEqual(
            parse_orcarouter_command(["connect", "orcarouter"]), ["connect", "orcarouter"]
        )

    def test_prefixed_form_is_recognised(self):
        self.assertEqual(parse_orcarouter_command(["orcarouter", "status"]), ["status"])
        self.assertEqual(
            parse_orcarouter_command(["orcarouter", "connect", "orcarouter_oauth"]),
            ["connect", "orcarouter_oauth"],
        )

    def test_normal_evolution_arguments_are_untouched(self):
        argv = ["initial.py", "eval.py", "--iterations", "5"]
        self.assertIsNone(parse_orcarouter_command(argv))
        self.assertIsNone(parse_orcarouter_command([]))

    def test_main_routes_commands_without_touching_evolution(self):
        with (
            patch("openevolve.cli.run_orcarouter_command", return_value=0) as routed,
            patch("sys.argv", ["openevolve-run.py", "status"]),
        ):
            self.assertEqual(main(), 0)
        routed.assert_called_once_with(["status"])


class CliCommandTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.secrets_path = str(Path(self._tmp.name) / "secrets.yaml")
        self.env = patch.dict(os.environ, {"OPENEVOLVE_SECRETS_FILE": self.secrets_path})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.store = OrcaCredentialStore(self.secrets_path)

    def _run(self, argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = run_orcarouter_command(argv)
        return code, buffer.getvalue()

    def test_status_reports_unauthenticated(self):
        code, out = self._run(["status"])
        self.assertEqual(code, 0)
        self.assertIn("authenticated : False", out)

    def test_status_never_prints_the_raw_key(self):
        self.store.save(OrcaCredential(api_key=FAKE_KEY, source="api_key", generation=1))
        code, out = self._run(["status"])
        self.assertNotIn(FAKE_KEY, out)
        self.assertIn("sk-orca-", out)

    def test_status_json_is_redacted(self):
        self.store.save(OrcaCredential(api_key=FAKE_KEY, source="api_key", generation=1))
        _, out = self._run(["status", "--json"])
        self.assertNotIn(FAKE_KEY, out)

    def test_status_flags_needs_reauth(self):
        self.store.save(
            OrcaCredential(api_key=FAKE_KEY, source="api_key", generation=1, needs_reauth=True)
        )
        _, out = self._run(["status"])
        self.assertIn("ACTION", out)

    def test_logout_removes_the_credential(self):
        self.store.save(OrcaCredential(api_key=FAKE_KEY, source="api_key"))
        code, out = self._run(["logout"])
        self.assertEqual(code, 0)
        self.assertIn("Removed", out)
        self.assertIsNone(self.store.load())

    def test_logout_when_nothing_is_stored(self):
        code, out = self._run(["logout"])
        self.assertEqual(code, 0)
        self.assertIn("No stored", out)

    def test_connect_api_key_from_environment(self):
        with patch.dict(os.environ, {"ORCAROUTER_API_KEY": FAKE_KEY}):
            code, out = self._run(["connect"])
        self.assertEqual(code, 0)
        self.assertNotIn(FAKE_KEY, out)
        self.assertEqual(self.store.load().api_key, FAKE_KEY)

    def test_connect_api_key_missing_is_actionable(self):
        with patch.dict(os.environ, {}, clear=True), patch("sys.stdin") as stdin:
            stdin.isatty.return_value = False
            code, out = self._run(["connect"])
        self.assertEqual(code, 1)
        self.assertIn("ORCAROUTER_API_KEY", out)

    def test_connect_oauth_reuses_a_stored_credential(self):
        self.store.save(OrcaCredential(api_key=FAKE_KEY, source="oauth_pkce", generation=2))
        code, out = self._run(["connect", "orcarouter_oauth"])
        self.assertEqual(code, 0)
        self.assertIn("already stored", out)

    def test_connect_oauth_without_a_stored_credential_starts_pkce(self):
        with patch(
            "openevolve.llm.orcarouter_auth.PkceCredentialProvider.acquire",
            return_value=OrcaCredential(api_key=FAKE_KEY, source="oauth_pkce"),
        ) as acquire:
            code, out = self._run(["connect", "orcarouter_oauth", "--oob"])
        self.assertEqual(code, 0)
        self.assertIn("Signed in", out)
        self.assertTrue(acquire.called)

    def test_connect_oauth_failure_exits_non_zero_without_a_traceback(self):
        from openevolve.llm.orcarouter_auth import OrcaAuthError

        with patch(
            "openevolve.llm.orcarouter_auth.PkceCredentialProvider.acquire",
            side_effect=OrcaAuthError("denied"),
        ):
            code, out = self._run(["connect", "orcarouter_oauth"])
        self.assertEqual(code, 1)
        self.assertIn("denied", out)

    def test_models_command_lists_the_catalog(self):
        result = CatalogResult(
            models=tuple(parse_catalog(FIXTURE)),
            source="live",
            api_base="https://api.orcarouter.ai/v1",
        )
        with patch(
            "openevolve.llm.orcarouter_catalog.OrcaCatalogClient.discover", return_value=result
        ):
            code, out = self._run(["models"])
        self.assertEqual(code, 0)
        self.assertIn("deepseek/deepseek-v4-pro", out)
        self.assertIn("source: live" if "source: live" in out else "live", out)

    def test_models_command_json_reports_filtered_ids(self):
        result = CatalogResult(
            models=tuple(parse_catalog(FIXTURE)),
            source="live",
            api_base="https://api.orcarouter.ai/v1",
        )
        with patch(
            "openevolve.llm.orcarouter_catalog.OrcaCatalogClient.discover", return_value=result
        ):
            _, out = self._run(["models", "--json", "--capability", "chat", "--modality", "image"])
        payload = json.loads(out)
        ids = [m["id"] for m in payload["models"]]
        self.assertIn("anthropic/claude-opus-4.8", ids)
        self.assertNotIn("deepseek/deepseek-v4-pro", ids)

    def test_models_command_surfaces_degraded_catalog(self):
        result = CatalogResult(
            models=tuple(parse_catalog(FIXTURE)),
            source="seed",
            api_base="https://api.orcarouter.ai/v1",
            degraded=True,
            error="network error",
        )
        with patch(
            "openevolve.llm.orcarouter_catalog.OrcaCatalogClient.discover", return_value=result
        ):
            _, out = self._run(["models"])
        self.assertIn("degraded", out)
        self.assertIn("verified fallback", out)


class ResolveModelsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.secrets_path = str(Path(self._tmp.name) / "secrets.yaml")
        self.env = patch.dict(
            os.environ,
            {"OPENEVOLVE_SECRETS_FILE": self.secrets_path, "ORCAROUTER_API_KEY": FAKE_KEY},
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_returns_only_compatible_chat_models(self):
        result = CatalogResult(
            models=tuple(parse_catalog(FIXTURE)),
            source="live",
            api_base="https://api.orcarouter.ai/v1",
        )
        with patch(
            "openevolve.llm.orcarouter_catalog.OrcaCatalogClient.discover", return_value=result
        ):
            configs, error = _resolve_orcarouter_models(None, None, "orcarouter")
        self.assertIsNone(error)
        names = [c.name for c in configs]
        self.assertIn("deepseek/deepseek-v4-pro", names)
        self.assertNotIn("openai/gpt-image-1", names)
        self.assertNotIn("kling/kling-v3", names)

    def test_requested_model_must_be_in_the_catalog(self):
        result = CatalogResult(
            models=tuple(parse_catalog(FIXTURE)),
            source="live",
            api_base="https://api.orcarouter.ai/v1",
        )
        with patch(
            "openevolve.llm.orcarouter_catalog.OrcaCatalogClient.discover", return_value=result
        ):
            configs, error = _resolve_orcarouter_models("vendor/not-real", None, "orcarouter")
        self.assertIsNone(configs)
        self.assertIn("not offered", error)

    def test_requested_incompatible_model_is_refused(self):
        result = CatalogResult(
            models=tuple(parse_catalog(FIXTURE)),
            source="live",
            api_base="https://api.orcarouter.ai/v1",
        )
        with patch(
            "openevolve.llm.orcarouter_catalog.OrcaCatalogClient.discover", return_value=result
        ):
            configs, error = _resolve_orcarouter_models("openai/gpt-image-1", None, "orcarouter")
        self.assertIsNone(configs)
        self.assertIsNotNone(error)

    def test_a_valid_requested_model_is_selected(self):
        result = CatalogResult(
            models=tuple(parse_catalog(FIXTURE)),
            source="live",
            api_base="https://api.orcarouter.ai/v1",
        )
        with patch(
            "openevolve.llm.orcarouter_catalog.OrcaCatalogClient.discover", return_value=result
        ):
            configs, error = _resolve_orcarouter_models(
                "deepseek/deepseek-v4-pro", None, "orcarouter"
            )
        self.assertIsNone(error)
        self.assertEqual([c.name for c in configs], ["deepseek/deepseek-v4-pro"])
        self.assertEqual(configs[0].provider, "orcarouter")

    def test_catalog_failure_falls_back_to_the_verified_seed(self):
        result = CatalogResult(
            models=tuple(parse_catalog(FIXTURE)),
            source="seed",
            api_base="https://api.orcarouter.ai/v1",
            degraded=True,
            error="network error",
        )
        with patch(
            "openevolve.llm.orcarouter_catalog.OrcaCatalogClient.discover", return_value=result
        ):
            configs, error = _resolve_orcarouter_models(None, None, "orcarouter")
        self.assertIsNone(error)
        # The fixture stands in for the seed; every entry still comes from the
        # catalog layer, never from a free-form string.
        self.assertTrue(configs)
        self.assertTrue(all(c.name for c in configs))

    def test_no_credential_is_actionable(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("openevolve.llm.orcarouter_auth.OrcaCredentialStore.load", return_value=None),
        ):
            configs, error = _resolve_orcarouter_models(None, None, "orcarouter")
        self.assertIsNone(configs)
        self.assertIn("connect orcarouter", error)

    def test_oauth_provider_requires_a_stored_credential(self):
        with (
            patch("openevolve.llm.orcarouter_auth.OrcaCredentialStore.load", return_value=None),
            patch.dict(os.environ, {}, clear=True),
        ):
            configs, error = _resolve_orcarouter_models(None, None, "orcarouter_oauth")
        self.assertIsNone(configs)
        self.assertIn("orcarouter_oauth", error)


if __name__ == "__main__":
    unittest.main()
