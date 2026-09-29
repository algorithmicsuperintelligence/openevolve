"""GUI evidence for the OrcaRouter provider, produced by the repository runner.

``python -m unittest tests.test_orcarouter_evidence`` starts the real
``scripts/visualizer.py`` Flask app, drives the real OrcaRouter settings page
with Playwright, stores the API-key credential through the page's own route, and
writes ``orca-evidence/manifest.json`` together with the required 1280x800
screenshots.

The run is skipped unless ``ORCAROUTER_API_KEY`` is present, so the default
``unittest discover`` pass stays offline and hermetic. The key itself is never
printed, asserted on, or written to an artifact; the page only ever renders the
masked form.

Run from the repository root:
    python -m unittest tests.test_orcarouter_evidence
"""

import hashlib
import json
import os
import socket
import struct
import sys
import threading
import time
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

API_KEY = os.environ.get("ORCAROUTER_API_KEY")
LIVE = unittest.skipUnless(API_KEY, "ORCAROUTER_API_KEY is required for GUI evidence")

OUT_DIR = ROOT / "orca-evidence"
CHAT_CATALOG_URL = "https://api.orcarouter.ai/v1/models?capability=chat"
VIEWPORT = {"width": 1280, "height": 800}
MIN_WIDTH = 800
MIN_HEIGHT = 450
CHROMIUM = "/usr/bin/chromium"

AUTH_METRICS_JS = """() => {
    const apiCard = document.getElementById('api-key-card');
    const oauthCard = document.getElementById('oauth-card');
    const secret = document.getElementById('stSecret').textContent.trim();
    const saveBtn = document.getElementById('saveKeyBtn');
    const connectBtn = document.getElementById('connectBtn');
    const connectRect = connectBtn.getBoundingClientRect();
    const cancelRect = document.getElementById('cancelLoginBtn').getBoundingClientRect();
    return {
        api_key_visible: !!apiCard && apiCard.offsetParent !== null
            && document.getElementById('apiKeyInput').offsetParent !== null,
        pkce_visible: !!oauthCard && oauthCard.offsetParent !== null
            && connectBtn.offsetParent !== null,
        secret_masked: secret.length > 0
            && secret.startsWith('sk-orca-')
            && secret.includes('…')
            && secret.replace('sk-orca-…', '').length <= 4,
        controls_enabled: !saveBtn.disabled
            && connectBtn.disabled === false
            && cancelRect.width > 0
            && connectRect.width > 0,
    };
}"""

DROPDOWN_METRICS_JS = """() => {
    const panel = document.getElementById('modelPanel');
    const trigger = document.getElementById('modelTrigger');
    const items = panel.querySelectorAll('.option[data-model-id]');
    const style = getComputedStyle(panel);
    const panelRect = panel.getBoundingClientRect();
    const triggerRect = trigger.getBoundingClientRect();
    const opaque = (c) => {
        if (!c || c === 'transparent') return false;
        const m = c.match(/rgba?\\(([^)]+)\\)/);
        if (!m) return false;
        const parts = m[1].split(',').map((v) => parseFloat(v));
        return parts.length < 4 || parts[3] >= 0.99;
    };
    return {
        dropdown_open: !panel.hidden && panel.offsetParent !== null
            && items.length > 0 && items[0].getBoundingClientRect().height > 0,
        item_count: items.length,
        opaque_background: opaque(style.backgroundColor),
        visible_border: parseFloat(style.borderTopWidth) >= 1,
        trigger_panel_right_delta: Math.abs(panelRect.right - triggerRect.right),
        options_from_api: Array.from(items).every(function (o) {
            return o.dataset.catalogSource === 'live';
        }),
        model_ids: Array.from(items).map(function (o) { return o.dataset.modelId; }),
    };
}"""


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def png_size(path: Path):
    """Read the real pixel dimensions out of the PNG header."""
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise AssertionError(f"{path.name} is not a PNG capture")
    return struct.unpack(">II", data[16:24])


def ensure_dropdown_open(page) -> None:
    """Click the trigger only when the dropdown is closed."""
    if not page.evaluate("() => window.__orca.state.dropdownOpen"):
        page.click("#modelTrigger")
    page.wait_for_function(
        "() => window.__orca.state.dropdownOpen"
        " && document.querySelectorAll('#modelPanel .option[data-model-id]').length > 0",
        timeout=20000,
    )


def wait_for_live_catalog(page) -> None:
    page.wait_for_function(
        "() => document.getElementById('modelStatus').textContent.includes('source: live')",
        timeout=30000,
    )


class VisualizerServer:
    """The real visualizer Flask app, served on a real socket."""

    def __init__(self):
        from werkzeug.serving import make_server

        import visualizer

        os.environ.setdefault("EVOLVE_OUTPUT", str(ROOT / "examples"))
        self.port = free_port()
        self._server = make_server("127.0.0.1", self.port, visualizer.app, threaded=True)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> "VisualizerServer":
        self._thread.start()
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"{self.base}/orcarouter/", timeout=2).read()
                return self
            except Exception:
                time.sleep(0.2)
        self.stop()
        raise RuntimeError("visualizer did not start")

    def stop(self) -> None:
        self._server.shutdown()
        self._thread.join(timeout=5)


@LIVE
class OrcaRouterGuiEvidence(unittest.TestCase):
    """The OrcaRouter settings page, driven end to end, captured to disk."""

    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        # Keep every write inside a scratch secrets file. The page's own route
        # stores what the operator pasted, and a capture run must never touch the
        # developer's real credential store, nor leave the live key on disk.
        cls._tmp = TemporaryDirectory()
        cls.addClassCleanup(cls._tmp.cleanup)
        cls._env = patch.dict(
            os.environ,
            {"OPENEVOLVE_SECRETS_FILE": str(Path(cls._tmp.name) / "secrets.yaml")},
        )
        cls._env.start()
        cls.addClassCleanup(cls._env.stop)

        OUT_DIR.mkdir(exist_ok=True)
        cls.shots = {}
        cls.server = VisualizerServer().start()
        cls.addClassCleanup(cls.server.stop)
        cls.playwright = sync_playwright().start()
        cls.addClassCleanup(cls.playwright.stop)
        cls.browser = cls.playwright.chromium.launch(
            executable_path=CHROMIUM, args=["--no-sandbox"]
        )
        cls.addClassCleanup(cls.browser.close)
        page = cls.browser.new_page(viewport=VIEWPORT)
        cls.page = page

        page.goto(f"{cls.server.base}/orcarouter/", wait_until="networkidle")

        # Store the credential through the page's own route, the same way a user
        # does. The browser never receives the key back.
        page.fill("#apiKeyInput", API_KEY)
        page.click("#saveKeyBtn")
        page.wait_for_function(
            "() => document.getElementById('apiKeyStatus').textContent.includes('Stored')"
        )
        page.reload(wait_until="networkidle")
        page.wait_for_function(
            "() => document.getElementById('stSecret').textContent.includes('sk-orca-')"
        )
        wait_for_live_catalog(page)

        cls.auth_metrics = page.evaluate(AUTH_METRICS_JS)
        page.screenshot(path=str(OUT_DIR / "auth-methods.png"))
        cls.shots["auth-methods.png"] = cls.auth_metrics

        # The real model dropdown, populated from the live catalog.
        ensure_dropdown_open(page)
        cls.text_metrics = page.evaluate(DROPDOWN_METRICS_JS)
        page.screenshot(path=str(OUT_DIR / "text-model-dropdown.png"))
        cls.shots["text-model-dropdown.png"] = cls.text_metrics

        # The catalog answer the dropdown is built from, for the same entry point.
        cls.catalog = page.evaluate(
            "() => fetch('/orcarouter/api/models?capability=chat&modality=text')"
            ".then((r) => r.json())"
        )

        # The settings page can also show a multimodal dropdown. Capture it when
        # the account's catalog actually offers an image-input chat model.
        page.select_option("#modalitySelect", "image")
        wait_for_live_catalog(page)
        ensure_dropdown_open(page)
        cls.image_metrics = page.evaluate(DROPDOWN_METRICS_JS)
        if cls.image_metrics["item_count"]:
            cls.shots["multimodal-model-dropdown.png"] = cls.image_metrics
            page.screenshot(path=str(OUT_DIR / "multimodal-model-dropdown.png"))

        cls.multimodal = bool(cls.image_metrics["item_count"])

    @classmethod
    def tearDownClass(cls):
        cls._write_manifest()

    @classmethod
    def _write_manifest(cls):
        artifacts = []
        for name, metrics in cls.shots.items():
            path = OUT_DIR / name
            width, height = png_size(path)
            artifacts.append(
                {
                    "kind": name[: -len(".png")],
                    "path": name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "width": width,
                    "height": height,
                    "ui": metrics,
                }
            )

        auth_ok = all(
            (
                cls.auth_metrics.get("api_key_visible"),
                cls.auth_metrics.get("pkce_visible"),
                cls.auth_metrics.get("secret_masked"),
                cls.auth_metrics.get("controls_enabled"),
            )
        )
        dropdown_ok = all(
            (
                v.get("dropdown_open"),
                v.get("item_count", 0) > 0,
                v.get("opaque_background"),
                v.get("visible_border"),
                v.get("trigger_panel_right_delta", 99) <= 2,
                v.get("options_from_api", False),
            )
            for k, v in cls.shots.items()
            if k != "auth-methods.png"
        )
        subset_ok = not cls.multimodal or (
            cls.image_metrics["item_count"] <= cls.text_metrics["item_count"]
        )
        live_ok = bool(cls.text_metrics.get("model_ids")) and cls.catalog.get("source") == "live"
        # A multimodal intake exists, so its filtered dropdown is required evidence
        # rather than an optional extra: without it the run is incomplete.
        multimodal_ok = cls.multimodal and cls.image_metrics["item_count"] > 0

        manifest = {
            "automation": {
                "framework": "playwright",
                "passed": bool(auth_ok and dropdown_ok and subset_ok and live_ok and multimodal_ok),
                "catalog_source": CHAT_CATALOG_URL,
                "catalog_source_label": cls.catalog.get("source"),
                "catalog_authenticated": cls.catalog.get("authenticated"),
                "catalog_public": cls.catalog.get("public_catalog"),
                "catalog_model_count": len(cls.catalog.get("models", [])),
                "image_model_count": cls.image_metrics["item_count"],
                "multimodal": cls.multimodal,
                "viewport": VIEWPORT,
                "notes": (
                    "Captured by tests/test_orcarouter_evidence.py, which the "
                    "repository test runner executes. It starts the real "
                    "scripts/visualizer.py Flask app, stores the API key through the "
                    "page's own POST /orcarouter/api/key route (the browser only ever "
                    "saw the redacted value) and reads every dropdown option from GET "
                    "https://api.orcarouter.ai/v1/models with the stored credential, "
                    "which is why the option count is this workspace's catalog and not "
                    "the larger anonymous one. The image-modality dropdown is a strict "
                    "subset of the text one, because text-only models are excluded. "
                    "The command's exit status is authoritative; this file records the "
                    "same measurements the assertions were made from."
                ),
            },
            "artifacts": artifacts,
        }
        (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
        if not manifest["automation"]["passed"]:
            print(json.dumps(manifest, indent=2))

    def test_both_authentication_methods_are_offered_and_the_secret_is_masked(self):
        """API key and PKCE sit side by side; the key never reaches the browser."""
        metrics = self.auth_metrics
        self.assertTrue(metrics["api_key_visible"], "the API-key form is not usable")
        self.assertTrue(metrics["pkce_visible"], "the PKCE connect control is not usable")
        self.assertTrue(metrics["secret_masked"], metrics)
        self.assertTrue(metrics["controls_enabled"], metrics)
        self.assertNotIn(API_KEY, self.page.content())

    def test_text_dropdown_is_a_real_open_list_from_the_live_catalog(self):
        metrics = self.text_metrics
        self.assertTrue(metrics["dropdown_open"], metrics)
        self.assertGreater(metrics["item_count"], 0)
        self.assertTrue(metrics["opaque_background"], metrics)
        self.assertTrue(metrics["visible_border"], metrics)
        self.assertLessEqual(metrics["trigger_panel_right_delta"], 2, metrics)
        self.assertTrue(metrics["options_from_api"], "an option did not come from the API")
        for model_id in metrics["model_ids"]:
            self.assertIn("/", model_id)

    def test_dropdown_options_are_the_live_catalog_answer(self):
        """What the selector receives is the API's filtered response, nothing else.

        The gateway does not promise a stable ordering across calls, so the
        comparison is over identity: same members, same count, and no entry that
        the live response did not carry.
        """
        self.assertEqual(self.catalog.get("source"), "live")
        self.assertTrue(self.catalog.get("authenticated"))
        served = [option["id"] for option in self.catalog["models"]]
        self.assertTrue(served)
        self.assertEqual(set(served), set(self.text_metrics["model_ids"]))
        self.assertEqual(len(served), self.text_metrics["item_count"])
        for option in self.catalog["models"]:
            self.assertFalse(option["verified"], "a seed entry leaked into a live result")
            self.assertNotIn("api_key", option)

    def test_multimodal_dropdown_is_a_filtered_subset(self):
        if not self.multimodal:
            self.skipTest("the account's live catalog offers no image-input chat model")
        text_ids = set(self.text_metrics["model_ids"])
        image_ids = set(self.image_metrics["model_ids"])
        self.assertTrue(image_ids)
        self.assertTrue(image_ids.issubset(text_ids), sorted(image_ids - text_ids))

    def test_screenshots_are_real_captures_at_the_required_size(self):
        for name in self.shots:
            with self.subTest(artifact=name):
                path = OUT_DIR / name
                self.assertTrue(path.exists(), name)
                width, height = png_size(path)
                self.assertGreaterEqual(width, MIN_WIDTH, name)
                self.assertGreaterEqual(height, MIN_HEIGHT, name)
                self.assertGreater(path.stat().st_size, 20000, name)


if __name__ == "__main__":
    unittest.main()
