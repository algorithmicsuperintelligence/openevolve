"""Tests for the Codex app-server LLM backend.

The auth boundary is exercised against an in-process fake of `codex
app-server` that speaks the same newline-delimited JSON-RPC over real OS
pipes. No network access and no real `codex` binary are involved; the fake
never touches (and the real backend must never read) the CLI's token files.
"""

import asyncio
import json
import os
import subprocess
import threading
import unittest
from unittest.mock import MagicMock, patch

from openevolve.llm.codex_app_server import (
    CodexAppServerLLM,
    CodexAuthError,
    init_codex_app_server_client,
)

AUTHENTICATED_ACCOUNT = {
    "type": "chatgpt",
    "email": "user@example.com",
    "planType": "pro",
}


class FakeAppServer:
    """In-process fake of `codex app-server` speaking JSONL JSON-RPC on real pipes.

    It validates the protocol the same way the real server does: requests
    before `initialize` are answered with an error, and the requested methods
    are recorded so tests can assert on the exact conversation.
    """

    def __init__(self, scenario="ok"):
        self.scenario = scenario
        self.requests = []
        client_to_server_r, client_to_server_w = os.pipe()
        server_to_client_r, server_to_client_w = os.pipe()
        self.stdin = os.fdopen(client_to_server_w, "w")  # client writes here
        self.stdout = os.fdopen(server_to_client_r)  # client reads here
        self._server_reader = os.fdopen(client_to_server_r)
        self._server_writer = os.fdopen(server_to_client_w, "w")
        self.stderr = None
        self.pid = os.getpid()
        self.returncode = None
        self._exiting = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    # -- Popen-compatible surface used by the client -------------------------
    def terminate(self):
        self._exiting.set()
        try:
            self._server_reader.close()
        except OSError:
            pass

    def kill(self):
        self.terminate()

    def wait(self, timeout=None):
        return 0

    # -- Fake server side ----------------------------------------------------
    def _reply(self, request_id, result):
        self._send({"id": request_id, "result": result})

    def _error(self, request_id, error):
        self._send({"id": request_id, "error": error})

    def _notify(self, method, params):
        self._send({"method": method, "params": params})

    def _send(self, payload):
        try:
            self._server_writer.write(json.dumps(payload) + "\n")
            self._server_writer.flush()
        except (ValueError, OSError):
            pass

    def _serve(self):
        try:
            for line in self._server_reader:
                line = line.strip()
                if not line:
                    continue
                message = json.loads(line)
                method = message.get("method")
                if not method:
                    continue
                params = message.get("params") or {}
                self.requests.append((method, params))
                if not self._dispatch(method, message.get("id"), params):
                    break
        except (ValueError, OSError):
            pass
        finally:
            try:
                self._server_writer.close()
            except OSError:
                pass

    def _dispatch(self, method, request_id, params):
        """Handle one request; return False to hang up on the client."""
        if self.scenario == "hang" and method == "account/read":
            return True  # never reply: the client must hit its deadline
        if method == "initialize":
            self._reply(request_id, {"model": "gpt-5.3-codex", "authMode": "chatgpt"})
        elif method == "initialized":
            pass  # notification, no reply
        elif method == "account/read":
            if self.scenario == "unauthenticated":
                self._reply(request_id, {"account": None, "requiresOpenaiAuth": True})
            elif self.scenario == "apikey_account":
                self._reply(
                    request_id,
                    {"account": {"type": "apiKey"}, "requiresOpenaiAuth": False},
                )
            else:
                self._reply(
                    request_id,
                    {"account": AUTHENTICATED_ACCOUNT, "requiresOpenaiAuth": False},
                )
        elif method == "thread/start":
            self._reply(
                request_id,
                {"thread": {"id": "thread-123", "model": params.get("model")}},
            )
        elif self.scenario == "crash_after_thread" and method == "turn/start":
            return False  # hang up: the client must see EOF
        elif method == "turn/start":
            self._reply(
                request_id,
                {"threadId": "thread-123", "turn": {"id": "turn-1", "status": "inProgress"}},
            )
            self._notify(
                "turn/started",
                {"threadId": "thread-123", "turn": {"id": "turn-1", "status": "inProgress"}},
            )
            self._notify(
                "item/completed",
                {
                    "threadId": "thread-123",
                    "item": {"id": "item-0", "type": "agentMessage", "text": "hello world"},
                },
            )
            status = "interrupted" if self.scenario == "interrupted" else "completed"
            if self.scenario == "empty":
                self._notify(
                    "item/completed",
                    {
                        "threadId": "thread-123",
                        "item": {"id": "item-0", "type": "agentMessage", "text": ""},
                    },
                )
            self._notify(
                "turn/completed",
                {"threadId": "thread-123", "turn": {"id": "turn-1", "status": status}},
            )
        else:
            if request_id is not None:
                self._error(request_id, {"code": -1, "message": f"unknown method {method}"})
        return True


def _make_cfg(**overrides):
    defaults = {
        "name": "gpt-5.3-codex",
        "system_message": None,
        "timeout": 10,
        "weight": 1.0,
        "retries": 0,
        "retry_delay": 0,
        "reasoning_effort": None,
        "cwd": None,
        "codex_bin": None,
        "init_client": None,
        "provider": None,
    }
    defaults.update(overrides)
    cfg = MagicMock()
    for key, value in defaults.items():
        setattr(cfg, key, value)
    return cfg


def run_generate(scenario="ok", prompt="test prompt", cfg_overrides=None, kwargs=None):
    """Drive one generate() call against a FakeAppServer via patched Popen."""
    fake = FakeAppServer(scenario)
    captured = {}

    def _popen(cmd, **kwargs_):
        captured["cmd"] = cmd
        return fake

    llm = CodexAppServerLLM(_make_cfg(**(cfg_overrides or {})))
    outcome = {}

    def _execute():
        try:
            outcome["result"] = asyncio.run(llm.generate(prompt, **(kwargs or {})))
            outcome["error"] = None
        except Exception as e:  # noqa: BLE001 - tests assert on the type
            outcome["error"] = e

    with patch("openevolve.llm.codex_app_server.subprocess.Popen", side_effect=_popen):
        _execute()
    try:
        fake.terminate()
    except OSError:
        pass
    return outcome, fake, captured


class TestCodexAppServerLLMBasics(unittest.TestCase):
    def test_init_defaults(self):
        llm = CodexAppServerLLM(_make_cfg())
        self.assertEqual(llm.model, "gpt-5.3-codex")
        self.assertEqual(llm.timeout, 10)
        self.assertEqual(llm.weight, 1.0)
        self.assertEqual(llm.retries, 0)

    def test_init_falls_back_when_config_values_are_none(self):
        llm = CodexAppServerLLM(_make_cfg(name=None, timeout=None, retries=None, retry_delay=None))
        self.assertEqual(llm.model, "gpt-5.3-codex")
        self.assertEqual(llm.timeout, 300)
        self.assertEqual(llm.retries, 3)
        self.assertEqual(llm.retry_delay, 5)

    def test_factory_function(self):
        llm = init_codex_app_server_client(_make_cfg())
        self.assertIsInstance(llm, CodexAppServerLLM)
        self.assertEqual(llm.model, "gpt-5.3-codex")

    def test_invalid_reasoning_effort_raises(self):
        with self.assertRaisesRegex(ValueError, "Invalid reasoning_effort: turbo"):
            CodexAppServerLLM(_make_cfg(reasoning_effort="turbo"))


class TestAuthBoundary(unittest.TestCase):
    def test_authenticated_chatgpt_account_generates(self):
        outcome, fake, captured = run_generate(scenario="ok")
        self.assertIsNone(outcome["error"])
        self.assertEqual(outcome["result"], "hello world")
        methods = [m for m, _ in fake.requests]
        self.assertEqual(
            methods, ["initialize", "initialized", "account/read", "thread/start", "turn/start"]
        )
        self.assertEqual(captured["cmd"], ["codex", "app-server"])

    def test_unauthenticated_account_is_rejected_before_any_turn(self):
        outcome, fake, _ = run_generate(scenario="unauthenticated")
        self.assertIsInstance(outcome["error"], CodexAuthError)
        self.assertIn("codex login", str(outcome["error"]))
        methods = [m for m, _ in fake.requests]
        self.assertNotIn("thread/start", methods)
        self.assertNotIn("turn/start", methods)
        self.assertIn("account/read", methods)

    def test_apikey_account_is_rejected_with_apikey_guidance(self):
        outcome, fake, _ = run_generate(scenario="apikey_account")
        self.assertIsInstance(outcome["error"], CodexAuthError)
        self.assertIn("OpenAI provider", str(outcome["error"]))
        self.assertNotIn("turn/start", [m for m, _ in fake.requests])

    def test_auth_error_is_not_retried(self):
        outcome, fake, _ = run_generate(
            scenario="unauthenticated", cfg_overrides={"retries": 3, "retry_delay": 0}
        )
        self.assertIsInstance(outcome["error"], CodexAuthError)
        # Exactly one account/read round-trip: auth failures are terminal.
        self.assertEqual([m for m, _ in fake.requests].count("account/read"), 1)


class TestGenerationProtocol(unittest.TestCase):
    def test_initialize_handshake_carries_client_info(self):
        _, fake, _ = run_generate()
        params = next(p for m, p in fake.requests if m == "initialize")
        self.assertEqual(params["clientInfo"]["name"], "openevolve")

    def test_thread_is_non_interactive_and_read_only(self):
        _, fake, _ = run_generate()
        params = next(p for m, p in fake.requests if m == "thread/start")
        self.assertEqual(params["model"], "gpt-5.3-codex")
        self.assertEqual(params["approvalPolicy"], "never")
        self.assertEqual(params["sandbox"]["mode"], "read-only")
        self.assertFalse(params["sandbox"]["networkAccess"])

    def test_turn_input_and_effort(self):
        _, fake, _ = run_generate(
            prompt="write code",
            cfg_overrides={"reasoning_effort": "high", "system_message": "Be brief."},
        )
        params = next(p for m, p in fake.requests if m == "turn/start")
        self.assertEqual(params["threadId"], "thread-123")
        self.assertEqual(params["input"], [{"type": "text", "text": "Be brief.\n\nwrite code"}])
        self.assertEqual(params["effort"], "high")

    def test_no_effort_param_when_unset(self):
        _, fake, _ = run_generate()
        params = next(p for m, p in fake.requests if m == "turn/start")
        self.assertNotIn("effort", params)

    def test_custom_codex_bin_is_used(self):
        _, _, captured = run_generate(cfg_overrides={"codex_bin": "/opt/codex/bin/codex"})
        self.assertEqual(captured["cmd"], ["/opt/codex/bin/codex", "app-server"])

    def test_interrupted_turn_raises(self):
        outcome, _, _ = run_generate(scenario="interrupted")
        self.assertIsInstance(outcome["error"], RuntimeError)
        self.assertIn("did not complete", str(outcome["error"]))

    def test_empty_agent_message_raises(self):
        outcome, _, _ = run_generate(scenario="empty")
        self.assertIsInstance(outcome["error"], RuntimeError)
        self.assertIn("Empty response", str(outcome["error"]))

    def test_server_crash_mid_turn_raises(self):
        outcome, _, _ = run_generate(scenario="crash_after_thread")
        self.assertIsInstance(outcome["error"], RuntimeError)
        self.assertIn("exited unexpectedly", str(outcome["error"]))

    def test_hung_server_hits_deadline(self):
        outcome, _, _ = run_generate(scenario="hang", cfg_overrides={"timeout": 1})
        self.assertIsInstance(outcome["error"], asyncio.TimeoutError)

    def test_missing_cli_raises_without_retrying(self):
        llm = CodexAppServerLLM(_make_cfg(retries=3, retry_delay=0))
        with patch(
            "openevolve.llm.codex_app_server.subprocess.Popen",
            side_effect=FileNotFoundError("codex"),
        ) as mock_popen:
            with self.assertRaises(FileNotFoundError):
                asyncio.run(llm.generate("prompt"))
        self.assertEqual(mock_popen.call_count, 1)

    def test_retry_on_transient_failure(self):
        llm = CodexAppServerLLM(_make_cfg(retries=1, retry_delay=0))
        good = FakeAppServer("ok")
        self.addCleanup(good.terminate)
        with patch(
            "openevolve.llm.codex_app_server.subprocess.Popen",
            side_effect=[subprocess.TimeoutExpired(cmd="codex", timeout=1), good],
        ):
            result = asyncio.run(llm.generate("prompt"))
        self.assertEqual(result, "hello world")


class TestCodexAppServerConfig(unittest.TestCase):
    def test_config_round_trip_without_api_key(self):
        from openevolve.config import Config

        config = Config.from_dict(
            {
                "llm": {
                    "provider": "codex_app_server",
                    "models": [
                        {
                            "name": "gpt-5.3-codex",
                            "weight": 1.0,
                            "reasoning_effort": "medium",
                        }
                    ],
                }
            }
        )
        model = config.llm.models[0]
        self.assertEqual(model.provider, "codex_app_server")
        self.assertIsNone(model.api_key)
        self.assertEqual(model.reasoning_effort, "medium")

    def test_provider_propagates_from_llm_config(self):
        from openevolve.config import Config

        config = Config.from_dict(
            {
                "llm": {
                    "provider": "codex_app_server",
                    "models": [{"name": "gpt-5.3-codex", "weight": 1.0}],
                }
            }
        )
        self.assertEqual(config.llm.models[0].provider, "codex_app_server")
        self.assertEqual(config.llm.evaluator_models[0].provider, "codex_app_server")


class TestProviderRegistry(unittest.TestCase):
    def test_codex_app_server_in_registry(self):
        from openevolve.llm.ensemble import _PROVIDER_REGISTRY

        self.assertIn("codex_app_server", _PROVIDER_REGISTRY)

    def test_ensemble_creates_codex_app_server(self):
        from openevolve.llm.ensemble import _create_model

        cfg = _make_cfg()
        cfg.provider = "codex_app_server"
        self.assertIsInstance(_create_model(cfg), CodexAppServerLLM)

    def test_apikey_default_path_still_opensai(self):
        """No provider configured -> the OpenAI (API key) path is untouched."""
        from openevolve.config import Config
        from openevolve.llm.ensemble import LLMEnsemble

        config = Config.from_dict({"llm": {"models": [{"name": "gpt-4"}]}})
        ensemble = LLMEnsemble(config.llm.models)
        self.assertEqual(type(ensemble.models[0]).__name__, "OpenAILLM")


if __name__ == "__main__":
    unittest.main()
