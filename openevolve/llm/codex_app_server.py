"""
Codex app-server interface for LLMs.

Uses the Codex CLI's app-server (`codex app-server`) as a non-interactive LLM
backend, enabling OpenEvolve to run with models available through a ChatGPT
(Codex) subscription without requiring direct API keys.

Authentication is delegated entirely to the app-server's managed ChatGPT login
(browser or device-code flow): the app-server persists the tokens itself and
refreshes them automatically. This adapter never reads the CLI's token files.
Before running a turn it asks the app-server for the current account state
(`account/read`) and requires a ChatGPT-plan login — if Codex is not
authenticated, it fails with actionable instructions instead of falling back
to an API key.

Usage in config.yaml:
    llm:
      provider: "codex_app_server"
      models:
        - name: "gpt-5.3-codex"
          weight: 1.0
          timeout: 300
          reasoning_effort: "medium"

Or inject programmatically:
    from openevolve.llm.codex_app_server import init_codex_app_server_client
    for model_cfg in config.llm.models:
        model_cfg.init_client = init_codex_app_server_client

Notes:
- `temperature`, `top_p` and `max_tokens` have no equivalent app-server option
  in this first pass and are therefore ignored.
- Each generation call spawns a fresh `codex app-server` process, speaks
  newline-delimited JSON-RPC over its stdio (the protocol's default transport),
  and shuts it down again. Threads are non-interactive: tool approvals are set
  to "never" and the workspace sandbox to read-only, so the model can answer
  but not mutate the tree while generating a response.
- The stdio transport is unsupported on Windows; the backend expects a POSIX
  shell there.
"""

import asyncio
import json
import logging
import queue
import subprocess
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)

# Union of reasoning efforts documented for the GPT-5.x / Codex model families.
VALID_REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")

_CLIENT_INFO = {"name": "openevolve", "title": "OpenEvolve", "version": "1.0.0"}


class CodexAuthError(RuntimeError):
    """Raised when the Codex app-server reports no usable ChatGPT login.

    The Codex app-server owns the ChatGPT browser/device-code login and
    refreshes tokens itself; OpenEvolve never reads or writes the CLI's token
    files, so the only remedy is to log in via the Codex CLI first.
    """

    def __init__(self, message: str):
        super().__init__(
            f"{message} Run `codex login` (browser flow) or `codex login --device-code` "
            "to authenticate with your ChatGPT account first."
        )


def _cfg_value(model_cfg: Any, attr: str, default: Any) -> Any:
    value = getattr(model_cfg, attr, None)
    return default if value is None else value


class _AppServerSession:
    """Client for one `codex app-server` process speaking JSONL JSON-RPC.

    Requests are written as one JSON object per line to the process's stdin;
    responses (which echo the request id) and notifications (which omit it) are
    read line by line from stdout. A background thread feeds a queue so reads
    can honor a wall-clock deadline instead of blocking forever on a hung
    app-server.
    """

    def __init__(self, proc: "subprocess.Popen", stderr_tail: int = 500):
        self.proc = proc
        self._next_id = 0
        self._messages: "queue.Queue[Any]" = queue.Queue()
        self._pending: "deque[Any]" = deque()
        self._stderr_tail = stderr_tail
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

    def _read_stdout(self) -> None:
        try:
            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._messages.put(json.loads(line))
                except json.JSONDecodeError:
                    logger.debug(f"Non-JSON line from codex app-server: {line[:200]}")
        except (ValueError, OSError):
            pass
        finally:
            self._messages.put(None)  # EOF sentinel

    def stderr_summary(self) -> str:
        try:
            raw = (self.proc.stderr.read() or "") if self.proc.stderr else ""
        except (ValueError, OSError):
            raw = ""
        return raw.strip()[-self._stderr_tail :]

    def request(self, method: str, params: Optional[Dict[str, Any]], deadline: float) -> Any:
        """Send a request and block until its response arrives or time runs out."""
        self._next_id += 1
        request_id = self._next_id
        self._send({"id": request_id, "method": method, "params": params or {}})
        while True:
            message = self.next_message(deadline)
            if message.get("id") != request_id:
                # A notification (e.g. an early item/completed) or a late
                # response: park it instead of dropping it.
                self._pending.append(message)
                continue
            if "error" in message:
                error = message["error"] or {}
                raise RuntimeError(
                    f"codex app-server rejected {method}: "
                    f"{error.get('code')}: {error.get('message')}"
                )
            return message.get("result")

    def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        self._send({"method": method, "params": params or {}})

    def next_message(self, deadline: float) -> Dict[str, Any]:
        """Return the next raw JSON-RPC message, honoring the wall-clock deadline."""
        if self._pending:
            return self._pending.popleft()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise asyncio.TimeoutError("codex app-server did not respond in time")
        try:
            message = self._messages.get(timeout=remaining)
        except queue.Empty:
            raise asyncio.TimeoutError("codex app-server did not respond in time")
        if message is None:  # EOF: the process exited
            raise RuntimeError(
                f"codex app-server exited unexpectedly. stderr: {self.stderr_summary()}"
            )
        return message

    def _send(self, payload: Dict[str, Any]) -> None:
        try:
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()
        except (ValueError, OSError, BrokenPipeError) as e:
            raise RuntimeError(f"Failed to write to codex app-server: {e}. {self.stderr_summary()}")

    def close(self) -> None:
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            try:
                if stream:
                    stream.close()
            except (OSError, ValueError):
                pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


class CodexAppServerLLM(LLMInterface):
    """LLM interface that generates through the Codex app-server.

    Requires the `codex` CLI to be installed and logged in with a ChatGPT
    account (the app-server manages that login and token refresh itself).
    """

    def __init__(self, model_cfg=None):
        self.model = _cfg_value(model_cfg, "name", "gpt-5.3-codex")
        self.system_message = getattr(model_cfg, "system_message", None)
        self.timeout = _cfg_value(model_cfg, "timeout", 300)
        self.weight = _cfg_value(model_cfg, "weight", 1.0)
        self.retries = _cfg_value(model_cfg, "retries", 3)
        self.retry_delay = _cfg_value(model_cfg, "retry_delay", 5)
        self.reasoning_effort = getattr(model_cfg, "reasoning_effort", None)
        self.cwd = getattr(model_cfg, "cwd", None)
        self.codex_bin = _cfg_value(model_cfg, "codex_bin", "codex")
        if (
            self.reasoning_effort is not None
            and self.reasoning_effort not in VALID_REASONING_EFFORTS
        ):
            raise ValueError(
                f"Invalid reasoning_effort: {self.reasoning_effort}. "
                f"Expected one of {', '.join(VALID_REASONING_EFFORTS)}"
            )
        logger.info(f"Initialized CodexAppServerLLM with model: {self.model}")

    async def generate(self, prompt: str, **kwargs) -> str:
        sys_msg = kwargs.pop("system_message", self.system_message) or ""
        return await self.generate_with_context(
            system_message=sys_msg,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        user_content = "\n\n".join(
            m.get("content", "") for m in messages if m.get("role") == "user"
        )
        prompt = f"{system_message}\n\n{user_content}" if system_message else user_content

        timeout = kwargs.get("timeout", self.timeout)
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)

        loop = asyncio.get_event_loop()
        for attempt in range(retries + 1):
            try:
                return await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: self._run_app_server(prompt, timeout)),
                    timeout=timeout + 30,
                )
            except FileNotFoundError:
                logger.error(
                    "The `codex` CLI was not found. Install it and run `codex login` first."
                )
                raise
            except CodexAuthError as e:
                # Retrying cannot help until the user logs in; fail immediately.
                logger.error(f"Codex app-server authentication error (not retried): {e}")
                raise
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(
                        f"Codex app-server timeout on attempt {attempt + 1}/{retries + 1}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with timeout")
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Codex app-server error on attempt {attempt + 1}/{retries + 1}: {e}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with error: {e}")
                    raise

    def _build_command(self) -> List[str]:
        return [self.codex_bin, "app-server"]

    def _run_app_server(self, prompt: str, timeout: int) -> str:
        deadline = time.monotonic() + timeout
        session = self._start_session()
        try:
            # JSON-RPC handshake (any request before this is rejected by the server).
            session.request("initialize", {"clientInfo": _CLIENT_INFO}, deadline)
            session.notify("initialized")

            # Auth boundary: the app-server owns the ChatGPT login and token
            # refresh; verify it has a usable subscription account up front.
            self._ensure_authenticated(session, deadline)

            thread_params: Dict[str, Any] = {
                "model": self.model,
                "approvalPolicy": "never",
                "sandbox": {"mode": "read-only", "networkAccess": False},
            }
            if self.cwd:
                thread_params["cwd"] = self.cwd
            thread_result = session.request("thread/start", thread_params, deadline)
            thread_id = (thread_result or {}).get("thread", {}).get("id")
            if not thread_id:
                raise RuntimeError(f"codex app-server returned no thread id: {thread_result}")

            turn_params: Dict[str, Any] = {
                "threadId": thread_id,
                "input": [{"type": "text", "text": prompt}],
            }
            if self.reasoning_effort is not None:
                turn_params["effort"] = self.reasoning_effort
            session.request("turn/start", turn_params, deadline)

            return self._collect_response(session, deadline)
        finally:
            session.close()

    def _ensure_authenticated(self, session: _AppServerSession, deadline: float) -> None:
        result = session.request("account/read", {}, deadline) or {}
        if result.get("requiresOpenaiAuth") or result.get("account") is None:
            raise CodexAuthError("Codex app-server reports the account is not authenticated.")
        account = result.get("account") or {}
        if account.get("type") != "chatgpt":
            raise CodexAuthError(
                "Codex app-server is not signed in with a ChatGPT account "
                f"(found auth type '{account.get('type')}'). The codex_app_server "
                "provider bills a ChatGPT subscription; to use an API key instead, "
                "keep the default OpenAI provider."
            )

    def _collect_response(self, session: _AppServerSession, deadline: float) -> str:
        """Stream notifications until the turn finishes; return the agent message.

        The agent text arrives as `item/completed` notifications
        (`item.type == "agentMessage"`); `turn/completed` closes the turn.
        """
        text: Optional[str] = None
        while True:
            message = session.next_message(deadline)
            if message.get("id") is not None:
                continue  # a response, not a notification
            method = message.get("method")
            params = message.get("params") or {}
            if method == "item/completed":
                item = params.get("item") or {}
                if item.get("type") == "agentMessage":
                    text = item.get("text") or ""
            elif method == "turn/completed":
                status = (params.get("turn") or {}).get("status")
                if status and status != "completed":
                    raise RuntimeError(f"codex app-server turn did not complete: {status}")
                break
            elif method == "error":
                raise RuntimeError(f"codex app-server reported an error: {params}")
        if not text or not text.strip():
            raise RuntimeError(
                f"Empty response from codex app-server. stderr: {session.stderr_summary()}"
            )
        return text

    def _start_session(self) -> _AppServerSession:
        proc = subprocess.Popen(
            self._build_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.cwd,
        )
        return _AppServerSession(proc)


def init_codex_app_server_client(model_cfg):
    """Factory function compatible with OpenEvolve's init_client config hook."""
    return CodexAppServerLLM(model_cfg)
