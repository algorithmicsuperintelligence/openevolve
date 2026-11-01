"""Real-process regressions for bounded app-server shutdown."""

import asyncio
import os
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch

from openevolve.llm.codex_app_server import CodexAppServerLLM, _AppServerSession


class TestCodexShutdown(unittest.TestCase):
    def _check_shutdown(self, request_timeout=False, ignore_termination=False):
        script = "import time; "
        if ignore_termination:
            script += "import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        script += 'print(\'{"method":"ready"}\', flush=True); time.sleep(60)'
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        session = _AppServerSession(process)
        failures = []
        worker = None

        def close_or_timeout():
            try:
                if request_timeout:
                    model = CodexAppServerLLM()
                    with patch.object(model, "_start_session", return_value=session):
                        model._run_app_server("test", timeout=0.05)
                else:
                    session.close()
            except BaseException as error:
                failures.append(error)

        try:
            self.assertEqual(session.next_message(time.monotonic() + 5)["method"], "ready")
            worker = threading.Thread(target=close_or_timeout, daemon=True)
            worker.start()
            worker.join(timeout=8 if ignore_termination else 2)
            returned_without_rescue = not worker.is_alive()
            exited_without_rescue = process.poll() is not None
        finally:
            # Even the broken implementation must not leave a sleeping child behind.
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            if worker is not None:
                worker.join(timeout=5)
            session._reader.join(timeout=5)

        self.assertTrue(returned_without_rescue, "shutdown blocked before terminating the child")
        self.assertTrue(exited_without_rescue, "shutdown returned without reaping the child")
        self.assertFalse(worker.is_alive())
        self.assertFalse(session._reader.is_alive())
        if request_timeout:
            self.assertEqual(len(failures), 1)
            self.assertIsInstance(failures[0], asyncio.TimeoutError)
        else:
            self.assertEqual(failures, [])

    def test_close_terminates_child_before_closing_reader_pipe(self):
        self._check_shutdown()

    def test_request_timeout_can_finish_cleanup(self):
        self._check_shutdown(request_timeout=True)

    @unittest.skipUnless(os.name == "posix", "POSIX signal handling required")
    def test_shutdown_kills_and_reaps_child_ignoring_sigterm(self):
        self._check_shutdown(ignore_termination=True)


if __name__ == "__main__":
    unittest.main()
