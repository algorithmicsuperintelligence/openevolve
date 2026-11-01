"""Exercise CLI text pipes with real processes under a non-UTF-8 locale."""

import os
import subprocess
import sys
import unittest

CHECK_CLI = r"""
import json
import locale
import sys
import time
from unittest.mock import patch

if locale.getpreferredencoding(False).lower().replace('-', '') == 'utf8':
    sys.exit(77)

text = chr(0x4E2D) + chr(0x6587) + chr(0x1F9EA)
write_text = 'import sys; sys.stdout.buffer.write(' + repr(text.encode('utf-8')) + ')'
command = [sys.executable, '-c', write_text]
case = sys.argv[1]

if case in ('claude_output', 'claude_input'):
    from openevolve.llm.claude_code import ClaudeCodeLLM
    model = ClaudeCodeLLM()
    if case == 'claude_input':
        command = [sys.executable, '-c', 'import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())']
        result = model._run_cli(command, timeout=5, prompt=text)
    else:
        result = model._run_cli(command, timeout=5)
elif case == 'copilot_output':
    from openevolve.llm.copilot_cli import CopilotCLILLM
    result = CopilotCLILLM()._run_cli(command, timeout=5)
else:
    from openevolve.llm.codex_app_server import CodexAppServerLLM
    payload = {'method': 'example', 'params': {'text': text}}
    wire = (json.dumps(payload, ensure_ascii=False) + chr(10)).encode('utf-8')
    command = [sys.executable, '-c', 'import sys; sys.stdout.buffer.write(' + repr(wire) + ')']
    model = CodexAppServerLLM()
    with patch.object(model, '_build_command', return_value=command):
        session = model._start_session()
    try:
        message = session.next_message(time.monotonic() + 5)
        assert message['method'] == 'example'
        result = message['params']['text']
    finally:
        session.close()

assert result == text, (ascii(result), ascii(text))
"""


class TestCliUtf8(unittest.TestCase):
    def _check_cli(self, case):
        env = dict(os.environ, LC_ALL="C", PYTHONUTF8="0", PYTHONCOERCECLOCALE="0")
        result = subprocess.run(
            [sys.executable, "-c", CHECK_CLI, case],
            capture_output=True,
            encoding="utf-8",
            timeout=15,
            env=env,
        )
        if result.returncode == 77:
            self.skipTest("This platform retains a UTF-8 default locale")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_claude_reads_utf8_output(self):
        self._check_cli("claude_output")

    def test_claude_sends_utf8_stdin(self):
        self._check_cli("claude_input")

    def test_copilot_reads_utf8_output(self):
        self._check_cli("copilot_output")

    def test_codex_reads_utf8_jsonl(self):
        self._check_cli("codex_output")


if __name__ == "__main__":
    unittest.main()
