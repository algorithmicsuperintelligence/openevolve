"""Program source is UTF-8 on disk whatever the locale, as the evaluator writes it."""

import os
import subprocess
import sys
import unittest

CHECK_CONTROLLER = r"""
import locale
import os
import sys
import tempfile

if locale.getpreferredencoding(False).lower().replace('-', '') == 'utf8':
    sys.exit(77)

from openevolve.config import Config
from openevolve.controller import OpenEvolve
from openevolve.database import Program

text = chr(0x4E2D) + chr(0x6587) + chr(0x1F9EA)
initial_code = '# ' + text + chr(10) + 'x = 1' + chr(10)
evolved_code = '# evolved ' + text + chr(10) + 'x = 2' + chr(10)

with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:
    paths = {name: os.path.join(root, name) for name in ('initial.py', 'seed.py', 'evaluator.py')}
    for name in ('initial.py', 'seed.py'):
        with open(paths[name], 'wb') as f:
            f.write(initial_code.encode('utf-8'))
    with open(paths['evaluator.py'], 'wb') as f:
        f.write(b'def evaluate(program_path):' + bytes([10]) + b'    return {}' + bytes([10]))

    config = Config()
    config.database.island_initial_programs = [None, paths['seed.py']]
    controller = OpenEvolve(
        paths['initial.py'], paths['evaluator.py'], config, os.path.join(root, 'output')
    )
    assert controller.initial_program_code == initial_code, ascii(controller.initial_program_code)
    seed_code = controller.island_initial_program_codes[1]
    assert seed_code == initial_code, ascii(seed_code)

    best = Program(id='best', code=evolved_code, metrics={'combined_score': 1.0})
    controller.database.add(best)
    controller._save_best_program(best)
    controller._save_checkpoint(1)
    for folder in ('best', os.path.join('checkpoints', 'checkpoint_1')):
        with open(os.path.join(controller.output_dir, folder, 'best_program.py'), 'rb') as f:
            saved = f.read().decode('utf-8')
        assert saved.replace(chr(13), '') == evolved_code, (folder, ascii(saved))
"""


class TestControllerUtf8Programs(unittest.TestCase):
    def test_programs_round_trip_under_non_utf8_locale(self):
        env = dict(
            os.environ,
            LC_ALL="C",
            PYTHONUTF8="0",
            PYTHONCOERCECLOCALE="0",
            OPENAI_API_KEY="test",
        )
        result = subprocess.run(
            [sys.executable, "-c", CHECK_CONTROLLER],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            env=env,
        )
        if result.returncode == 77:
            self.skipTest("This platform retains a UTF-8 default locale")
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
