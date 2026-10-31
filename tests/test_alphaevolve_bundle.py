"""Offline bundle creation, relocation, execution and failure-path tests."""

import contextlib
import copy
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from openevolve.compat.alphaevolve_bundle import export_to_alphaevolve, main
from openevolve.config import Config

EVALUATOR = """from dataclasses import dataclass
from pathlib import Path

@dataclass
class Score:
    value: int

def evaluate(path):
    print("evaluator diagnostic")
    return {"score": Score(len(Path(path).read_bytes())).value}
"""


class TestAlphaEvolveBundle(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.program = self.inputs / "seed.py"
        self.program.write_bytes("# 中文\r\n7\r\n".encode("utf-8"))
        self.evaluator = self.inputs / "evaluator.py"
        self.evaluator.write_text(EVALUATOR, encoding="utf-8")
        self.config = Config.from_dict(
            {
                "language": "python",
                "max_iterations": 2,
                "llm": {"models": [{"name": "gemini-2.5-flash", "weight": 1}]},
            }
        )
        self.output = self.root / "bundle"

    def export(self, **options):
        arguments = {
            "title": "Offline bundle",
            "problem_description": "Count bytes in a seed. 中文。",
            "objective_metric": "score",
            **options,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return export_to_alphaevolve(
                self.program, self.evaluator, self.config, self.output, **arguments
            )

    def check(self, bundle=None):
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir(exist_ok=True)
        return subprocess.run(
            [sys.executable, str((bundle or self.output) / "check.py")],
            cwd=elsewhere,
            env=environment,
            capture_output=True,
            text=True,
            timeout=20,
        )

    def cli_arguments(self):
        self.yaml = self.inputs / "config.yaml"
        # Deliberately omit language, as existing Python example YAMLs do.
        self.yaml.write_text(
            "max_iterations: 2\nllm:\n  primary_model: local-model\n", encoding="utf-8"
        )
        self.instructions = self.inputs / "instructions.md"
        self.instructions.write_text("Count source bytes. 中文。\n", encoding="utf-8")
        return [
            str(self.program),
            str(self.evaluator),
            "--config",
            str(self.yaml),
            "--output-dir",
            str(self.output),
            "--title",
            "CLI example",
            "--instructions",
            str(self.instructions),
            "--objective-metric",
            "score",
            "--model-map",
            "local-model=gemini-2.5-flash",
        ]

    def test_bundle_has_only_expected_files_and_preserves_source_bytes(self):
        before = copy.deepcopy(self.config)
        output = self.export()
        self.assertEqual(output, self.output.absolute())
        self.assertEqual(
            {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()},
            {
                "src/program.py",
                "src/openevolve_evaluator.py",
                "src/evaluate.py",
                "src/__init__.py",
                "experiment.json",
                "evaluation.json",
                "check.py",
                "run_alphaevolve.py",
                "README.md",
            },
        )
        self.assertEqual((output / "src/program.py").read_bytes(), self.program.read_bytes())
        self.assertEqual(
            (output / "src/openevolve_evaluator.py").read_bytes(), self.evaluator.read_bytes()
        )
        experiment = json.loads((output / "experiment.json").read_text())
        self.assertEqual(experiment["run_settings"], {"max_programs": 2})
        self.assertEqual(experiment["problem_description"], "Count bytes in a seed. 中文。")
        self.assertEqual(self.config, before)
        self.assertNotIn(str(self.inputs), (output / "README.md").read_text())

    def test_relocated_bundle_runs_without_original_files_or_original_working_directory(self):
        expected_score = len(self.program.read_bytes())
        self.export()
        moved = self.root / "relocated"
        self.output.rename(moved)
        shutil.rmtree(self.inputs)
        completed = self.check(moved)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["scores"]["scores"], [{"metric": "score", "score": expected_score}])
        self.assertIn("evaluator diagnostic", completed.stderr)
        self.assertNotIn("evaluator diagnostic", completed.stdout)

    def test_export_does_not_import_execute_sources_or_connect_to_network(self):
        marker = self.root / "must-not-exist"
        side_effect = f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
        self.program.write_text(side_effect + "x = 1\n", encoding="utf-8")
        self.evaluator.write_text(side_effect + EVALUATOR, encoding="utf-8")
        with patch("socket.socket.connect", side_effect=AssertionError("network forbidden")):
            self.export()
        self.assertFalse(marker.exists())

    def test_explicit_dependency_file_is_copied_without_discovering_other_files(self):
        dependencies = self.inputs / "deps.txt"
        dependencies.write_bytes(b"# reviewed dependencies\r\nnumpy>=1.22\r\n")
        (self.inputs / "private-data.txt").write_text("not selected")
        self.export(requirements_file=dependencies)
        self.assertEqual((self.output / "requirements.txt").read_bytes(), dependencies.read_bytes())
        self.assertFalse((self.output / "private-data.txt").exists())

    def test_existing_directory_file_and_symlinks_are_preserved(self):
        for kind in ("directory", "file", "symlink", "dangling-symlink"):
            with self.subTest(kind=kind):
                destination = self.root / kind
                if kind == "directory":
                    destination.mkdir()
                    (destination / "keep.txt").write_text("keep")
                elif kind == "file":
                    destination.write_text("keep")
                else:
                    destination.symlink_to(
                        self.root / ("inputs" if kind == "symlink" else "missing")
                    )
                with self.assertRaises(FileExistsError):
                    export_to_alphaevolve(
                        self.program,
                        self.evaluator,
                        self.config,
                        destination,
                        title="Example",
                        problem_description="Problem",
                        objective_metric="score",
                    )
                if kind == "directory":
                    self.assertEqual((destination / "keep.txt").read_text(), "keep")
                elif kind == "file":
                    self.assertEqual(destination.read_text(), "keep")
                else:
                    self.assertTrue(destination.is_symlink())

    def test_write_error_and_interrupt_remove_only_new_output(self):
        original_write = Path.write_bytes
        for error in (OSError("test disk failure"), KeyboardInterrupt()):
            calls = 0

            def fail_write(path, data):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise error
                return original_write(path, data)

            with self.subTest(error=type(error)), patch.object(Path, "write_bytes", fail_write):
                with self.assertRaises(type(error)):
                    self.export()
            self.assertFalse(self.output.exists())
            self.assertEqual(self.evaluator.read_text(), EVALUATOR)

    def test_destination_created_by_another_writer_is_not_removed(self):
        original_mkdir = Path.mkdir

        def create_competing_directory(path, *args, **kwargs):
            if path == self.output:
                original_mkdir(path)
                (path / "keep.txt").write_text("other writer")
                raise FileExistsError("destination was claimed")
            return original_mkdir(path, *args, **kwargs)

        with patch.object(Path, "mkdir", create_competing_directory):
            with self.assertRaises(FileExistsError):
                self.export()
        self.assertEqual((self.output / "keep.txt").read_text(), "other writer")

    def test_invalid_python_encodings_syntax_and_scopes_are_rejected_before_writes(self):
        for source in (b"def invalid(:\n", b"return 1\n", b"# coding: latin-1\nx=1\n", b"\xff"):
            self.program.write_bytes(source)
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.export()
            self.assertFalse(self.output.exists())

    def test_bom_utf8_is_preserved(self):
        source = b"\xef\xbb\xbf7\r\n"
        self.program.write_bytes(source)
        self.export()
        self.assertEqual((self.output / "src/program.py").read_bytes(), source)
        self.assertEqual(self.check().returncode, 0)

    def test_relative_and_sibling_imports_are_rejected_before_writes(self):
        (self.inputs / "helper.py").write_text("x = 1")
        (self.inputs / "local_package").mkdir()
        for imported in (
            "from .helper import x\n",
            "from . import helper\n",
            "import helper\n",
            "from helper import x\n",
            "import local_package.module\n",
        ):
            self.evaluator.write_text(imported + EVALUATOR, encoding="utf-8")
            with self.subTest(imported=imported), self.assertRaisesRegex(ValueError, "imports"):
                self.export()
            self.assertFalse(self.output.exists())

    def test_evaluator_must_accept_one_path_synchronously(self):
        for source in (
            "async def evaluate(path): return {}\n",
            "def other(path): return {}\n",
            "def evaluate(): return {}\n",
            "def evaluate(path, other): return {}\n",
            "def evaluate(path, *, other): return {}\n",
        ):
            self.evaluator.write_text(source)
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "Evaluator"):
                self.export()
            self.assertFalse(self.output.exists())

    def test_optional_evaluator_arguments_are_supported(self):
        self.evaluator.write_text("def evaluate(path, other=None): return {'score': 2}\n")
        self.export()
        self.assertEqual(self.check().returncode, 0)

    def test_invalid_config_and_adapter_options_leave_no_output(self):
        for options in ({"failure_score": float("nan")}, {"objective_metric": "error"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.export(**options)
            self.assertFalse(self.output.exists())
        self.config.llm.models[0].name = "local-model"
        with self.assertRaises(ValueError):
            self.export()
        self.assertFalse(self.output.exists())

    def test_missing_inputs_and_dependencies_leave_no_partial_output(self):
        with self.assertRaises(FileNotFoundError):
            self.export(requirements_file=self.inputs / "missing.txt")
        self.assertFalse(self.output.exists())
        self.program.unlink()
        with self.assertRaises(FileNotFoundError):
            self.export()
        self.assertFalse(self.output.exists())

    def test_missing_destination_parent_does_not_create_it(self):
        self.output = self.root / "missing-parent" / "bundle"
        with self.assertRaises(FileNotFoundError):
            self.export()
        self.assertFalse(self.output.parent.exists())

    def test_failed_seed_check_has_nonzero_status_and_sanitized_json(self):
        self.evaluator.write_text("def evaluate(path): raise RuntimeError('private-detail')\n")
        self.export(failure_score=-123.0)
        result = self.check()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(json.loads(result.stdout)["scores"]["scores"][0]["score"], -123.0)
        self.assertNotIn("private-detail", result.stdout)

    def test_unicode_and_code_like_metric_names_remain_data(self):
        metric = "score'); raise Exception('no'); #\n中文🚀"
        self.evaluator.write_text(
            f"def evaluate(path): return {{{metric!r}: 7}}\n", encoding="utf-8"
        )
        self.export(objective_metric=metric)
        self.assertNotIn(metric, (self.output / "src/evaluate.py").read_text())
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["scores"]["scores"][0]["metric"], metric)

    def test_cli_exports_with_explicit_model_mapping_and_preserves_yaml(self):
        arguments = self.cli_arguments()
        original = self.yaml.read_bytes()
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()) as output:
            warnings.simplefilter("ignore", UserWarning)
            self.assertEqual(main(arguments), 0)
        self.assertEqual(output.getvalue().strip(), str(self.output))
        self.assertEqual(self.yaml.read_bytes(), original)
        self.assertEqual(self.check().returncode, 0)
        experiment = json.loads((self.output / "experiment.json").read_text())
        self.assertEqual(experiment["program_language"], "python")
        self.assertEqual(experiment["generation_settings"]["models"][0]["name"], "gemini-2.5-flash")

    def test_cli_invalid_or_duplicate_mappings_fail_without_output(self):
        arguments = self.cli_arguments()
        for entry in ("not-a-mapping", "local-model=gemini-other"):
            with self.subTest(entry=entry), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    main(arguments + ["--model-map", entry])
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(self.output.exists())

    def test_cli_malformed_yaml_is_an_argument_error_without_partial_output(self):
        arguments = self.cli_arguments()
        self.yaml.write_text("llm: [\n")
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit) as caught:
                main(arguments)
        self.assertEqual(caught.exception.code, 2)
        self.assertNotIn("Traceback", errors.getvalue())
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
