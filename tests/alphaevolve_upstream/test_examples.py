"""Opt-in integration checks against pinned, unchanged official AE examples."""

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SUITE = Path(__file__).resolve().parent
REPO = SUITE.parents[1]
MARKER = "# AlphaEvolve upstream CI: evaluated fixed proposal."
EXAMPLES = {
    "circle_packing": ("sum_of_radii", 0.9414554065860808),
    "tsp": ("neg_tour_length", -7.223903565915199),
    "signal_processing": ("overall_score", 0.49904861783269006),
}


def source_hashes(example):
    paths = list((example / "src").rglob("*.py"))
    if (example / "instructions.md").is_file():
        paths.append(example / "instructions.md")
    return {
        str(path.relative_to(example)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


class OfflineSubprocessTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="oe_upstream_test_")
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.audit_dir = self.workspace / "audit"
        self.audit_dir.mkdir()

    def run_python(self, arguments):
        # Do not forward developer/CI secrets, proxies, SDK paths or Python startup
        # settings. The only credential-like values are deliberate dummy traps.
        environment = {
            key: os.environ[key]
            for key in ("PATH", "HOME", "TMPDIR", "SYSTEMROOT", "WINDIR")
            if key in os.environ
        }
        environment.update(
            PYTHONPATH=os.pathsep.join((str(SUITE / "support"), str(REPO))),
            PYTHONNOUSERSITE="1",
            MPLBACKEND="Agg",
            MPLCONFIGDIR=str(self.workspace / "matplotlib"),
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            OE_INTEROP_AUDIT_DIR=str(self.audit_dir),
            OE_INTEROP_CALL_LOG=str(self.workspace / "calls.jsonl"),
            OPENAI_API_KEY="dummy-ambient-key-must-not-be-used",
            GOOGLE_APPLICATION_CREDENTIALS=str(self.workspace / "nonexistent-credentials.json"),
            SAVE_PROGRAM_DIR=str(self.workspace / "must-not-write-here"),
        )
        return subprocess.run(
            [sys.executable, *arguments],
            cwd=self.workspace,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def audits(self):
        return [event for path in self.audit_dir.glob("*.jsonl") for event in json_lines(path)]

    def check_offline(self):
        events = self.audits()
        self.assertTrue(any(event["kind"] == "active" for event in events), "Tripwires not loaded")
        self.assertEqual([event for event in events if event["kind"] != "active"], [])


class TestOfflineTripwires(OfflineSubprocessTest):
    def test_caught_network_attempts_still_fail_offline_validation(self):
        result = self.run_python(
            [
                "-c",
                "import socket\ntry: socket.create_connection(('127.0.0.1', 1))\nexcept AssertionError: pass",
            ]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        with self.assertRaises(AssertionError):
            self.check_offline()
        self.assertTrue(any(event["kind"].startswith("socket.") for event in self.audits()))

    def test_caught_cloud_sdk_or_auth_imports_still_fail_offline_validation(self):
        result = self.run_python(
            [
                "-c",
                "import importlib\nfor name in ('alpha_evolve', 'google.auth'):\n"
                "    try: importlib.import_module(name)\n    except AssertionError: pass",
            ]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        with self.assertRaises(AssertionError):
            self.check_offline()
        self.assertEqual(sum(event["kind"] == "cloud_import" for event in self.audits()), 2)


class TestUpstreamExamples(OfflineSubprocessTest):
    @classmethod
    def setUpClass(cls):
        directory = os.environ.get("ALPHAEVOLVE_UPSTREAM_DIR")
        if not directory:
            raise unittest.SkipTest("Set ALPHAEVOLVE_UPSTREAM_DIR to opt in to upstream examples")
        cls.upstream = Path(os.path.realpath(directory))
        if not cls.upstream.is_relative_to(REPO.parent):
            raise RuntimeError("Keep the upstream checkout inside the OE parent workspace")
        revision = (SUITE / "revision.txt").read_text().strip()
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise RuntimeError("revision.txt must contain a full upstream commit SHA")
        actual = subprocess.check_output(
            ["git", "-C", str(cls.upstream), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != revision:
            raise RuntimeError(f"Expected pinned upstream revision {revision}, got {actual}")
        paths = [f"examples/{name}" for name in EXAMPLES]
        dirty = subprocess.check_output(
            [
                "git",
                "-C",
                str(cls.upstream),
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                *paths,
            ],
            text=True,
        )
        if dirty:
            raise RuntimeError("Use a clean checkout of the pinned upstream examples")

    def run_case(self, name, iterations):
        example = self.upstream / "examples" / name
        original = source_hashes(example)
        self.assertIn("src/program.py", original)
        self.assertIn("src/evaluate.py", original)
        result = self.run_python(
            [str(SUITE / "run_example.py"), str(example), str(iterations), str(self.workspace)]
        )
        self.assertEqual(result.returncode, 0, result.stderr[-8000:] + result.stdout[-2000:])
        self.check_offline()
        self.assertEqual(source_hashes(example), original)
        output = self.workspace / "run"
        self.assertEqual(source_hashes(output), original)
        self.assertFalse((output / ".env").exists())
        self.assertFalse((self.workspace / "must-not-write-here").exists())
        summary = json.loads((self.workspace / "result.json").read_text())
        metric, baseline = EXAMPLES[name]
        self.assertEqual(summary["example"], name)
        self.assertEqual(summary["generation_attempt_limit"], iterations)
        self.assertNotIn("error", summary["metrics"])
        score = summary["metrics"][metric]
        self.assertTrue(math.isfinite(score) and math.isclose(score, baseline, rel_tol=1e-6))
        self.assertEqual(summary["best_score"], score)
        self.assertTrue((output / "oe-output/best/best_program.py").is_file())
        calls = json_lines(self.workspace / "calls.jsonl")
        traces = json_lines(self.workspace / "evolution.jsonl")
        self.assertEqual(len(calls), iterations)
        self.assertEqual(len(traces), iterations)
        if iterations:
            self.assertEqual(calls[0]["model"], "fixed-offline-model")
            process = json.loads((self.workspace / "process.json").read_text())
            self.assertNotEqual(
                calls[0]["pid"], process["pid"], "Proposal must run in a real OE worker"
            )
            self.assertIn(MARKER, traces[0]["child_code"])
            self.assertEqual(traces[0]["parent_code"], (example / "src/program.py").read_text())
            self.assertNotIn("error", traces[0]["child_metrics"])
            self.assertTrue(
                math.isclose(traces[0]["child_metrics"][metric], baseline, rel_tol=1e-6)
            )
        if name == "tsp":
            images = {path.name for path in (output / "report").glob("*.png")}
            self.assertEqual(
                images,
                {
                    "best_all_instances.png",
                    "evolution_progress.png",
                    "per_instance_comparison.png",
                    "seed_vs_best.png",
                    "tour_length_distribution.png",
                },
            )
            for image in images:
                self.assertEqual((output / "report" / image).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertTrue((output / "evolved_program/main.py").is_file())
        elif name == "circle_packing":
            self.assertIn("Visualizing Rank 1:", result.stderr)
            self.assertNotIn("ERROR visualizing program", result.stderr)
            self.assertNotIn("ERROR: construct_packing", result.stderr)
        else:
            self.assertEqual(summary["metrics"]["success_rate"], 1.0)
            archives = list((output / "programs").rglob("*.json"))
            self.assertEqual(len(archives), 1)
            archived = json.loads(archives[0].read_text())
            self.assertEqual(MARKER in archived["content"]["files"][0]["content"], bool(iterations))
        print(
            json.dumps(
                {
                    "example": name,
                    "iterations": iterations,
                    "score": score,
                    "evaluated_proposals": len(traces),
                    "network_or_cloud_attempts": 0,
                }
            ),
            flush=True,
        )

    def test_circle_packing_seed(self):
        self.run_case("circle_packing", 0)

    def test_circle_packing_proposal(self):
        self.run_case("circle_packing", 1)

    def test_tsp_seed(self):
        self.run_case("tsp", 0)

    def test_tsp_proposal(self):
        self.run_case("tsp", 1)

    def test_signal_processing_seed(self):
        self.run_case("signal_processing", 0)

    def test_signal_processing_proposal(self):
        self.run_case("signal_processing", 1)


if __name__ == "__main__":
    # Explicit CI/manual execution must never go green by skipping all examples.
    if not os.environ.get("ALPHAEVOLVE_UPSTREAM_DIR"):
        raise SystemExit("ALPHAEVOLVE_UPSTREAM_DIR is required; see the interop walkthrough")
    unittest.main()
