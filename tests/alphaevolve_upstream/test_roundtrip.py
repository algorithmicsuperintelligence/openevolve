"""Offline execution equivalence of OE examples through exported AE bundles."""

import hashlib
import json
import math
import sys
import unittest

from test_examples import REPO, SUITE, OfflineSubprocessTest, json_lines

MARKER = "# AlphaEvolve upstream CI: evaluated fixed proposal."


class TestBundleRoundTrip(OfflineSubprocessTest):
    def run_case(self, name):
        example = REPO / "examples" / name
        originals = {
            file: hashlib.sha256((example / file).read_bytes()).hexdigest()
            for file in ("initial_program.py", "evaluator.py")
        }
        completed = self.run_python([str(SUITE / "run_roundtrip.py"), name, str(self.workspace)])
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.check_offline()
        summary = json.loads((self.workspace / "roundtrip-result.json").read_text())
        self.assertTrue(summary["sources_unchanged"] and summary["inputs_removed"])
        self.assertEqual(
            originals,
            {file: hashlib.sha256((example / file).read_bytes()).hexdigest() for file in originals},
        )
        self.assertFalse((self.workspace / "inputs").exists())
        self.assertFalse((self.workspace / "bundle").exists())
        warnings = "\n".join(summary["export_warnings"])
        for field in ("database.num_islands", "evaluator.cascade_evaluation", "llm.api_base"):
            self.assertIn(field, warnings)
        if name == "circle_packing":
            self.assertIn("Model names explicitly remapped", warnings)

        seed = (example / "initial_program.py").read_text()
        expected_child = seed.replace(
            "# EVOLVE-BLOCK-START\n", f"# EVOLVE-BLOCK-START\n{MARKER}\n", 1
        )
        expected_hashes = {
            hashlib.sha256(code.encode()).hexdigest() for code in (seed, expected_child)
        }
        traces = {}
        for label in ("direct", "roundtrip"):
            result = summary["results"][label]
            metrics = result["metrics"]
            self.assertNotIn("error", metrics)
            self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
            self.assertGreater(result["best_score"], 0)
            self.assertEqual(result["best_score"], metrics["combined_score"])
            self.assertIn(result["code"], (seed, expected_child))
            traces[label] = json_lines(self.workspace / (label + ".jsonl"))
            self.assertEqual(len(traces[label]), 1, "Must evaluate one actual proposal per run")
            trace = traces[label][0]
            self.assertEqual(trace["parent_code"], seed)
            self.assertEqual(trace["child_code"], expected_child)
            self.assertNotIn("error", trace["child_metrics"])
            self.assertGreater(trace["child_metrics"]["combined_score"], 0)
            evaluations = json_lines(self.workspace / (label + "-evaluations.jsonl"))
            self.assertEqual(len(evaluations), 2, "Both seed and child must be evaluated")
            self.assertEqual({item["hash"] for item in evaluations}, expected_hashes)
            self.assertTrue(any(item["pid"] != summary["pid"] for item in evaluations))
            self.assertTrue((self.workspace / (label + "-output/best/best_program.py")).is_file())

        # Runtime is inherently variable. All other returned numeric metrics must
        # survive the evaluator-envelope round trip, for seed AND evaluated child.
        for left, right in (
            (summary["results"]["direct"]["metrics"], summary["results"]["roundtrip"]["metrics"]),
            (traces["direct"][0]["parent_metrics"], traces["roundtrip"][0]["parent_metrics"]),
            (traces["direct"][0]["child_metrics"], traces["roundtrip"][0]["child_metrics"]),
        ):
            self.assertEqual(set(left), set(right))
            for metric in left:
                if metric != "eval_time":
                    self.assertTrue(
                        math.isclose(left[metric], right[metric], rel_tol=1e-9, abs_tol=1e-12),
                        metric,
                    )
        calls = json_lines(self.workspace / "calls.jsonl")
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(item["pid"] != summary["pid"] for item in calls))
        self.assertTrue(all(item["model"] in dict(summary["models"]) for item in calls))
        print(
            json.dumps(
                {
                    "example": name,
                    "direct_score": summary["results"]["direct"]["best_score"],
                    "roundtrip_score": summary["results"]["roundtrip"]["best_score"],
                    "evaluated_proposals": len(calls),
                    "network_or_cloud_attempts": 0,
                }
            ),
            flush=True,
        )

    def test_function_minimization_roundtrip(self):
        self.run_case("function_minimization")

    def test_circle_packing_roundtrip(self):
        self.run_case("circle_packing")

    def test_unguarded_helper_fails_before_executing_examples(self):
        completed = self.run_python(
            [
                "-c",
                "import runpy, sitecustomize; sitecustomize.ACTIVE = False; "
                f"runpy.run_path({str(SUITE / 'run_roundtrip.py')!r}, run_name='__main__')",
            ]
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("offline tripwires", completed.stderr)
        self.assertEqual(list(self.workspace.glob("*-output")), [])


if __name__ == "__main__":
    unittest.main()
