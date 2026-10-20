"""Offline comparison using real engines/workers and scripted, not LLM, proposals."""

import argparse
import copy
import json
import logging
import random
import signal
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from openevolve import Opponent, Population, run_coevolution, run_evolution
from openevolve.config import Config, LLMModelConfig


def policy(weights):
    return (
        "# EVOLVE-BLOCK-START\ndef allocate():\n    return "
        + repr(weights)
        + "\n# EVOLVE-BLOCK-END\n"
    )


def allocation(code):
    namespace = {}
    exec(compile(code, "<policy>", "exec"), namespace)
    values = namespace["allocate"]()
    if len(values) != 3 or min(values) < 0 or abs(sum(values) - 1) > 1e-6:
        raise ValueError("Expected a probability distribution over three channels")
    return values


def evaluator(log_path):
    def evaluate(path, population, opponents):
        own = allocation(Path(path).read_text())
        coverage = [sum(a * b for a, b in zip(own, allocation(p.code))) for p in opponents]
        # Defender maximizes detection, attacker maximizes evasion. Use the same
        # worst-opponent objective for both competition and static baselines.
        scores = coverage if population == "defender" else [1 - value for value in coverage]
        with open(log_path, "a") as stream:
            stream.write(json.dumps({"population": population, "matches": len(opponents)}) + "\n")
        return {"combined_score": min(scores)}

    return evaluate


class ProposalServer(ThreadingHTTPServer):
    def reset(self, seed):
        # The same complete candidate cycle and seed are used by every treatment.
        self.proposals = [(a / 6, b / 6, (6 - a - b) / 6) for a in range(7) for b in range(7 - a)]
        random.Random(seed).shuffle(self.proposals)
        self.attacks = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        random.Random(seed).shuffle(self.attacks)
        self.counts = Counter()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        name = request["model"]
        index = self.server.counts[name]
        self.server.counts[name] += 1
        choices = (
            self.server.attacks
            if name == "attacker" and not self.server.mixed_attacker
            else self.server.proposals
        )
        code = policy(choices[index % len(choices)])
        data = json.dumps(
            {
                "id": f"offline-{name}-{index}",
                "object": "chat.completion",
                "created": 0,
                "model": name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": f"```python\n{code}```"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def config(port, name, seed):
    cfg = Config()
    cfg.random_seed = seed
    cfg.language = "python"
    cfg.diff_based_evolution = False
    cfg.checkpoint_interval = 1000
    cfg.database.num_islands = 1
    cfg.database.population_size = 32
    cfg.database.archive_size = 16
    cfg.evaluator.cascade_evaluation = False
    cfg.evaluator.max_retries = 0
    cfg.evaluator.parallel_evaluations = 1
    cfg.llm.models = [
        LLMModelConfig(
            name=name,
            api_base=f"http://127.0.0.1:{port}/v1",
            api_key="offline",
            retries=0,
            weight=1.0,
        )
    ]
    cfg.llm.evaluator_models = []
    return cfg


def measure(code):
    # Declared before optimization: all three pure attacks. No optimization
    # routine reads this assessment function. These attacks may also occur in
    # training, so this is a fixed assessment set, not an unseen test set.
    weights = allocation(code)
    return {
        "allocation": weights,
        "worst_channel_detection": min(weights),
        "mean_detection": sum(weights) / 3,
    }


def counts(path):
    totals = Counter()
    for line in path.read_text().splitlines():
        entry = json.loads(line)
        totals[entry["population"]] += entry["matches"]
    return dict(totals)


def compare(root, seed, server):
    root.mkdir(parents=True, exist_ok=False)
    initial = root / "initial.py"
    initial.write_text(policy((1.0, 0.0, 0.0)))
    specs = {
        name: Population(initial, config(server.server_port, name, seed))
        for name in ("defender", "attacker")
    }
    initial_opponent = (Opponent("initial", initial.read_text(), "python"),)
    server.reset(seed)
    competitive_log = root / "competitive_matches.jsonl"
    result = run_coevolution(
        specs,
        evaluator(competitive_log),
        output_dir=str(root / "competitive"),
        evaluation_id="three-channel-allocation-v1",
        rounds=3,
        iterations_per_phase=28,
        opponent_count=4,
    )
    budget = counts(competitive_log)
    competitive = {
        **measure(result.champions["defender"].code),
        "match_calls": budget,
        "generation_calls": dict(server.counts),
    }

    def baseline(label, iterations):
        server.reset(seed)
        log = root / f"{label}_matches.jsonl"
        evaluate = evaluator(log)
        results = {}
        for name, spec in specs.items():

            def static(path, name=name):
                return evaluate(path, name, initial_opponent)

            results[name] = run_evolution(
                initial,
                static,
                config=copy.deepcopy(spec.config),
                iterations=iterations[name],
                output_dir=str(root / label / name),
                cleanup=True,
            )
        return {
            **measure(results["defender"].best_code),
            "match_calls": counts(log),
            "generation_calls": dict(server.counts),
        }

    generation_matched = baseline("static_generation_matched", dict.fromkeys(specs, 84))
    # Duplicate/unchanged proposals can be skipped by the existing engine. Give
    # this baseline twice the match budget in proposals, then report actual calls.
    match_matched = baseline("static_larger_budget", {name: 2 * budget[name] for name in specs})
    if any(match_matched["match_calls"][name] < budget[name] for name in specs):
        raise RuntimeError("Baseline did not spend the required match budget")
    return {
        "seed": seed,
        "competitive": competitive,
        "static_generation_matched": generation_matched,
        "static_larger_budget": match_matched,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--mixed-attacker",
        action="store_true",
        help="Also allow mixed attacks; a separate game without a robustness-gain guarantee",
    )
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    server = ProposalServer(("127.0.0.1", 0), Handler)
    server.mixed_attacker = args.mixed_attacker
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    signals = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        results = [compare(args.output / f"seed_{seed}", seed, server) for seed in args.seeds]
        report = {
            "generation": "scripted grid proposals, not an LLM",
            "mixed_attacker": args.mixed_attacker,
            "results": results,
        }
        (args.output / "results.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        for sig, handler in signals.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
