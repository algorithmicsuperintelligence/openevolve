"""Alternating evolution against frozen cohorts of opposing champions."""

import asyncio
import copy
import hashlib
import json
import logging
import math
import signal
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Tuple

from openevolve.api import _prepare_evaluator
from openevolve.config import Config
from openevolve.controller import OpenEvolve
from openevolve.database import Program
from openevolve.utils.code_utils import extract_code_language


@dataclass(frozen=True)
class Population:
    """An initial program file and an independent engine configuration."""

    initial_program: Path
    config: Config


@dataclass(frozen=True)
class Opponent:
    """A frozen opponent; the evaluator decides how to execute its code."""

    id: str
    code: str
    language: str


@dataclass
class CoevolutionResult:
    """Champions have scores relative to their own last phase's opponents."""

    champions: Dict[str, Program]
    completed_phases: int
    checkpoint_path: str


CompetitiveEvaluator = Callable[[str, str, Tuple[Opponent, ...]], Dict[str, float]]


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _config_identity(config: Config) -> dict:
    # Credentials may rotate without changing the experiment. Never save them.
    def without_keys(value):
        if isinstance(value, dict):
            return {k: without_keys(v) for k, v in value.items() if k != "api_key"}
        if isinstance(value, list):
            return [without_keys(v) for v in value]
        return value

    return without_keys(config.to_dict())


def _write_checkpoint(path: Path, state: dict) -> None:
    # Commit both population references and the phase cursor together. A failed
    # phase leaves the previous checkpoint intact; external calls can be repeated.
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(state, stream, indent=2, allow_nan=False)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _champion_history(history: list, champion: Program, limit: int) -> list:
    # Refresh recency for an existing code snapshot, without duplicating its weight.
    history = [item for item in history if item["code"] != champion.code]
    history.append(asdict(Opponent(champion.id, champion.code, champion.language)))
    return history[-limit:]


def run_coevolution(
    populations: Mapping[str, Population],
    evaluator: CompetitiveEvaluator,
    *,
    output_dir: str,
    evaluation_id: str,
    rounds: int,
    iterations_per_phase: int = 10,
    opponent_count: int = 4,
    resume: bool = False,
) -> CoevolutionResult:
    """Alternate two existing engines, refreshing fitness before each phase.

    Mapping insertion order defines which population moves first. Each round
    runs one phase per population. ``rounds`` is the total target, including
    completed rounds when resuming. Each phase uses the other population's most
    recent distinct champions, including its seed until that snapshot ages out.

    ``evaluator(path, population_name, opponents)`` returns metrics including a
    finite ``combined_score`` (larger is better for that population). It must be
    serializable by cloudpickle for the existing engine's process workers. The
    caller supplies ``evaluation_id`` to identify the evaluator, task data and
    scoring definition; change it when any of those change. Resume checks that
    identity, engine configurations, seed code and phase settings.

    Checkpoints commit completed phases only. A failed phase restarts from its
    previous population and opponent cohort, not from partial worker output.
    This is a single-writer API; evaluator side effects are not exactly-once.
    Programs and evaluators execute with the same trust requirements as ordinary
    OpenEvolve runs. No changes are made to single-population evolution.
    """
    if len(populations) != 2 or any(not name for name in populations):
        raise ValueError("Provide exactly two named populations")
    if not evaluation_id:
        raise ValueError("evaluation_id must identify the evaluator and task data")
    if rounds < 1 or iterations_per_phase < 1 or opponent_count < 1:
        raise ValueError("rounds, iterations_per_phase and opponent_count must be positive")
    specs = dict(populations)
    for spec in specs.values():
        if spec.config.database.db_path:
            raise ValueError("Population db_path must be unset; output_dir owns both populations")
    return asyncio.run(
        _run_coevolution(
            specs,
            evaluator,
            Path(output_dir),
            evaluation_id,
            rounds,
            iterations_per_phase,
            opponent_count,
            resume,
        )
    )


async def _run_coevolution(
    populations,
    evaluator,
    output_dir,
    evaluation_id,
    rounds,
    iterations_per_phase,
    opponent_count,
    resume,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "coevolution.json"
    names = list(populations)
    seeds = {name: Path(spec.initial_program).read_text() for name, spec in populations.items()}
    identity = {
        "evaluation_id": evaluation_id,
        "populations": [
            {
                "name": name,
                "seed": _digest(seeds[name]),
                "suffix": Path(spec.initial_program).suffix,
                "config": _digest(_config_identity(spec.config)),
            }
            for name, spec in populations.items()
        ],
        "iterations_per_phase": iterations_per_phase,
        "opponent_count": opponent_count,
    }
    if resume:
        state = json.loads(checkpoint.read_text())
        if state["version"] != 1 or state["identity"] != identity:
            raise ValueError("Checkpoint context differs from the requested experiment")
    else:
        if checkpoint.exists():
            raise FileExistsError("Checkpoint already exists; use resume=True or a new output_dir")
        state = {
            "version": 1,
            "identity": identity,
            "completed_phases": 0,
            "populations": {},
            "history": [],
            "opponents": {},
        }
        for name, spec in populations.items():
            seed = Program(
                id=f"{name}-seed",
                code=seeds[name],
                language=spec.config.language or extract_code_language(seeds[name]),
            )
            state["populations"][name] = {
                "programs": [seed.to_dict()],
                "last_iteration": 0,
                "champion": seed.id,
            }
            state["opponents"][name] = _champion_history([], seed, opponent_count)
        _write_checkpoint(checkpoint, state)

    while state["completed_phases"] < rounds * 2:
        phase = state["completed_phases"]
        name, other = names[phase % 2], names[(phase + 1) % 2]
        spec = populations[name]
        opponents = tuple(Opponent(**item) for item in state["opponents"][other])
        context = {
            "evaluation_id": evaluation_id,
            "population": name,
            "opponents": [asdict(item) for item in opponents],
        }
        context_id = _digest(context)

        # Bind values now: both re-evaluation and every worker in this phase use
        # the same cohort, even though the other population changes next phase.
        def evaluate(path, name=name, opponents=opponents):
            return evaluator(path, name, opponents)

        # Separate attempt directories keep abandoned partial output out of resume.
        attempt = Path(tempfile.mkdtemp(prefix=f"phase_{phase:06d}_", dir=output_dir))
        (attempt / "opponents.json").write_text(json.dumps(context, indent=2))
        temporary_files = []
        evaluation_file = _prepare_evaluator(evaluate, str(attempt), temporary_files)
        config = copy.deepcopy(spec.config)
        config.evaluator.cascade_evaluation = False
        config.database.artifacts_base_path = str(attempt / "artifacts")
        if config.random_seed is not None:
            config.random_seed += phase
        old = state["populations"][name]
        signals = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        root_logger = logging.getLogger()
        handlers, level = set(root_logger.handlers), root_logger.level
        engine = None
        try:
            seed_file = attempt / ("initial" + Path(spec.initial_program).suffix)
            seed_file.write_text(seeds[name])
            engine = OpenEvolve(str(seed_file), evaluation_file, config, str(attempt))
            # Rebuild the grid, archive, feature statistics and best references
            # from freshly evaluated retained programs, rather than changing only
            # metrics under indices that still encode the previous matchup.
            # UUIDs are not a reproducible tie-breaker for rebuilding the archive.
            for saved in sorted(
                old["programs"],
                key=lambda item: (item["iteration_found"], item["generation"], item["code"]),
            ):
                program = Program.from_dict(copy.deepcopy(saved))
                program.metrics = await engine.evaluator.evaluate_program(program.code, program.id)
                score = program.metrics.get("combined_score")
                if not isinstance(score, (int, float)) or not math.isfinite(score):
                    raise ValueError(
                        f"Re-evaluation failed for {name}/{program.id}: {program.metrics}"
                    )
                program.artifacts_json = None
                program.artifact_dir = None
                program.prompts = None
                program.metadata["coevolution_context"] = context_id
                engine.database.add(program, target_island=program.metadata.get("island", 0))
                artifacts = engine.evaluator.get_pending_artifacts(program.id)
                if artifacts and program.id in engine.database.programs:
                    engine.database.store_artifacts(program.id, artifacts)
            # run() treats an already populated database's cursor as its next iteration.
            engine.database.last_iteration = old["last_iteration"] + 1
            for key, value in old.get("schedule", {}).items():
                setattr(engine.database, key, copy.deepcopy(value))
            champion = await engine.run(iterations=iterations_per_phase)
            if champion is None or not math.isfinite(
                champion.metrics.get("combined_score", float("nan"))
            ):
                raise ValueError(f"No valid champion for {name}")
            updated = copy.deepcopy(state)
            for program in engine.database.programs.values():
                program.metadata["coevolution_context"] = context_id
            updated["populations"][name] = {
                "programs": [p.to_dict() for p in engine.database.programs.values()],
                "last_iteration": engine.database.last_iteration,
                "champion": champion.id,
                "schedule": {
                    key: copy.deepcopy(getattr(engine.database, key))
                    for key in ("current_island", "island_generations", "last_migration_generation")
                },
            }
            updated["opponents"][name] = _champion_history(
                state["opponents"][name], champion, opponent_count
            )
            updated["history"].append(
                {
                    "phase": phase,
                    "population": name,
                    "context": context_id,
                    "opponent_ids": [item.id for item in opponents],
                    "refreshed_programs": len(old["programs"]),
                    "champion": champion.id,
                    "metrics": champion.metrics,
                    "output_dir": str(attempt.resolve()),
                }
            )
            updated["completed_phases"] = phase + 1
            _write_checkpoint(checkpoint, updated)
            state = updated
        finally:
            for sig, handler in signals.items():
                signal.signal(sig, handler)
            if engine and engine.evolution_tracer:
                engine.evolution_tracer.close()
            for handler in set(root_logger.handlers) - handlers:
                root_logger.removeHandler(handler)
                handler.close()
            root_logger.setLevel(level)

    champions = {}
    for name, population in state["populations"].items():
        champions[name] = Program.from_dict(
            next(p for p in population["programs"] if p["id"] == population["champion"])
        )
    return CoevolutionResult(champions, state["completed_phases"], str(checkpoint))
