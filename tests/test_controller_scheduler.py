from __future__ import annotations

import pytest

from openevolve.config import Config, ControllerSchedulerConfig
from openevolve.database import Program, ProgramDatabase
from openevolve.process_parallel import (
    ProcessParallelController,
    SerializableResult,
)


def _controller(config: Config) -> ProcessParallelController:
    return ProcessParallelController(
        config,
        "unused-evaluator.py",
        ProgramDatabase(config.database),
    )


def test_controller_scheduler_loads_from_nested_config():
    config = Config.from_dict(
        {
            "evaluator": {"parallel_evaluations": 3},
            "database": {
                    "controller_scheduler": {
                        "enabled": True,
                        "score_metric": "selection_score",
                        "diversity_artifact": "phenotype.txt",
                        "exploitation_weight": 1.0,
                        "underexplored_weight": 0.2,
                        "validity_weight": 0.5,
                        "diversity_weight": 0.5,
                        "token_efficiency_weight": 0.5,
                        "rejection_penalty": 0.2,
                        "minimum_calls": 3,
                        "leader_score_band": 0.05,
                        "parent_score_band": 0.02,
                        "reserve_islands": 2,
                        "adaptive_parallelism": 2,
                    }
                }
            }
    )

    assert isinstance(config.database.controller_scheduler, ControllerSchedulerConfig)
    assert config.database.controller_scheduler.enabled is True
    assert config.database.controller_scheduler.score_metric == "selection_score"
    assert config.database.controller_scheduler.diversity_artifact == "phenotype.txt"
    assert config.database.controller_scheduler.minimum_calls == 3
    assert config.database.controller_scheduler.leader_score_band == 0.05
    assert config.database.controller_scheduler.parent_score_band == 0.02
    assert config.database.controller_scheduler.reserve_islands == 2
    assert config.database.controller_scheduler.adaptive_parallelism == 2


def test_controller_scheduler_rejects_negative_weight():
    config = Config()
    config.database.controller_scheduler.exploitation_weight = -0.1

    with pytest.raises(ValueError, match="finite and nonnegative"):
        config.validate()


def test_controller_scheduler_rejects_negative_leader_score_band():
    config = Config()
    config.database.controller_scheduler.leader_score_band = -0.01

    with pytest.raises(ValueError, match="leader_score_band"):
        config.validate()


def test_controller_scheduler_rejects_negative_parent_score_band():
    config = Config()
    config.database.controller_scheduler.parent_score_band = -0.01

    with pytest.raises(ValueError, match="parent_score_band"):
        config.validate()


def test_controller_scheduler_rejects_invalid_reserve():
    config = Config()
    config.database.num_islands = 3
    config.database.controller_scheduler.enabled = True
    config.database.controller_scheduler.reserve_islands = 3

    with pytest.raises(ValueError, match="reserve_islands"):
        config.validate()


def test_score_band_parent_sampling_excludes_weak_programs():
    config = Config()
    config.database.num_islands = 1
    database = ProgramDatabase(config.database)
    for name, score in (("leader", 0.9), ("near", 0.895), ("weak", 0.7)):
        program = Program(
            id=name,
            code=f"def candidate(): return {score}",
            metrics={"selection_score": score},
            metadata={"island": 0},
        )
        database.programs[name] = program
        database.islands[0].add(name)

    selected = {
        database.sample_from_island_score_band(
            0,
            score_metric="selection_score",
            score_band=0.01,
            num_inspirations=1,
        )[0].id
        for _ in range(30)
    }

    assert selected <= {"leader", "near"}
    assert "weak" not in selected


def test_observed_result_allocator_prefers_productive_island():
    config = Config()
    config.database.num_islands = 3
    config.database.controller_scheduler = ControllerSchedulerConfig(
        enabled=True,
        exploitation_weight=1.0,
        minimum_calls=1,
    )
    controller = _controller(config)
    result = SerializableResult(
        child_program_dict={
            "code": "def candidate(): return 1",
            "metrics": {"combined_score": 0.9},
        },
        llm_metadata={
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 20,
            }
        },
        target_island=2,
    )

    controller._record_controller_result(2, result)
    controller.submitted_proposal_count = 1
    selected = controller._select_controller_island(
        {0: [], 1: [], 2: []},
        batch_size=3,
    )

    assert selected == 2
    assert controller._controller_island_state[2]["calls"] == 1
    assert controller._controller_island_state[2]["tokens"] == 20


def test_controller_scheduler_uses_configured_score_metric():
    config = Config()
    config.database.num_islands = 2
    config.database.controller_scheduler = ControllerSchedulerConfig(
        enabled=True,
        score_metric="selection_score",
        exploitation_weight=1.0,
        minimum_calls=1,
    )
    controller = _controller(config)
    controller._record_controller_result(
        1,
        SerializableResult(
            child_program_dict={
                "code": "def candidate(): return 1",
                "metrics": {
                    "combined_score": 0.9,
                    "selection_score": 0.4,
                },
            },
            target_island=1,
        ),
    )

    assert controller._controller_island_state[1]["best_score"] == 0.4


def test_controller_scheduler_reads_seed_and_migrated_archive_leaders():
    config = Config()
    config.database.num_islands = 2
    config.database.controller_scheduler = ControllerSchedulerConfig(
        enabled=True,
        score_metric="selection_score",
        exploitation_weight=1.0,
        minimum_calls=1,
    )
    database = ProgramDatabase(config.database)
    database.add(
        Program(
            id="initial-low",
            code="def candidate(): return 0",
            metrics={"selection_score": 0.2},
        ),
        target_island=0,
    )
    controller = ProcessParallelController(
        config,
        "unused-evaluator.py",
        database,
    )
    database.add(
        Program(
            id="migrated-high",
            code="def candidate(): return 1",
            metrics={"selection_score": 0.9},
        ),
        target_island=1,
    )
    controller.submitted_proposal_count = 1

    selected = controller._select_controller_island(
        {0: [], 1: []},
        batch_size=1,
    )

    assert selected == 1
    assert controller.controller_allocation["islands"][1]["best_score"] == 0.9


def test_controller_scheduler_keeps_post_warmup_calls_on_quality_frontier():
    config = Config()
    config.database.num_islands = 3
    config.database.controller_scheduler = ControllerSchedulerConfig(
        enabled=True,
        score_metric="selection_score",
        exploitation_weight=0.1,
        underexplored_weight=1.0,
        minimum_calls=3,
        leader_score_band=0.01,
    )
    database = ProgramDatabase(config.database)
    for island, score in enumerate((0.8, 0.795, 0.6)):
        database.add(
            Program(
                id=f"seed-{island}",
                code=f"def candidate(): return {island}",
                metrics={"selection_score": score},
            ),
            target_island=island,
        )
    controller = ProcessParallelController(
        config,
        "unused-evaluator.py",
        database,
    )
    controller.submitted_proposal_count = 3
    controller._controller_island_state[0]["calls"] = 10
    controller._controller_island_state[1]["calls"] = 9

    selected = controller._select_controller_island(
        {0: [], 1: [], 2: []},
        batch_size=1,
    )

    assert selected == 1
    assert controller.controller_allocation["leader_score_band"] == 0.01


def test_controller_scheduler_counts_evaluator_defined_behavior():
    config = Config()
    config.database.num_islands = 2
    config.database.controller_scheduler = ControllerSchedulerConfig(
        enabled=True,
        diversity_artifact="phenotype.txt",
        diversity_weight=1.0,
        minimum_calls=1,
    )
    controller = _controller(config)
    for code in ("return 1", "return 2"):
        controller._record_controller_result(
            1,
            SerializableResult(
                child_program_dict={
                    "code": code,
                    "metrics": {"combined_score": 0.5},
                },
                artifacts={"phenotype.txt": "same-measured-behavior"},
                target_island=1,
            ),
        )

    assert controller._controller_island_state[1]["accepted"] == 2
    assert len(controller._controller_island_state[1]["unique"]) == 1
    assert controller.controller_allocation["islands"][1]["distinct_behaviors"] == 1


def test_controller_scheduler_reserves_a_real_allocation_choice():
    config = Config()
    config.database.num_islands = 5
    config.evaluator.parallel_evaluations = 4
    config.database.controller_scheduler = ControllerSchedulerConfig(
        enabled=True,
        reserve_islands=1,
    )
    controller = _controller(config)

    assert controller._controller_parallelism(20) == 4
    assert controller._adaptive_controller_parallelism(20) == 4
    assert controller._controller_parallelism(2) == 2

    config.database.num_islands = 4
    config.database.controller_scheduler.adaptive_parallelism = 2
    controller = _controller(config)
    assert controller._controller_parallelism(20) == 3
    assert controller._adaptive_controller_parallelism(20) == 2
    assert controller._target_controller_parallelism(20) == 3
    controller.submitted_proposal_count = 1
    assert controller._target_controller_parallelism(20) == 2


def test_controller_scheduler_rejects_excess_adaptive_parallelism():
    config = Config()
    config.database.num_islands = 5
    config.evaluator.parallel_evaluations = 3
    config.database.controller_scheduler = ControllerSchedulerConfig(
        enabled=True,
        adaptive_parallelism=4,
    )

    with pytest.raises(ValueError, match="adaptive_parallelism"):
        config.validate()


def test_disabled_scheduler_preserves_default_configuration():
    config = Config()

    assert config.database.controller_scheduler.enabled is False
    config.validate()
