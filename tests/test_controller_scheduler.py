from __future__ import annotations

import pytest

from openevolve.config import Config, ControllerSchedulerConfig
from openevolve.database import ProgramDatabase
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
            "database": {
                "controller_scheduler": {
                    "enabled": True,
                    "exploitation_weight": 1.0,
                    "underexplored_weight": 0.2,
                    "validity_weight": 0.5,
                    "diversity_weight": 0.5,
                    "token_efficiency_weight": 0.5,
                    "rejection_penalty": 0.2,
                    "minimum_calls": 3,
                }
            }
        }
    )

    assert isinstance(config.database.controller_scheduler, ControllerSchedulerConfig)
    assert config.database.controller_scheduler.enabled is True
    assert config.database.controller_scheduler.minimum_calls == 3


def test_controller_scheduler_rejects_negative_weight():
    config = Config()
    config.database.controller_scheduler.exploitation_weight = -0.1

    with pytest.raises(ValueError, match="finite and nonnegative"):
        config.validate()


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
    selected = controller._select_controller_island(
        {0: [], 1: [], 2: []},
        batch_size=3,
    )

    assert selected == 2
    assert controller._controller_island_state[2]["calls"] == 1
    assert controller._controller_island_state[2]["tokens"] == 20


def test_disabled_scheduler_preserves_default_configuration():
    config = Config()

    assert config.database.controller_scheduler.enabled is False
    config.validate()
