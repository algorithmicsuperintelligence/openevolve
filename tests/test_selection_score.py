"""Selection-score conventions for exact scientific evaluators."""

import logging

from openevolve.utils.metrics_utils import get_fitness_score
from openevolve.config import DatabaseConfig
from openevolve.database import Program, ProgramDatabase


def test_selection_score_precedes_combined_score() -> None:
    metrics = {
        "selection_eligible": 1.0,
        "selection_score": 0.42,
        "combined_score": 0.99,
    }
    assert get_fitness_score(metrics) == 0.42


def test_ineligible_measurement_ranks_below_valid_zero() -> None:
    declined = get_fitness_score(
        {
            "selection_eligible": 0.0,
            "selection_score": 100.0,
            "combined_score": 100.0,
        }
    )
    valid_zero = get_fitness_score(
        {
            "selection_eligible": 1.0,
            "selection_score": 0.0,
            "combined_score": 0.0,
        }
    )
    assert declined < valid_zero


def test_legacy_combined_score_is_unchanged() -> None:
    assert get_fitness_score({"combined_score": 0.73, "other": 1.0}) == 0.73


def test_ineligible_program_is_lineage_only() -> None:
    database = ProgramDatabase(
        DatabaseConfig(
            in_memory=True,
            population_size=8,
            archive_size=4,
            num_islands=2,
        )
    )
    program = Program(
        id="declined",
        code="return 0",
        metrics={"selection_eligible": 0.0, "selection_score": 100.0},
    )
    database.add(program)

    assert "declined" in database.programs
    assert "declined" not in database.archive
    assert all("declined" not in island for island in database.islands)
    assert program.metadata["selection_ineligible"] is True


def test_best_program_log_reports_the_deciding_metric(caplog) -> None:
    database = ProgramDatabase(
        DatabaseConfig(
            in_memory=True,
            population_size=8,
            archive_size=4,
            num_islands=2,
        )
    )
    database.add(
        Program(
            id="old",
            code="return 0",
            metrics={"selection_score": 0.70, "combined_score": 0.90},
        )
    )
    with caplog.at_level(logging.INFO, logger="openevolve.database"):
        database.add(
            Program(
                id="new",
                code="return 1",
                metrics={"selection_score": 0.75, "combined_score": 0.60},
            )
        )

    message = next(
        record.message
        for record in caplog.records
        if record.message.startswith("New best program new")
    )
    assert "selection_score: 0.7000 → 0.7500, +0.0500" in message
    assert "combined_score" not in message
