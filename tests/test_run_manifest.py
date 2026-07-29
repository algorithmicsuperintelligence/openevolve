import json
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, Mock, patch

import pytest

from openevolve.config import Config as OpenEvolveConfig
from openevolve.config import LLMModelConfig
from openevolve.controller import OpenEvolve
from openevolve.run_manifest import MANIFEST_FILENAME, RunManifest


@dataclass
class Model:
    name: str = "test-model"
    provider: str = "test-provider"
    api_key: str = "super-secret"
    api_base: str = "https://provider.example/v1"


@dataclass
class LLM:
    api_key: str = "top-level-secret"
    api_base: str = "https://top.example/v1"
    models: list[Model] = field(default_factory=lambda: [Model()])


@dataclass
class Trace:
    enabled: bool = True
    format: str = "jsonl"
    output_path: str | None = None


@dataclass
class Config:
    max_iterations: int = 3
    random_seed: int = 7
    llm: LLM = field(default_factory=LLM)
    evolution_trace: Trace = field(default_factory=Trace)
    log_dir: str = "/private/user/logs"


def test_manifest_writes_redacted_resolved_config_and_lifecycle_fields(tmp_path):
    initial = tmp_path / "initial.py"
    evaluator = tmp_path / "evaluator.py"
    initial.write_text("print('initial')", encoding="utf-8")
    evaluator.write_text("def evaluate(path): return {}", encoding="utf-8")
    (tmp_path / "checkpoints" / "checkpoint_3").mkdir(parents=True)
    (tmp_path / "evolution_trace.jsonl").write_text("{}\n", encoding="utf-8")

    manifest = RunManifest(str(tmp_path), Config(), str(initial), str(evaluator))
    manifest.write(max_iterations=3)
    manifest.write(max_iterations=3, stop_reason="max_iterations")

    data = json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["max_iterations"] == 3
    assert data["stop_reason"] == "max_iterations"
    assert data["latest_checkpoint"] == "checkpoints/checkpoint_3"
    assert data["evolution_trace"] == "evolution_trace.jsonl"
    assert data["resolved_config"]["llm"]["api_key"] == "<redacted>"
    assert data["resolved_config"]["llm"]["models"][0]["api_key"] == "<redacted>"
    assert "api_base" not in data["resolved_config"]["llm"]
    assert "api_base" not in data["resolved_config"]["llm"]["models"][0]
    assert data["resolved_config"]["log_dir"] == "<redacted>"
    assert "super-secret" not in json.dumps(data)
    assert "provider.example" not in json.dumps(data)


def test_manifest_rejects_unknown_stop_reasons(tmp_path):
    initial = tmp_path / "initial.py"
    evaluator = tmp_path / "evaluator.py"
    initial.write_text("pass", encoding="utf-8")
    evaluator.write_text("pass", encoding="utf-8")
    manifest = RunManifest(str(tmp_path), Config(), str(initial), str(evaluator))

    with pytest.raises(ValueError, match="Unsupported stop reason"):
        manifest.write(max_iterations=1, stop_reason="early_stopping")


@pytest.mark.asyncio
async def test_controller_writes_manifest_for_library_runs(tmp_path):
    initial = tmp_path / "initial.py"
    evaluator = tmp_path / "evaluator.py"
    initial.write_text("# EVOLVE-BLOCK-START\npass\n# EVOLVE-BLOCK-END\n", encoding="utf-8")
    evaluator.write_text("def evaluate(path): return {'combined_score': 0.0}\n", encoding="utf-8")
    config = OpenEvolveConfig(max_iterations=1)
    config.llm.models = [
        LLMModelConfig(
            name="test-model",
            provider="test-provider",
            api_key="super-secret",
            api_base="https://example.test",
        )
    ]
    config.evaluator.cascade_evaluation = False

    mock_evaluator = Mock()
    mock_evaluator.evaluate_program = AsyncMock(return_value={"combined_score": 0.0})
    mock_evaluator.get_pending_artifacts.return_value = []

    with (
        patch("openevolve.controller.Evaluator", return_value=mock_evaluator),
        patch("openevolve.controller.ProcessParallelController") as parallel_controller_class,
    ):
        parallel = Mock()
        parallel.start = Mock()
        parallel.stop = Mock()
        parallel.run_evolution = AsyncMock()
        parallel.shutdown_event.is_set.return_value = False
        parallel.early_stopping_triggered = False
        parallel_controller_class.return_value = parallel

        controller = OpenEvolve(str(initial), str(evaluator), config, str(tmp_path))
        await controller.run(iterations=1)

    data = json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert data["stop_reason"] == "max_iterations"
    assert data["resolved_config"]["llm"]["models"][0]["api_key"] == "<redacted>"
    assert "api_base" not in data["resolved_config"]["llm"]["models"][0]
