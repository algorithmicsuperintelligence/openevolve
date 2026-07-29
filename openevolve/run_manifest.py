"""Redacted, run-level provenance records for evolution runs."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from openevolve._version import __version__

SCHEMA_VERSION = 1
MANIFEST_FILENAME = "run_manifest.json"
_REDACTED = "<redacted>"
_SECRET_KEYS = {"api_key", "apikey", "access_token", "auth_token", "token", "password", "secret"}
_OMITTED_KEYS = {"api_base", "base_url", "api_url", "_manual_queue_dir"}
_PATH_KEYS = {"db_path", "artifacts_base_path", "log_dir", "template_dir", "output_path"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_absolute_path(value: Any) -> bool:
    if not isinstance(value, (str, Path)):
        return False
    text = str(value)
    return (
        Path(text).is_absolute() or text.startswith("/") or bool(re.match(r"^[A-Za-z]:[\\/]", text))
    )


def _sanitize_config(value: Any, key: Optional[str] = None) -> Any:
    """Make a config JSON-safe without exposing credentials, origins, or user paths."""
    normalized_key = key.lower() if key else None
    if normalized_key in _SECRET_KEYS:
        return _REDACTED if value is not None else None
    if normalized_key in _OMITTED_KEYS:
        return None
    if normalized_key in _PATH_KEYS and _is_absolute_path(value):
        return _REDACTED

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _sanitize_config(getattr(value, field.name), field.name)
            for field in dataclasses.fields(value)
            if field.name not in _OMITTED_KEYS
        }
    if isinstance(value, dict):
        return {
            str(item_key): _sanitize_config(item_value, str(item_key))
            for item_key, item_value in value.items()
            if str(item_key).lower() not in _OMITTED_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_config(item, key) for item in value]
    if isinstance(value, Path):
        return _REDACTED if value.is_absolute() else str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if callable(value):
        return f"<{type(value).__name__}>"
    return f"<{type(value).__name__}>"


class RunManifest:
    """Write a small, redacted provenance record outside checkpoint metadata."""

    def __init__(
        self, output_dir: str, config: Any, initial_program_path: str, evaluator_path: str
    ):
        self.output_dir = Path(output_dir)
        self.config = config
        self.initial_program_path = initial_program_path
        self.evaluator_path = evaluator_path
        self.run_id = str(uuid.uuid4())
        self.started_at = _utc_now()
        self.path = self.output_dir / MANIFEST_FILENAME

    def write(self, max_iterations: int, stop_reason: Optional[str] = None) -> None:
        if stop_reason not in {None, "completed", "max_iterations", "interrupted", "error"}:
            raise ValueError(f"Unsupported stop reason: {stop_reason}")

        manifest: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "openevolve_version": __version__,
            "started_at": self.started_at,
            "max_iterations": max_iterations,
            "random_seed": getattr(self.config, "random_seed", None),
            "resolved_config": _sanitize_config(self.config),
            "initial_program_sha256": _sha256_file(self.initial_program_path),
            "evaluator_sha256": _sha256_file(self.evaluator_path),
            "runtime": {
                "python_version": platform.python_version(),
                "platform": platform.system(),
                "platform_release": platform.release(),
                "architecture": platform.machine(),
                "implementation": platform.python_implementation(),
                "python_executable": Path(sys.executable).name,
            },
        }

        trace_path = getattr(getattr(self.config, "evolution_trace", None), "output_path", None)
        if trace_path:
            trace = Path(trace_path)
            if trace.exists() and trace.is_relative_to(self.output_dir):
                manifest["evolution_trace"] = trace.relative_to(self.output_dir).as_posix()
        else:
            trace = (
                self.output_dir
                / f"evolution_trace.{getattr(self.config.evolution_trace, 'format', 'jsonl')}"
            )
            if trace.exists():
                manifest["evolution_trace"] = trace.name

        checkpoints = self.output_dir / "checkpoints"
        if checkpoints.exists():
            checkpoint_candidates = [path for path in checkpoints.iterdir() if path.is_dir()]
            if checkpoint_candidates:
                latest_checkpoint = max(
                    checkpoint_candidates, key=lambda path: path.stat().st_mtime
                )
                manifest["latest_checkpoint"] = latest_checkpoint.relative_to(
                    self.output_dir
                ).as_posix()

        if stop_reason is not None:
            manifest["ended_at"] = _utc_now()
            manifest["stop_reason"] = stop_reason

        self.output_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(".json.tmp")
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_path, self.path)
