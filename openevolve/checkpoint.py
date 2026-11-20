"""Immutable checkpoint objects published through one atomic population manifest."""

import errno
import hashlib
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)
_VERSION = 1
_ARTIFACTS = ".checkpoint_artifacts"


@dataclass
class CheckpointData:
    """Metadata and exact program records selected by a single manifest read."""

    metadata: Dict[str, Any]
    programs: List[Dict[str, Any]]


def _atomic_write(path: Path, content: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _artifact_contents(directory: Path) -> Tuple[str, List[Tuple[str, bytes]]]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Checkpoint artifact directory is missing: {directory}")
    contents = []
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(directory).as_posix()
        content = path.read_bytes()
        # Length prefixes distinguish file/name boundaries without depending
        # on any delimiter being absent from the artifact bytes.
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        contents.append((name, content))
    return digest.hexdigest(), contents


def _store_artifacts(root: Path, directory: str) -> str:
    digest, contents = _artifact_contents(Path(directory))
    base = root / _ARTIFACTS
    base.mkdir(exist_ok=True)
    destination = base / digest
    if destination.exists():
        existing_digest, _contents = _artifact_contents(destination)
        if existing_digest != digest:
            raise ValueError(f"Checkpoint artifact object is corrupt: {destination}")
    else:
        temporary = Path(tempfile.mkdtemp(prefix=".staging-", dir=base))
        try:
            for name, content in contents:
                path = temporary / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            try:
                os.rename(temporary, destination)
            except OSError as error:
                if error.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not destination.is_dir():
                    raise
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return destination.relative_to(root).as_posix()


def save_checkpoint(
    path: str,
    metadata: Dict[str, Any],
    programs: List[Dict[str, Any]],
    previous_files: Optional[Dict[str, str]] = None,
    program_ids: Optional[Set[str]] = None,
) -> Dict[str, str]:
    """Publish a complete population after all of its immutable objects exist.

    Previous objects remain available for readers that already captured an
    earlier manifest. A failure before the final replace cannot change that
    manifest or any of the objects it references.
    """
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    directory = root / "programs" / ".objects"
    directory.mkdir(parents=True, exist_ok=True)
    program_files = {
        identifier: filename
        for identifier, filename in (previous_files or {}).items()
        if program_ids is not None and identifier in program_ids
    }
    for program in programs:
        # The caller owns its record. Artifact relocation changes only the
        # serialized copy, never the live Program's working directory.
        record = dict(program)
        if record.get("artifact_dir"):
            record["artifact_dir"] = _store_artifacts(root, record["artifact_dir"])
        content = json.dumps(record).encode("utf-8")
        filename = hashlib.sha256(content).hexdigest() + ".json"
        target = directory / filename
        if target.exists():
            if target.read_bytes() != content:
                raise ValueError(f"Checkpoint program object is corrupt: {record['id']}")
        else:
            _atomic_write(target, content)
        program_files[record["id"]] = filename
    if program_ids is not None and set(program_files) != program_ids:
        raise ValueError("Checkpoint is missing program object references")
    manifest = {**metadata, "checkpoint_version": _VERSION, "program_files": program_files}
    content = json.dumps(manifest).encode("utf-8")
    _atomic_write(root / "metadata.json", content)
    return program_files


def _object_path(directory: Path, filename: str, suffix: str = "") -> Path:
    if (
        not isinstance(filename, str)
        or len(filename) != 64 + len(suffix)
        or (suffix and not filename.endswith(suffix))
        or any(character not in "0123456789abcdef" for character in filename[:64])
    ):
        raise ValueError(f"Invalid checkpoint object name: {filename!r}")
    return directory / filename


def load_checkpoint(path: str) -> CheckpointData:
    """Read one committed manifest and its objects, accepting legacy directories.

    Versioned checkpoints are complete snapshots: missing or corrupt referenced
    objects raise instead of silently deleting members of the population.
    Legacy directories retain their best-effort program loading behavior.
    """
    root = Path(path)
    metadata_file = root / "metadata.json"
    metadata = (
        json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.exists() else {}
    )
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata must be an object")
    directory = root / "programs"
    programs = []
    if "checkpoint_version" not in metadata:
        if directory.exists():
            for program_file in sorted(directory.glob("*.json")):
                try:
                    programs.append(json.loads(program_file.read_text(encoding="utf-8")))
                except (OSError, ValueError) as error:
                    logger.warning("Error loading program %s: %s", program_file.name, error)
        return CheckpointData(metadata, programs)
    if metadata["checkpoint_version"] != _VERSION:
        raise ValueError(f"Unsupported checkpoint version: {metadata['checkpoint_version']!r}")
    files = metadata.get("program_files")
    if not isinstance(files, dict):
        raise ValueError("Checkpoint program_files must be an object")
    directory = directory / ".objects"
    for identifier, filename in files.items():
        content = _object_path(directory, filename, ".json").read_bytes()
        if hashlib.sha256(content).hexdigest() != filename[:-5]:
            raise ValueError(f"Checkpoint program object checksum mismatch: {identifier}")
        record = json.loads(content)
        if not isinstance(record, dict) or record.get("id") != identifier:
            raise ValueError(f"Checkpoint program identity mismatch: {identifier}")
        if record.get("artifact_dir"):
            relative = Path(record["artifact_dir"])
            if len(relative.parts) != 2 or relative.parts[0] != _ARTIFACTS:
                raise ValueError(f"Invalid checkpoint artifact directory: {relative}")
            artifact_dir = _object_path(root / _ARTIFACTS, relative.parts[1])
            digest, _contents = _artifact_contents(artifact_dir)
            if digest != artifact_dir.name:
                raise ValueError(f"Checkpoint artifact checksum mismatch: {identifier}")
            record["artifact_dir"] = str(artifact_dir.resolve())
        programs.append(record)
    return CheckpointData(metadata, programs)
