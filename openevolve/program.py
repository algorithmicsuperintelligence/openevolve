"""Program value shared by all program database implementations."""

import logging
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Program:
    """Represents a program in the database"""

    # Program identification
    id: str
    code: str
    changes_description: str = (
        ""  # compact program changes description (via LLM) stored per program
    )
    language: str = "python"

    # Evolution information
    parent_id: Optional[str] = None
    generation: int = 0
    timestamp: float = field(default_factory=time.time)
    iteration_found: int = 0  # Track which iteration this program was found

    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    # Derived features
    complexity: float = 0.0
    diversity: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Prompts
    prompts: Optional[Dict[str, Any]] = None

    # Artifact storage
    artifacts_json: Optional[str] = None  # JSON-serialized small artifacts
    artifact_dir: Optional[str] = None  # Path to large artifact files

    # Embedding vector for novelty rejection sampling
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Program":
        """Create from dictionary representation"""
        # old DBs don't have changes_description (backward-compatibility)
        if "changes_description" not in data:
            metadata = data.get("metadata") or {}
            if isinstance(metadata, dict):
                data = {
                    **data,
                    "changes_description": metadata.get("changes_description")
                    or metadata.get("changes")
                    or "empty",
                }
            else:
                data = {**data, "changes_description": "empty"}

        # Get the valid field names for the Program dataclass
        valid_fields = {f.name for f in fields(cls)}

        # Filter the data to only include valid fields
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        # Log if we're filtering out any fields
        if len(filtered_data) != len(data):
            filtered_out = set(data.keys()) - set(filtered_data.keys())
            logger.debug(f"Filtered out unsupported fields when loading Program: {filtered_out}")

        return cls(**filtered_data)
