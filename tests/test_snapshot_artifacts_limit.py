"""
Tests for configurable max_snapshot_artifacts limit.
Controls how many artifacts are included in worker process snapshots.
"""

import unittest

from openevolve.config import Config, DatabaseConfig
from openevolve.database import ProgramDatabase, Program


class TestMaxSnapshotArtifactsConfig(unittest.TestCase):
    """Tests for max_snapshot_artifacts configuration"""

    def test_default_value_is_100(self):
        """Test that max_snapshot_artifacts defaults to 100"""
        config = Config()
        self.assertEqual(config.database.max_snapshot_artifacts, 100)

    def test_database_config_default(self):
        """Test DatabaseConfig default for max_snapshot_artifacts"""
        db_config = DatabaseConfig()
        self.assertEqual(db_config.max_snapshot_artifacts, 100)

    def test_custom_value_from_dict(self):
        """Test loading custom max_snapshot_artifacts from config dict"""
        config_dict = {
            "llm": {"primary_model": "gpt-4"},
            "database": {
                "max_snapshot_artifacts": 500,
            }
        }
        config = Config.from_dict(config_dict)
        self.assertEqual(config.database.max_snapshot_artifacts, 500)

    def test_unlimited_artifacts_with_none(self):
        """Test setting unlimited artifacts with None"""
        config_dict = {
            "llm": {"primary_model": "gpt-4"},
            "database": {
                "max_snapshot_artifacts": None,
            }
        }
        config = Config.from_dict(config_dict)
        self.assertIsNone(config.database.max_snapshot_artifacts)

    def test_zero_artifacts(self):
        """Test setting max_snapshot_artifacts to 0"""
        config_dict = {
            "llm": {"primary_model": "gpt-4"},
            "database": {
                "max_snapshot_artifacts": 0,
            }
        }
        config = Config.from_dict(config_dict)
        self.assertEqual(config.database.max_snapshot_artifacts, 0)


class TestArtifactStorageWithLimit(unittest.TestCase):
    """Tests for artifact storage respecting the limit"""

    def test_store_artifacts_within_limit(self):
        """Test storing artifacts when within the limit"""
        db_config = DatabaseConfig(max_snapshot_artifacts=5)
        db = ProgramDatabase(db_config)

        # Add programs with artifacts
        for i in range(3):
            program = Program(
                id=f"prog_{i}",
                code=f"def func_{i}(): pass",
                generation=0,
                metrics={"score": i * 0.1},
            )
            db.add(program)
            db.store_artifacts(f"prog_{i}", {"output": f"result_{i}"})

        # All artifacts should be retrievable
        for i in range(3):
            artifacts = db.get_artifacts(f"prog_{i}")
            self.assertEqual(artifacts.get("output"), f"result_{i}")

    def test_store_many_artifacts(self):
        """Test storing more artifacts than the limit"""
        db_config = DatabaseConfig(max_snapshot_artifacts=5)
        db = ProgramDatabase(db_config)

        # Add 10 programs with artifacts
        for i in range(10):
            program = Program(
                id=f"prog_{i}",
                code=f"def func_{i}(): pass",
                generation=0,
                metrics={"score": i * 0.1},
            )
            db.add(program)
            db.store_artifacts(f"prog_{i}", {"output": f"result_{i}"})

        # All artifacts should still be stored in the database
        # (the limit only affects snapshots, not storage)
        for i in range(10):
            artifacts = db.get_artifacts(f"prog_{i}")
            self.assertEqual(artifacts.get("output"), f"result_{i}")

    def test_artifacts_for_nonexistent_program_returns_empty(self):
        """Test retrieving artifacts for non-existent program"""
        db_config = DatabaseConfig()
        db = ProgramDatabase(db_config)

        artifacts = db.get_artifacts("nonexistent_id")
        self.assertEqual(artifacts, {})

    def test_store_artifacts_for_nonexistent_program_logs_warning(self):
        """Test that storing artifacts for non-existent program doesn't crash"""
        db_config = DatabaseConfig()
        db = ProgramDatabase(db_config)

        # Should not raise an error
        db.store_artifacts("nonexistent", {"output": "test"})


class TestSnapshotCreation(unittest.TestCase):
    """Tests for snapshot creation with artifact limits"""

    def test_config_accessible_from_database(self):
        """Test that max_snapshot_artifacts is accessible from database config"""
        db_config = DatabaseConfig(max_snapshot_artifacts=50)
        db = ProgramDatabase(db_config)

        self.assertEqual(db.config.max_snapshot_artifacts, 50)

    def test_unlimited_config_is_none(self):
        """Test that unlimited artifacts config is None"""
        db_config = DatabaseConfig(max_snapshot_artifacts=None)
        db = ProgramDatabase(db_config)

        self.assertIsNone(db.config.max_snapshot_artifacts)


if __name__ == "__main__":
    unittest.main()
