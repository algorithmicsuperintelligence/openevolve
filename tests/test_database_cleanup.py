# tests/test_database_cleanup.py

import os
import shutil
import tempfile
import time
import unittest

from openevolve.config import DatabaseConfig
from openevolve.database import Program
from openevolve.database_memory import InMemoryProgramDatabase as ProgramDatabase


class TestArtifactCleanup(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "db")
        os.makedirs(self.db_path, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_artifact_cleanup(self):
        # 1. Configure the database for cleanup
        config = DatabaseConfig(
            db_path=self.db_path, cleanup_old_artifacts=True, artifact_retention_days=1
        )
        db = ProgramDatabase(config)

        # 2. Create dummy artifact directories
        artifacts_path = os.path.join(self.db_path, "artifacts")
        os.makedirs(artifacts_path, exist_ok=True)

        dir_to_keep = os.path.join(artifacts_path, "artifact_to_keep")
        dir_to_delete = os.path.join(artifacts_path, "artifact_to_delete")

        os.makedirs(dir_to_keep, exist_ok=True)
        os.makedirs(dir_to_delete, exist_ok=True)

        # 3. Set the modification time of one directory to be old
        old_time = time.time() - (2 * 24 * 60 * 60)  # 2 days ago
        os.utime(dir_to_delete, (old_time, old_time))

        self.assertTrue(os.path.exists(dir_to_keep))
        self.assertTrue(os.path.exists(dir_to_delete))

        # 4. Storing evaluation evidence triggers retention cleanup
        db.add(Program(id="new", code="pass"))
        db.store_artifacts("new", {"stdout": "new evidence"})

        # 5. Assert that the old directory was deleted and the new one was kept
        self.assertTrue(
            os.path.exists(dir_to_keep), "New artifact directory should not be deleted."
        )
        self.assertFalse(
            os.path.exists(dir_to_delete), "Old artifact directory should have been deleted."
        )


if __name__ == "__main__":
    unittest.main()
