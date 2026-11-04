"""Population checkpoints expose committed state rather than directory leftovers."""

import builtins
import dataclasses
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openevolve.config import DatabaseConfig
from openevolve.database import Program, ProgramDatabase


class _CheckpointTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "checkpoint"
        self.config = DatabaseConfig(num_islands=2, feature_dimensions=["complexity"])
        self.database = ProgramDatabase(self.config)

    def add(self, identifier="seed", code="committed", island=0, score=1):
        program = Program(id=identifier, code=code, metrics={"combined_score": score})
        self.database.add(program, target_island=island)
        return program

    def restored(self):
        database = ProgramDatabase(self.config)
        database.load(str(self.path))
        return database


class TestCheckpointTransactions(_CheckpointTestCase):
    def test_failed_metadata_publication_preserves_committed_program(self):
        program = self.add()
        self.database.save(str(self.path), iteration=1)
        before = (self.path / "metadata.json").read_bytes()
        program.code = "uncommitted"
        original = builtins.open

        def fail_metadata(path, mode="r", *args, **kwargs):
            if Path(path) == self.path / "metadata.json" and mode == "w":
                raise OSError("injected metadata failure")
            return original(path, mode, *args, **kwargs)

        original_replace = os.replace

        def fail_replace(source, destination):
            if Path(destination) == self.path / "metadata.json":
                raise OSError("injected metadata failure")
            return original_replace(source, destination)

        with patch("builtins.open", fail_metadata), patch("os.replace", fail_replace):
            with self.assertRaisesRegex(OSError, "injected metadata"):
                self.database.save(str(self.path), iteration=2)

        self.assertEqual((self.path / "metadata.json").read_bytes(), before)
        loaded = self.restored()
        self.assertEqual(loaded.get("seed").code, "committed")
        self.assertEqual(loaded.last_iteration, 1)

    def test_failed_program_serialization_does_not_truncate_committed_file(self):
        program = self.add()
        self.database.save(str(self.path))
        program.metadata["not_serializable"] = object()
        with self.assertRaises(TypeError):
            self.database.save(str(self.path))
        self.assertEqual(self.restored().get("seed").code, "committed")

    def test_saving_empty_population_does_not_resurrect_removed_programs(self):
        self.add()
        self.database.save(str(self.path))
        empty = ProgramDatabase(self.config)
        empty.save(str(self.path), iteration=2)
        loaded = self.restored()
        self.assertEqual(loaded.programs, {})
        self.assertEqual(loaded.islands, [set(), set()])
        self.assertEqual(loaded.archive, set())
        self.assertIsNone(loaded.best_program_id)

    def test_load_replaces_population_instead_of_merging_runs(self):
        self.add("saved")
        self.database.save(str(self.path))
        live = ProgramDatabase(self.config)
        live.add(Program(id="unrelated", code="old", metrics={"combined_score": 9}))
        live.load(str(self.path))
        self.assertEqual(set(live.programs), {"saved"})
        self.assertEqual(set.union(*live.islands), {"saved"})
        self.assertNotIn("unrelated", live.archive)

    def test_failed_load_preserves_live_population_and_metadata(self):
        self.add()
        self.database.save(str(self.path), iteration=3)
        live = ProgramDatabase(self.config)
        live.add(Program(id="live", code="live", metrics={"combined_score": 2}))
        old_programs, old_archive, old_islands = live.programs, live.archive, live.islands
        for path in (self.path / "programs").rglob("*.json"):
            path.write_text("{broken", encoding="utf-8")
        with self.assertRaises((ValueError, OSError)):
            live.load(str(self.path))
        self.assertIs(live.programs, old_programs)
        self.assertIs(live.archive, old_archive)
        self.assertIs(live.islands, old_islands)
        self.assertEqual(live.last_iteration, 0)

    def test_prompt_logs_survive_constructor_load_and_second_save(self):
        self.add()
        prompt = {"system": "system instructions", "user": "test request"}
        self.database.log_prompt("seed", "template", prompt)
        self.database.save(str(self.path))
        config = dataclasses.replace(self.config, db_path=str(self.path))
        loaded = ProgramDatabase(config)
        self.assertEqual(loaded.prompts_by_program, {"seed": {"template": prompt}})
        loaded.save(str(self.root / "second"))
        second = ProgramDatabase(self.config)
        second.load(str(self.root / "second"))
        self.assertEqual(second.prompts_by_program, {"seed": {"template": prompt}})

    def test_checkpoint_carries_large_artifacts_after_relocation(self):
        self.add()
        source = self.root / "original_artifacts"
        source.mkdir()
        (source / "report.txt").write_text("large report", encoding="utf-8")
        (source / "weights.bin").write_bytes(b"\x00\xffweights")
        self.database.get("seed").artifact_dir = str(source)
        self.database.save(str(self.path))
        relocated = self.root / "relocated"
        shutil.move(self.path, relocated)
        shutil.rmtree(source)
        loaded = ProgramDatabase(self.config)
        loaded.load(str(relocated))
        self.assertEqual(
            loaded.get_artifacts("seed"),
            {"report.txt": "large report", "weights.bin": b"\x00\xffweights"},
        )

    def test_incremental_durable_add_commits_island_metadata(self):
        config = dataclasses.replace(self.config, db_path=str(self.path), in_memory=False)
        database = ProgramDatabase(config)
        database.add(
            Program(id="seed", code="code", metrics={"combined_score": 1}), target_island=1
        )
        loaded = ProgramDatabase(config)
        self.assertEqual(loaded.islands, [set(), {"seed"}])
        self.assertEqual(loaded.island_best_programs, [None, "seed"])
        self.assertEqual(loaded.best_program_id, "seed")


class TestCheckpointPublication(_CheckpointTestCase):
    """Faults, interruption and overlapping readers at the filesystem boundary."""

    def test_program_write_failure_keeps_previous_manifest(self):
        program = self.add()
        self.database.save(str(self.path), iteration=4)
        old_metadata = (self.path / "metadata.json").read_bytes()
        program.code = "new program"
        original = os.replace

        def fail(source, destination):
            if Path(destination).suffix == ".json" and Path(destination).name != "metadata.json":
                raise OSError("injected object failure")
            return original(source, destination)

        with patch("os.replace", fail):
            with self.assertRaisesRegex(OSError, "object failure"):
                self.database.save(str(self.path), iteration=5)
        self.assertEqual((self.path / "metadata.json").read_bytes(), old_metadata)
        self.assertEqual(self.restored().get("seed").code, "committed")
        self.assertEqual(list(self.path.glob("programs/*.tmp")), [])

    def test_failed_first_save_has_no_visible_checkpoint(self):
        self.add()
        original = os.replace

        def fail(source, destination):
            if Path(destination) == self.path / "metadata.json":
                raise OSError("injected first save failure")
            return original(source, destination)

        with patch("os.replace", fail):
            with self.assertRaisesRegex(OSError, "first save failure"):
                self.database.save(str(self.path))
        self.assertFalse((self.path / "metadata.json").exists())
        # Unpublished objects must not be interpreted as a legacy population.
        self.assertEqual(self.restored().programs, {})

    def test_manifest_serialization_failure_preserves_previous_population(self):
        program = self.add()
        self.database.save(str(self.path))
        self.database.feature_stats["invalid"] = {"min": object()}
        program.code = "new program"
        with self.assertRaises(TypeError):
            self.database.save(str(self.path))
        self.assertEqual(self.restored().get("seed").code, "committed")

    def test_unreferenced_files_are_not_population_members(self):
        self.add()
        self.database.save(str(self.path))
        (self.path / "programs" / "orphan.json").write_text(
            json.dumps(
                {"id": "orphan", "code": "not committed", "metrics": {"combined_score": 99}}
            ),
            encoding="utf-8",
        )
        self.assertEqual(set(self.restored().programs), {"seed"})

    def test_reader_keeps_its_committed_generation_during_replacement(self):
        import concurrent.futures
        import threading

        program = self.add()
        self.database.save(str(self.path), iteration=4)
        manifest = json.loads((self.path / "metadata.json").read_text())
        old_file = self.path / "programs" / ".objects" / manifest["program_files"]["seed"]
        entered, release = threading.Event(), threading.Event()
        original = Path.read_bytes

        def read(path):
            if path == old_file:
                entered.set()
                self.assertTrue(release.wait(5))
            return original(path)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            with patch.object(Path, "read_bytes", read):
                pending = executor.submit(self.restored)
                try:
                    self.assertTrue(entered.wait(5))
                    program.code = "next generation"
                    self.database.save(str(self.path), iteration=5)
                finally:
                    release.set()
                previous = pending.result(5)
        self.assertEqual(previous.get("seed").code, "committed")
        self.assertEqual(previous.last_iteration, 4)
        current = self.restored()
        self.assertEqual(current.get("seed").code, "next generation")
        self.assertEqual(current.last_iteration, 5)

    def test_process_interruption_exposes_only_complete_population(self):
        import subprocess
        import sys

        program = self.add()
        script = r"""
import os
import sys
from pathlib import Path
from openevolve.config import DatabaseConfig
from openevolve.database import ProgramDatabase
root, phase = Path(sys.argv[1]), sys.argv[2]
database = ProgramDatabase(DatabaseConfig(num_islands=2, feature_dimensions=['complexity']))
database.load(str(root/'checkpoint'))
program = database.get('seed')
program.code = 'interrupted'
source = root/'artifacts'
(source/'report.txt').write_text('new artifact', encoding='utf-8')
program.artifact_dir = str(source)
replace, rename = os.replace, os.rename
def replace_and_exit(source, destination):
    replace(source, destination)
    path = Path(destination)
    if (phase == 'program' and path.parent.name == '.objects') or (phase == 'manifest' and path.name == 'metadata.json'):
        os._exit(17)
def rename_and_exit(source, destination):
    rename(source, destination)
    if phase == 'artifact' and Path(destination).parent.name == '.checkpoint_artifacts':
        os._exit(17)
os.replace, os.rename = replace_and_exit, rename_and_exit
database.save(str(root/'checkpoint'), iteration=9)
"""
        for phase in ("artifact", "program", "manifest"):
            with self.subTest(phase=phase):
                case_root = self.root / phase
                source = case_root / "artifacts"
                source.mkdir(parents=True)
                (source / "report.txt").write_text("committed artifact", encoding="utf-8")
                program.artifact_dir = str(source)
                self.path = case_root / "checkpoint"
                self.database.save(str(self.path), iteration=2)
                completed = subprocess.run(
                    [sys.executable, "-c", script, str(case_root), phase],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(completed.returncode, 17, completed.stderr)
                loaded = self.restored()
                if phase == "manifest":
                    self.assertEqual(loaded.get("seed").code, "interrupted")
                    self.assertEqual(loaded.last_iteration, 9)
                    self.assertEqual(loaded.get_artifacts("seed"), {"report.txt": "new artifact"})
                else:
                    self.assertEqual(loaded.get("seed").code, "committed")
                    self.assertEqual(loaded.last_iteration, 2)

    def test_mutable_artifact_updates_do_not_change_previous_checkpoint(self):
        config = dataclasses.replace(
            self.config, artifacts_base_path=str(self.root / "working"), artifact_size_threshold=1
        )
        database = ProgramDatabase(config)
        database.add(Program(id="seed", code="code", metrics={"combined_score": 1}))
        database.store_artifacts("seed", {"kept.txt": "keep", "updated.txt": "original"})
        database.save(str(self.path))
        restored = ProgramDatabase(config)
        restored.load(str(self.path))
        restored.store_artifacts("seed", {"updated.txt": "updated"})
        self.assertEqual(
            restored.get_artifacts("seed"), {"kept.txt": "keep", "updated.txt": "updated"}
        )
        self.assertEqual(
            self.restored().get_artifacts("seed"), {"kept.txt": "keep", "updated.txt": "original"}
        )

    def test_resumed_artifact_update_ignores_uncommitted_working_files(self):
        config = dataclasses.replace(
            self.config, artifacts_base_path=str(self.root / "working"), artifact_size_threshold=1
        )
        database = ProgramDatabase(config)
        database.add(Program(id="seed", code="code", metrics={"combined_score": 1}))
        database.store_artifacts("seed", {"report.txt": "committed report"})
        database.save(str(self.path))
        working = Path(database.get("seed").artifact_dir)
        (working / "uncommitted.txt").write_text("not part of checkpoint", encoding="utf-8")
        restored = ProgramDatabase(config)
        restored.load(str(self.path))
        restored.store_artifacts("seed", {"new.txt": "new report"})
        self.assertEqual(
            restored.get_artifacts("seed"),
            {"report.txt": "committed report", "new.txt": "new report"},
        )
        restored.store_artifacts("seed", {"third.txt": "third report"})
        self.assertEqual(
            set(restored.get_artifacts("seed")), {"report.txt", "new.txt", "third.txt"}
        )

    def test_missing_artifact_source_fails_without_publishing(self):
        program = self.add()
        self.database.save(str(self.path))
        program.artifact_dir = str(self.root / "does-not-exist")
        with self.assertRaisesRegex(FileNotFoundError, "artifact directory"):
            self.database.save(str(self.path))
        self.assertIsNone(self.restored().get("seed").artifact_dir)

    def test_retention_listing_failure_does_not_fail_committed_save(self):
        self.config.cleanup_old_artifacts = True
        program = self.add()
        self.database.save(str(self.path), iteration=1)
        artifacts = self.path / "artifacts"
        artifacts.mkdir()
        program.code = "new committed program"
        original = os.listdir

        def fail(directory):
            if Path(directory) == artifacts:
                raise PermissionError("retention directory cannot be listed")
            return original(directory)

        with (
            patch("os.listdir", fail),
            self.assertLogs("openevolve.database", level="WARNING") as logs,
        ):
            self.database.save(str(self.path), iteration=2)
        loaded = self.restored()
        self.assertEqual(loaded.get("seed").code, "new committed program")
        self.assertEqual(loaded.last_iteration, 2)
        self.assertTrue(
            any("retention directory cannot be listed" in message for message in logs.output)
        )

    def test_retention_listing_failure_keeps_strategy_add_committed(self):
        from openevolve.population import PopulationStrategy

        config = dataclasses.replace(
            self.config, db_path=str(self.path), cleanup_old_artifacts=True
        )
        database = ProgramDatabase(config, PopulationStrategy(replace_cell=lambda *args: True))
        database.add(
            Program(id="seed", code="seed", metrics={"combined_score": 1}), target_island=0
        )
        artifacts = self.path / "artifacts"
        artifacts.mkdir()
        original = os.listdir

        def fail(directory):
            if Path(directory) == artifacts:
                raise PermissionError("retention directory cannot be listed")
            return original(directory)

        with (
            patch("os.listdir", fail),
            self.assertLogs("openevolve.database", level="WARNING") as logs,
        ):
            result = database.add(
                Program(id="candidate", code="candidate", metrics={"combined_score": 2}),
                target_island=1,
            )
        loaded = ProgramDatabase(config)
        self.assertEqual(result, "candidate")
        self.assertEqual(set(database.programs), {"candidate", "seed"})
        self.assertEqual(set(database.programs), set(loaded.programs))
        self.assertEqual(database.islands, loaded.islands)
        self.assertEqual(database.archive, loaded.archive)
        self.assertEqual(database.best_program_id, loaded.best_program_id)
        self.assertTrue(
            any("retention directory cannot be listed" in message for message in logs.output)
        )

    def test_retention_cannot_remove_artifacts_needed_by_checkpoint(self):
        import time

        config = dataclasses.replace(
            self.config,
            db_path=str(self.path),
            cleanup_old_artifacts=True,
            artifact_retention_days=1,
        )
        database = ProgramDatabase(config)
        program = Program(id="seed", code="code", metrics={"combined_score": 1})
        database.add(program)
        source = self.path / "artifacts" / "seed"
        source.mkdir(parents=True)
        (source / "report.txt").write_text("retained in checkpoint", encoding="utf-8")
        old = time.time() - 3 * 86400
        os.utime(source, (old, old))
        program.artifact_dir = str(source)
        database.save()
        self.assertTrue(source.exists())
        database.save()
        self.assertEqual(
            self.restored().get_artifacts("seed"), {"report.txt": "retained in checkpoint"}
        )


class TestCheckpointRecovery(_CheckpointTestCase):
    def test_legacy_checkpoint_upgrades_without_losing_programs(self):
        directory = self.path / "programs"
        directory.mkdir(parents=True)
        for index in range(2):
            (directory / f"legacy-{index}.json").write_text(
                json.dumps(
                    {
                        "id": f"legacy-{index}",
                        "code": f"code {index}",
                        "metrics": {"combined_score": index + 1},
                        "prompts": {"old": {"user": "request"}},
                    }
                ),
                encoding="utf-8",
            )
        (self.path / "metadata.json").write_text(
            json.dumps(
                {
                    "islands": [["legacy-0"], ["legacy-1"]],
                    "archive": ["legacy-1"],
                    "best_program_id": "legacy-1",
                    "last_iteration": 7,
                }
            ),
            encoding="utf-8",
        )
        loaded = self.restored()
        self.assertEqual(loaded.islands, [{"legacy-0"}, {"legacy-1"}])
        loaded.save(str(self.path))
        resumed = self.restored()
        self.assertEqual(set(resumed.programs), {"legacy-0", "legacy-1"})
        self.assertEqual(resumed.get_best_program().id, "legacy-1")
        self.assertEqual(resumed.last_iteration, 7)
        self.assertEqual(resumed.prompts_by_program["legacy-0"], {"old": {"user": "request"}})

    def test_legacy_program_only_directory_is_still_readable(self):
        directory = self.path / "programs"
        directory.mkdir(parents=True)
        (directory / "old.json").write_text(
            json.dumps({"id": "old", "code": "legacy code"}), encoding="utf-8"
        )
        (directory / "damaged.json").write_text("{damaged", encoding="utf-8")
        loaded = self.restored()
        self.assertEqual(set(loaded.programs), {"old"})
        self.assertEqual(loaded.get("old").code, "legacy code")
        self.assertEqual(loaded.islands[0], {"old"})

    def test_corrupt_manifest_preserves_active_database(self):
        self.add()
        self.database.save(str(self.path))
        old_programs = self.database.programs
        (self.path / "metadata.json").write_text("{broken", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.database.load(str(self.path))
        self.assertIs(self.database.programs, old_programs)
        self.assertEqual(self.database.get("seed").code, "committed")

    def test_invalid_manifest_object_references_are_rejected(self):
        self.add()
        self.database.save(str(self.path))
        original = json.loads((self.path / "metadata.json").read_text())
        for filename in ("../outside.json", "/tmp/outside.json", "not-a-digest.json", 42):
            with self.subTest(filename=filename):
                metadata = {**original, "program_files": {"seed": filename}}
                (self.path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Invalid checkpoint object"):
                    self.database.load(str(self.path))
                self.assertEqual(self.database.get("seed").code, "committed")

    def test_future_manifest_version_is_not_loaded_as_legacy(self):
        self.add()
        self.database.save(str(self.path))
        metadata = json.loads((self.path / "metadata.json").read_text())
        metadata["checkpoint_version"] = 100
        (self.path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unsupported checkpoint version"):
            self.database.load(str(self.path))
        self.assertEqual(self.database.get("seed").code, "committed")

    def test_changed_program_identity_is_not_accepted(self):
        import hashlib

        self.add()
        self.database.save(str(self.path))
        metadata = json.loads((self.path / "metadata.json").read_text())
        content = json.dumps({"id": "other", "code": "different"}).encode()
        filename = hashlib.sha256(content).hexdigest() + ".json"
        (self.path / "programs" / ".objects" / filename).write_bytes(content)
        metadata["program_files"]["seed"] = filename
        (self.path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.database.load(str(self.path))
        self.assertEqual(set(self.database.programs), {"seed"})

    def test_damaged_artifacts_fail_load_before_replacing_population(self):
        program = self.add()
        source = self.root / "source"
        source.mkdir()
        (source / "report.txt").write_text("report", encoding="utf-8")
        program.artifact_dir = str(source)
        self.database.save(str(self.path))
        artifact = next((self.path / ".checkpoint_artifacts").glob("*/report.txt"))
        artifact.write_text("damaged", encoding="utf-8")
        old_programs = self.database.programs
        with self.assertRaisesRegex(ValueError, "artifact checksum mismatch"):
            self.database.load(str(self.path))
        self.assertIs(self.database.programs, old_programs)

    def test_save_rejects_corrupted_existing_objects(self):
        self.add()
        self.database.save(str(self.path))
        original = (self.path / "metadata.json").read_bytes()
        program_file = next((self.path / "programs" / ".objects").glob("*.json"))
        program_file.write_text("corrupted", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "program object is corrupt"):
            self.database.save(str(self.path))
        self.assertEqual((self.path / "metadata.json").read_bytes(), original)

    def test_repeated_load_resets_sampling_caches(self):
        self.add()
        self.database.save(str(self.path))
        self.database.diversity_cache = {123: {"value": 99, "timestamp": 0}}
        self.database.diversity_reference_set = ["unrelated code"]
        self.database._code_shape_cache["stale code"] = (1, 1, frozenset())
        for _ in range(2):
            self.database.load(str(self.path))
            self.assertEqual(self.database.diversity_cache, {})
            self.assertEqual(self.database.diversity_reference_set, [])
            self.assertNotIn("stale code", self.database._code_shape_cache)
            parent, _inspirations = self.database.sample_from_island(0)
            self.assertEqual(parent.id, "seed")

    def test_incremental_add_reuses_unchanged_program_objects(self):
        config = dataclasses.replace(self.config, population_size=200, archive_size=200)
        database = ProgramDatabase(config)
        for index in range(100):
            program = Program(
                id=f"program-{index}", code=f"code {index}", metrics={"combined_score": index}
            )
            database.programs[program.id] = program
            database.islands[index % 2].add(program.id)
        database.save(str(self.path))
        database.config.db_path = str(self.path)
        before = len(list((self.path / "programs" / ".objects").glob("*.json")))
        original = Program.to_dict
        serialized = []

        def capture(program):
            serialized.append(program.id)
            return original(program)

        with patch.object(Program, "to_dict", capture):
            database.add(Program(id="new", code="new program", metrics={"combined_score": 1000}))
        self.assertEqual(serialized, ["new"])
        self.assertEqual(
            len(list((self.path / "programs" / ".objects").glob("*.json"))), before + 1
        )
        loaded = self.restored()
        self.assertEqual(set(loaded.programs), set(database.programs))
        self.assertEqual(loaded.islands, database.islands)
        self.assertEqual(loaded.best_program_id, "new")

    def test_python_string_and_json_key_inputs_keep_existing_support(self):
        program = self.add(code="π = '𝄞'\n# \ud800")
        program.metadata = {1: "numeric", "text": "string"}
        self.database.save(str(self.path))
        loaded = self.restored().get("seed")
        self.assertEqual(loaded.code, program.code)
        self.assertEqual(loaded.metadata["1"], "numeric")
        self.assertEqual(loaded.metadata["text"], "string")


class TestCheckpointTraceConsumers(_CheckpointTestCase):
    def test_trace_readers_use_committed_records(self):
        from openevolve.evolution_trace import (
            extract_evolution_trace_from_checkpoint,
            extract_full_lineage_traces,
        )

        parent = self.add("parent", "parent code", island=0)
        child = Program(
            id="child",
            code="old child code",
            parent_id=parent.id,
            metrics={"combined_score": 2},
            iteration_found=3,
        )
        self.database.add(child, target_island=1)
        self.database.log_prompt(
            "child", "diff", {"system": "system", "user": "request"}, responses=["response"]
        )
        self.database.save(str(self.path))
        child.code = "committed child code"
        self.database.save(str(self.path))
        traces = extract_evolution_trace_from_checkpoint(self.path)
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0].parent_code, "parent code")
        self.assertEqual(traces[0].child_code, "committed child code")
        lineages = extract_full_lineage_traces(self.path)
        self.assertEqual(len(lineages), 1)
        self.assertEqual(lineages[0]["final_program_id"], "child")
        self.assertEqual(lineages[0]["improvement_steps"][0]["action"]["llm_response"], "response")

    def test_trace_readers_accept_programs_without_prompt_logs(self):
        from openevolve.evolution_trace import (
            extract_evolution_trace_from_checkpoint,
            extract_full_lineage_traces,
        )

        parent = self.add("parent", island=0)
        self.database.add(
            Program(
                id="child", code="child code", parent_id=parent.id, metrics={"combined_score": 2}
            ),
            target_island=1,
        )
        self.database.save(str(self.path))
        traces = extract_evolution_trace_from_checkpoint(self.path)
        self.assertEqual(len(traces), 1)
        self.assertIsNone(traces[0].prompt)
        lineages = extract_full_lineage_traces(self.path)
        self.assertEqual(len(lineages), 1)
        self.assertIsNone(lineages[0]["improvement_steps"][0]["action"])


if __name__ == "__main__":
    unittest.main()
