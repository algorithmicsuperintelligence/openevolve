"""Workers receive bounded query results, with evidence for the actual parent."""

import pickle
import unittest
from unittest.mock import AsyncMock, Mock, patch

from openevolve import process_parallel as workers
from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.database_memory import InMemoryProgramDatabase
from openevolve.process_parallel import IterationContext, ProcessParallelController


class TestIterationContext(unittest.TestCase):
    def test_large_population_does_not_expand_worker_context(self):
        config = Config()
        config.database.num_islands = 110
        config.prompt.num_top_programs = 1
        config.prompt.num_diverse_programs = 0
        store = InMemoryProgramDatabase(config.database)
        for i in range(110):
            store.add(
                Program(id=f"p{i}", code=f"return {i}", metrics={"combined_score": i}),
                target_island=i,
            )
            store.store_artifacts(f"p{i}", {"stdout": f"evidence {i}"})
        database = Mock(spec_set=ProgramDatabase, wraps=store)
        controller = ProcessParallelController(config, "unused.py", database)
        # This parent fell outside the previous first-100-program artifact snapshot.
        context = controller._select_iteration_context(109)
        self.assertEqual(context.parent.id, "p109")
        self.assertEqual(context.parent_artifacts, {"stdout": "evidence 109"})
        self.assertEqual([p.id for p in context.top_programs], ["p109"])
        self.assertEqual(context.inspirations, [])
        database.get_artifacts.assert_called_once_with("p109")
        database.get_top_programs.assert_called_once_with(n=1, island_idx=109)
        wire_data = pickle.dumps(context)
        self.assertNotIn(b"evidence 108", wire_data)
        self.assertNotIn(b"return 108", wire_data)
        self.assertEqual(pickle.loads(wire_data), context)

    def test_each_candidate_queries_the_current_database(self):
        config = Config()
        config.database.num_islands = 1
        config.database.feature_dimensions = ["score"]
        config.prompt.num_top_programs = 1
        store = InMemoryProgramDatabase(config.database)
        database = Mock(spec_set=ProgramDatabase, wraps=store)
        database.add(Program(id="first", code="return 1", metrics={"combined_score": 0.5}))
        controller = ProcessParallelController(config, "unused.py", database)
        first_context = controller._select_iteration_context(0)
        database.add(Program(id="better", code="return 2", metrics={"combined_score": 0.9}))
        second_context = controller._select_iteration_context(0)
        self.assertEqual(first_context.top_programs[0].id, "first")
        self.assertEqual(second_context.top_programs[0].id, "better")
        self.assertEqual(database.sample_from_island.call_count, 2)
        self.assertEqual(database.get_top_programs.call_count, 2)

    def test_worker_uses_selected_context_and_explicit_target_island(self):
        config = Config()
        config.language = "python"
        config.diff_based_evolution = False
        config.prompt.num_top_programs = 1
        parent = Program(
            id="parent", code="return 1", metrics={"combined_score": 0.5}, metadata={"island": 0}
        )
        context = IterationContext(
            parent=parent,
            inspirations=[],
            top_programs=[parent],
            parent_artifacts={"stderr": "parent evidence"},
            target_island=1,
            feature_dimensions=("complexity",),
        )
        sampler = Mock()
        sampler.build_prompt.return_value = {"system": "system", "user": "user"}
        llm = Mock()
        llm.generate_with_context = AsyncMock(return_value="```python\ndef solve(): return 2\n```")
        evaluator = Mock()
        evaluator.evaluate_program = AsyncMock(return_value={"combined_score": 0.9})
        evaluator.get_pending_artifacts.return_value = {"stdout": "child evidence"}
        with (
            patch.object(workers, "_lazy_init_worker_components"),
            patch.object(workers, "_worker_config", config, create=True),
            patch.object(workers, "_worker_prompt_sampler", sampler, create=True),
            patch.object(workers, "_worker_llm_ensemble", llm, create=True),
            patch.object(workers, "_worker_evaluator", evaluator, create=True),
        ):
            result = workers._run_iteration_worker(7, pickle.loads(pickle.dumps(context)))
        self.assertIsNone(result.error)
        self.assertEqual(result.target_island, 1)
        self.assertEqual(result.child_program_dict["metadata"]["island"], 1)
        self.assertEqual(result.child_program_dict["iteration_found"], 7)
        self.assertEqual(result.parent_id, "parent")
        self.assertEqual(result.artifacts, {"stdout": "child evidence"})
        prompt_args = sampler.build_prompt.call_args.kwargs
        self.assertEqual(prompt_args["program_artifacts"], {"stderr": "parent evidence"})
        self.assertEqual(prompt_args["previous_programs"], [parent.to_dict()])
        self.assertEqual(prompt_args["feature_dimensions"], ["complexity"])
