"""One-shot synchronous local experiments using OpenEvolve's public API."""

import asyncio
import copy
import inspect
import math
import tempfile
import threading
import uuid
import warnings
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Optional

from openevolve.api import run_evolution
from openevolve.compat.alphaevolve import adapt_evaluator, config_from_alphaevolve

from .client import AlphaEvolveClient


class AlphaEvolveExperiment:
    """A limited SDK-shaped lifecycle, not a cloud client or full SDK replacement.

    Creation only validates inputs. ``start_experiment`` blocks until the local
    run finishes and may invoke the explicitly configured model endpoint.
    ``list_programs`` exposes only the best result, not the whole OE archive.
    Instances are single-use, including after failure or interruption.
    """

    def __init__(
        self,
        ae_client: AlphaEvolveClient,
        evaluator_function: Callable,
        max_programs_evaluated: int,
        parallel_evaluation: bool = False,
    ):
        if not isinstance(ae_client, AlphaEvolveClient):
            raise TypeError("ae_client must be the local AlphaEvolveClient")
        if (
            not callable(evaluator_function)
            or inspect.iscoroutinefunction(evaluator_function)
            or inspect.iscoroutinefunction(getattr(evaluator_function, "__call__", None))
        ):
            raise TypeError("evaluator_function must be a synchronous callable")
        if (
            isinstance(max_programs_evaluated, bool)
            or not isinstance(max_programs_evaluated, int)
            or max_programs_evaluated < 1
        ):
            raise ValueError("max_programs_evaluated must be a positive integer")
        if parallel_evaluation is not False:
            raise ValueError("This local facade supports only parallel_evaluation=False")
        self.client = ae_client
        self.max_programs_evaluated = max_programs_evaluated
        self.experiment_name = None
        self.initial_program_name = None
        self.result = None
        self.state = "NEW"
        self._evaluator = adapt_evaluator(
            evaluator_function, objective_metric=ae_client.objective_metric
        )
        self._config = None
        self._source = None
        self._source_path = None

    def _require_state(self, expected: str) -> None:
        if self.state != expected:
            raise RuntimeError(f"Expected experiment state {expected}, got {self.state}")

    def create_experiment(self, config: Mapping) -> None:
        """Import the existing supported experiment-config subset; no execution."""
        self._require_state("NEW")
        local = config_from_alphaevolve(
            config, api_base=self.client.api_base, api_key=self.client.api_key
        )
        if local.max_iterations != self.max_programs_evaluated:
            raise ValueError(
                "max_programs_evaluated must equal run_settings.max_programs in this facade; "
                "distinct generation/evaluation budgets are unsupported"
            )
        iterations = self.client.iterations
        if iterations is not None:
            if iterations > local.max_iterations:
                raise ValueError("iterations cannot increase the imported generation budget")
            local.max_iterations = iterations
        warnings.warn(
            "Local max_programs_evaluated is a generation-attempt ceiling, not an "
            "evaluation/request/cost cap; seed evaluation and retries are additional",
            UserWarning,
            stacklevel=2,
        )
        self._config = local
        self.experiment_name = "local/experiments/" + uuid.uuid4().hex
        self.state = "CREATED"
        self.client._experiments[self.experiment_name] = self

    def create_initial_program(self, initial_program: Mapping) -> None:
        """Accept one UTF-8 Python source; supplied seed scores are not trusted."""
        self._require_state("CREATED")
        if not isinstance(initial_program, Mapping):
            raise ValueError("initial_program must be a candidate mapping")
        content = initial_program.get("content")
        files = content.get("files") if isinstance(content, Mapping) else None
        if not isinstance(files, (list, tuple)) or len(files) != 1:
            raise ValueError("Only single-file Python candidates are supported")
        entry = files[0]
        if not isinstance(entry, Mapping):
            raise ValueError("Candidate file must be a mapping")
        source, path = entry.get("content"), entry.get("path")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Candidate content must be non-empty Python source")
        if not isinstance(path, str) or not path.endswith(".py"):
            raise ValueError("Candidate path must name a Python .py file")
        if entry.get("program_language") not in (None, "python"):
            raise ValueError("Only Python candidates are supported")
        # Validate without executing; the supplied path is metadata, never a write target.
        source.encode("utf-8")
        compile(source, "<alphaevolve-seed>", "exec")
        if initial_program.get("evaluation") is not None:
            warnings.warn(
                "Supplied seed evaluation is ignored; OpenEvolve re-evaluates the seed",
                UserWarning,
                stacklevel=2,
            )
        self._source = source
        self._source_path = path
        self.initial_program_name = self.experiment_name + "/programs/seed"
        self.state = "READY"

    def start_experiment(self) -> None:
        """Run once on the main thread, outside an already-running asyncio loop."""
        self._require_state("READY")
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Start local experiments on the main thread for OE signal handling")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Call start_experiment outside a running asyncio loop")
        self.state = "RUNNING"
        try:
            with tempfile.TemporaryDirectory(prefix="openevolve_ae_seed_") as directory:
                seed = Path(directory) / "program.py"
                seed.write_bytes(self._source.encode("utf-8"))
                result = run_evolution(
                    seed,
                    self._evaluator,
                    config=copy.deepcopy(self._config),
                    output_dir=self.client.output_dir,
                    cleanup=True,
                )
            if result.best_program is None:
                raise RuntimeError("Local evolution returned no best program")
            self.result = result
        except BaseException:
            self.state = "FAILED"
            raise
        self.state = "COMPLETED"

    def list_programs(self, params: Optional[Mapping] = None) -> dict[str, Any]:
        """Return the best result in the SDK envelope; no archive or pagination."""
        self._require_state("COMPLETED")
        if params is not None and not isinstance(params, Mapping):
            raise ValueError("params must be a mapping or None")
        params = params or {}
        if set(params) - {"order_by"}:
            raise ValueError("Only order_by is supported; archive queries are unavailable")
        metrics = self.result.metrics
        scores = [
            {"metric": metric, "score": float(value)}
            for metric, value in metrics.items()
            if isinstance(value, Real)
            and not isinstance(value, bool)
            and math.isfinite(value)
            and metric != "error"
            and (metric != "combined_score" or metric == self.client.objective_metric)
        ]
        if "order_by" in params:
            order = params["order_by"]
            parts = order.rsplit(" ", 1) if isinstance(order, str) else []
            # Some upstream entrypoints use a generic score ordering. It names
            # the explicitly selected objective, not an average of all metrics.
            if len(parts) == 2 and parts[0] == "score":
                parts[0] = self.client.objective_metric
            if (
                len(parts) != 2
                or parts[1] not in ("asc", "desc")
                or parts[0] not in {score["metric"] for score in scores}
            ):
                raise ValueError("order_by must name a returned metric (or score) and asc or desc")
        evaluation = {"scores": {"scores": scores}}
        if "error" in metrics:
            evaluation["insights"] = {
                "insights": [{"label": "OpenEvolve Evaluation Error", "text": "Evaluation failed"}]
            }
        return {
            "alphaEvolvePrograms": [
                {
                    "name": self.experiment_name + "/programs/" + self.result.best_program.id,
                    "content": {
                        "files": [{"path": self._source_path, "content": self.result.best_code}]
                    },
                    "evaluation": evaluation,
                }
            ]
        }
