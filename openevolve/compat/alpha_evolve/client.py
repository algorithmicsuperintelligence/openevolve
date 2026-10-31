"""Local execution options with a small AlphaEvolve-shaped constructor."""

import warnings
from pathlib import Path
from typing import Optional, Union

from openevolve.compat.alphaevolve import adapt_evaluator


class AlphaEvolveClient:
    """Configure local execution without a Google SDK, credentials or requests.

    ``api_base`` and ``objective_metric`` are explicit OpenEvolve settings.
    Cloud arguments are accepted only for entrypoint migration and warn when
    supplied; ``base_url`` never becomes the model endpoint. ``iterations`` may
    reduce the imported generation budget, including zero for seed-only use.
    An omitted key retains OpenEvolve's usual provider/environment behavior.
    """

    def __init__(
        self,
        project_id=None,
        location=None,
        collection=None,
        engine=None,
        assistant=None,
        base_url=None,
        *,
        api_base: str,
        objective_metric: str,
        api_key: Optional[str] = None,
        iterations: Optional[int] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        if not isinstance(api_base, str) or not api_base.strip():
            raise ValueError("api_base must explicitly name the OpenEvolve model endpoint")
        # Reuse the evaluator adapter's objective validation without executing it.
        adapt_evaluator(lambda candidate: None, objective_metric=objective_metric)
        if iterations is not None and (
            isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0
        ):
            raise ValueError("iterations must be a non-negative integer or None")
        if output_dir is not None and not isinstance(output_dir, (str, Path)):
            raise TypeError("output_dir must be a path or None")
        if any(
            value is not None
            for value in (project_id, location, collection, engine, assistant, base_url)
        ):
            warnings.warn(
                "Local AlphaEvolve facade ignores Google Cloud connection arguments; "
                "api_base selects the OpenEvolve endpoint",
                UserWarning,
                stacklevel=2,
            )
        self.api_base = api_base
        self.objective_metric = objective_metric
        self.api_key = api_key
        self.iterations = iterations
        self.output_dir = str(output_dir) if output_dir is not None else None
        self._experiments = {}

    def get_alpha_evolve_experiment(self, experiment_name: str) -> dict:
        """Return name/state for an experiment created by this local client.

        This is an in-memory diagnostic lookup, not a cloud query or SDK
        experiment resource. Configuration, credentials and program data are
        intentionally not returned. The client retains its local experiments.
        """
        if not isinstance(experiment_name, str) or experiment_name not in self._experiments:
            raise ValueError("experiment_name must identify an experiment created by this client")
        experiment = self._experiments[experiment_name]
        return {"name": experiment.experiment_name, "state": experiment.state}
