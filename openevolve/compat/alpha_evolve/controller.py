"""Async-shaped completion check for synchronous local experiments."""

import warnings
from typing import Optional

from .experiment import AlphaEvolveExperiment


async def run_controller_loop(
    experiment: AlphaEvolveExperiment,
    *,
    num_samplers: Optional[int] = None,
    num_evaluators: Optional[int] = None,
) -> None:
    """Check completion without running again, polling queues or creating workers.

    The local ``start_experiment`` already ran OpenEvolve to completion.
    Explicit positive sampler/evaluator counts are accepted for entrypoint
    migration and warn that they do not create workers or affect concurrency.
    Other cloud controller options remain unsupported.
    """
    if not isinstance(experiment, AlphaEvolveExperiment):
        raise TypeError("experiment must be the local AlphaEvolveExperiment")
    for value in (num_samplers, num_evaluators):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError("num_samplers and num_evaluators must be positive integers or None")
    experiment._require_state("COMPLETED")
    if num_samplers is not None or num_evaluators is not None:
        warnings.warn(
            "Local run_controller_loop ignores num_samplers/num_evaluators; "
            "start_experiment already completed the synchronous OE run",
            UserWarning,
            stacklevel=2,
        )
