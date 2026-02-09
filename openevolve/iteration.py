import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.evaluator import Evaluator
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.prompt.sampler import PromptSampler
from openevolve.utils.code_utils import (
    apply_diff,
    apply_diff_blocks,
    extract_diffs,
    format_diff_summary,
    parse_full_rewrite,
    split_diffs_by_target,
)


@dataclass
class Result:
    """Resulting program and metrics from an iteration of OpenEvolve"""

    child_program: str = None
    parent: str = None
    child_metrics: str = None
    iteration_time: float = None
    prompt: str = None
    llm_response: str = None
    artifacts: dict = None


async def run_iteration_with_shared_db(
    iteration: int,
    config: Config,
    database: ProgramDatabase,
    evaluator: Evaluator,
    llm_ensemble: LLMEnsemble,
    prompt_sampler: PromptSampler,
):
    """
    Run a single iteration using shared memory database

    This is optimized for use with persistent worker processes.
    """
    logger = logging.getLogger(__name__)

    try:
        # Sample parent and inspirations from database
        parent, inspirations = database.sample(num_inspirations=config.prompt.num_top_programs)

        # Get artifacts for the parent program if available
        parent_artifacts = database.get_artifacts(parent.id)

        # Get island-specific top programs for prompt context (maintain island isolation)
        parent_island = parent.metadata.get("island", database.current_island)
        island_top_programs = database.get_top_programs(5, island_idx=parent_island)
        island_previous_programs = database.get_top_programs(3, island_idx=parent_island)

        # Build prompt
        if config.prompt.programs_as_changes_description:
            parent_changes_desc = (
                parent.changes_description or config.prompt.initial_changes_description
            )
            child_changes_desc = parent_changes_desc
        else:
            parent_changes_desc = None
            child_changes_desc = None

        prompt = prompt_sampler.build_prompt(
            current_program=parent.code,
            parent_program=parent.code,
            program_metrics=parent.metrics,
            previous_programs=[p.to_dict() for p in island_previous_programs],
            top_programs=[p.to_dict() for p in island_top_programs],
            inspirations=[p.to_dict() for p in inspirations],
            language=config.language,
            evolution_round=iteration,
            diff_based_evolution=config.diff_based_evolution,
            program_artifacts=parent_artifacts if parent_artifacts else None,
            feature_dimensions=database.config.feature_dimensions,
            current_changes_description=parent_changes_desc,
        )

        result = Result(parent=parent)
        iteration_start = time.time()

        # Generate code modification
        llm_response = await llm_ensemble.generate_with_context(
            system_message=prompt["system"],
            messages=[{"role": "user", "content": prompt["user"]}],
        )

        # Parse the response
        if config.diff_based_evolution:
            diff_blocks = extract_diffs(llm_response, config.diff_pattern)

            if not diff_blocks:
                logger.warning(f"Iteration {iteration+1}: No valid diffs found in response")
                return None

            if config.prompt.programs_as_changes_description:
                try:
                    code_blocks, desc_blocks, _unmatched = split_diffs_by_target(
                        diff_blocks,
                        code_text=parent.code,
                        changes_description_text=parent_changes_desc,
                    )
                except Exception as e:
                    logger.warning(f"Iteration {iteration + 1}: {e}")
                    return None

                child_code, _ = apply_diff_blocks(parent.code, code_blocks)
                child_changes_desc, desc_applied = apply_diff_blocks(
                    parent_changes_desc, desc_blocks
                )

                # Must update the previous changes description
                if (
                    desc_applied == 0
                    or not child_changes_desc.strip()
                    or child_changes_desc.strip() == parent_changes_desc.strip()
                ):
                    logger.warning(
                        f"Iteration {iteration+1}: changes_description was not updated or empty, program is discarded"
                    )
                    return None

                changes_summary = format_diff_summary(
                    code_blocks,
                    max_line_len=config.prompt.diff_summary_max_line_len,
                    max_lines=config.prompt.diff_summary_max_lines,
                )
            else:
                # All diffs applied only to code
                child_code = apply_diff(parent.code, llm_response, config.diff_pattern)
                changes_summary = format_diff_summary(
                    diff_blocks,
                    max_line_len=config.prompt.diff_summary_max_line_len,
                    max_lines=config.prompt.diff_summary_max_lines,
                )
        else:
            # Parse full rewrite
            new_code = parse_full_rewrite(llm_response, config.language)

            if not new_code:
                logger.warning(f"Iteration {iteration+1}: No valid code found in response")
                return None

            child_code = new_code
            changes_summary = "Full rewrite"

        # Check code length
        if len(child_code) > config.max_code_length:
            logger.warning(
                f"Iteration {iteration+1}: Generated code exceeds maximum length "
                f"({len(child_code)} > {config.max_code_length})"
            )
            return None

        # Evaluate the child program
        child_id = str(uuid.uuid4())
        result.child_metrics = await evaluator.evaluate_program(child_code, child_id)

        # Handle artifacts if they exist
        artifacts = evaluator.get_pending_artifacts(child_id)

        # Set template_key of Prompts
        template_key = "full_rewrite_user" if not config.diff_based_evolution else "diff_user"

        # Create a child program
        result.child_program = Program(
            id=child_id,
            code=child_code,
            changes_description=child_changes_desc,
            language=config.language,
            parent_id=parent.id,
            generation=parent.generation + 1,
            metrics=result.child_metrics,
            iteration_found=iteration,
            metadata={
                "changes": changes_summary,
                "parent_metrics": parent.metrics,
            },
            prompts=(
                {
                    template_key: {
                        "system": prompt["system"],
                        "user": prompt["user"],
                        "responses": [llm_response] if llm_response is not None else [],
                    }
                }
                if database.config.log_prompts
                else None
            ),
        )

        result.prompt = prompt
        result.llm_response = llm_response
        result.artifacts = artifacts
        result.iteration_time = time.time() - iteration_start
        result.iteration = iteration

        return result

    except Exception as e:
        logger.exception(f"Error in iteration {iteration}: {e}")
        return None
