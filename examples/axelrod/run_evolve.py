import asyncio
import logging
import os
import uuid

from openevolve import OpenEvolve
from openevolve.config import load_config
from openevolve.database import Program

AXELROD_DIR = os.path.join("examples", "axelrod")

logger = logging.getLogger(__name__)


async def main():
    initial_seeds_files = ["seed1.py", "seed2.py", "seed3.py", "seed4.py"]
    initial_seeds = [os.path.join(AXELROD_DIR, f) for f in initial_seeds_files]

    # Load base config from file or defaults
    config = load_config(os.path.join(AXELROD_DIR, "config.yaml"))

    openevolve = OpenEvolve(
        initial_program_path=initial_seeds[0], # Default for index 0
        evaluation_file=os.path.join(AXELROD_DIR, "evaluator.py"),
        config=config,
    )

    # Manually insert other seeds into the database/islands before running
    for i, seed_path in enumerate(initial_seeds):
        with open(seed_path, 'r') as f:
            code = f.read()
            logger.info("Adding initial program to database")
            initial_program_id = str(uuid.uuid4())

            metrics = await openevolve.evaluator.evaluate_program(
                code, initial_program_id
            )

            program = Program(
                id=initial_program_id,
                code=code,
                language=openevolve.config.language,
                metrics=metrics,
                iteration_found=0,
            )
            openevolve.database.add(program, target_island=i)

    # Run evolution
    best_program = await openevolve.run()  # checkpoint_path="examples/axelrod/openevolve_output/checkpoints/checkpoint_25")

    print(f"\nEvolution complete!")
    print(f"Best program metrics:")
    for name, value in best_program.metrics.items():
        # Handle mixed types: format numbers as floats, others as strings
        if isinstance(value, (int, float)):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
