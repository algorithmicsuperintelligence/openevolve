import sys
import pathlib
import asyncio

BASE_DIR = pathlib.Path(__file__).parent
sys.path.append(str(BASE_DIR))

# click & related imports
import click

# openevolve & related imports
from openevolve import Config, OpenEvolve

# local imports
from utils.utils import *
from utils.code_to_query import *


async def run_evolution(evolve: OpenEvolve) -> None:
    best_program = await evolve.run()

    print("Best program metrics:")
    for name, value in best_program.metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")


@click.command(context_settings={"show_default": True})
@click.option(
    "--initial_program_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path),
    default=BASE_DIR / "initial_program",
    help="Directory with the initial program source (folder).",
)
@click.option(
    "--openevolve_output_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=pathlib.Path),
    default=BASE_DIR / "openevolve_output",
    help="Output directory for OpenEvolve results and logs.",
)
def cli(initial_program_dir: pathlib.Path, openevolve_output_dir: pathlib.Path) -> None:
    initial_program_dir = initial_program_dir.resolve()
    openevolve_output_dir = openevolve_output_dir.resolve()
    openevolve_output_dir.mkdir(parents=True, exist_ok=True)

    initial_program_path = openevolve_output_dir / "initial_program.txt"

    print(f"Initial program dir: '{initial_program_dir}'")
    print(f"Initial program path: '{initial_program_path}'")

    initial_program_path.write_text(format_query_code(str(initial_program_dir)))

    evolve = OpenEvolve(
        initial_program_path=str(initial_program_path),
        evaluation_file=str(BASE_DIR / "evaluator.py"),
        config=Config.from_yaml(str(BASE_DIR / "config.yaml")),
        output_dir=str(openevolve_output_dir),
    )

    asyncio.run(run_evolution(evolve))


if __name__ == "__main__":
    cli()
