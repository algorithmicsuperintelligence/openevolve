import re
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


def latest_checkpoint(dir_path: pathlib.Path) -> str | None:
    if not dir_path.exists():
        return None

    candidates: list[tuple[int, pathlib.Path]] = []
    for path in dir_path.glob("checkpoint_*"):
        if not path.is_dir():
            continue

        match = re.fullmatch(r"checkpoint_(\d+)", path.name)
        if match:
            candidates.append((int(match.group(1)), path))

    if not candidates:
        return None

    _, newest_path = max(candidates, key=lambda t: (t[0], t[1].stat().st_mtime))
    return str(newest_path)


async def run_evolution(evolve: OpenEvolve, checkpoint_path: str | None) -> None:
    best_program = await evolve.run(checkpoint_path=checkpoint_path)

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
    help="Output directory for OpenEvolve runs (checkpoints, logs, etc.).",
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

    checkpoint_path = latest_checkpoint(openevolve_output_dir / "checkpoints")
    print(f"Using checkpoint: '{checkpoint_path}'")

    asyncio.run(run_evolution(evolve, checkpoint_path))


if __name__ == "__main__":
    cli()
