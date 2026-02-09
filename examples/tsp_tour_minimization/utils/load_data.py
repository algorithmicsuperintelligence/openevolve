import sys
import pathlib

BASE_DIR = pathlib.Path(__file__).parents[1]
sys.path.append(str(BASE_DIR))

# torch & related imports
import numpy as np

# local imports
from utils.utils import *


def generate_random_dataset(file_path: str, n: int, instances_number: int = 128) -> None:
    np.save(file_path, np.random.uniform(0, 1, size=(instances_number, n, 2)))


def _load_points(file_path: str) -> np.ndarray:
    return np.load(file_path)


def load_points(output_dir: str, n: int, instances_count: int = 128) -> np.ndarray:
    file_path = pathlib.Path(output_dir) / f"points/{n}.npy"

    if not is_file_exist(str(file_path)):
        create_dir(str(file_path.parent))

        # generating a new dataset
        generate_random_dataset(str(file_path), n, instances_number=instances_count)

    return _load_points(str(file_path))[:instances_count]
