import os
import sys
import pathlib
import argparse
import numpy as np

BASE_DIR = pathlib.Path(__file__).parent
sys.path.append(str(BASE_DIR))

# other imports
from heat_map_train import *


def calc_heat_map(cities: np.ndarray) -> np.ndarray:
    cities_number = cities.shape[0]

    # You can call the pretrained model here (that was trained in `heat_map_train.py` and saved in str(BASE_DIR) / "pretrained")
    # If you want to run some algorithm that is not trainable, then maybe it is better to implement it in TSP.cpp (just because it is faster in c++).

    return np.zeros((cities_number, cities_number), dtype=float)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Path to .npy to write the heat map.")
    args = p.parse_args()

    # reading cities from stdin (mind the input format if you want to change this part, as your solution will be tested in an environment)
    first_stdin_line = sys.stdin.readline().strip()
    cities_number = int(first_stdin_line)
    cities = np.loadtxt(sys.stdin, max_rows=cities_number)  # of shape (cities_number, 2)

    # calculating heat map
    heat_map = calc_heat_map(cities).astype(np.float32, copy=False)
    print("Sample output to log")

    # writing heat map atomically to out .npy file (mind the output format if you want to change this part, as your solution will be tested in an environment)
    tmp_path = args.out + ".tmp"

    with open(tmp_path, "wb") as output_temp_file:
        np.save(output_temp_file, heat_map, allow_pickle=False)  # no auto “.npy” added
        output_temp_file.flush()
        os.fsync(output_temp_file.fileno())
    
    os.replace(tmp_path, args.out)