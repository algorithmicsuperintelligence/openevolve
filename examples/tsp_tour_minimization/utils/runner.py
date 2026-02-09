import sys
import json
import pathlib

BASE_DIR = pathlib.Path(__file__).parents[1]
sys.path.append(str(BASE_DIR))

# local imports
from utils.utils import *
from utils.tsp_runner import *
from utils.heat_map_runner import *


def format_input_file(input_file_path: str, cities: np.ndarray, heat_map: np.ndarray) -> None:
    with open(input_file_path, 'w') as file:
        cities_str = '\n'.join(map(lambda x: ' '.join(map(str, x)), cities))
        heat_map_str = '\n'.join(map(lambda x: ' '.join(map(str, x)), heat_map))
        file.write(f"{cities.shape[0]}\n{cities_str}\n{heat_map_str}")


def change_config_file(input_config_file_path: str, output_config_file_path: str, cities_number: int, input_path: str, output_path: str) -> None:
    config = json.loads(read_file(input_config_file_path))

    config["cities_number"] = cities_number
    config["input_path"] = input_path
    config["output_path"] = output_path

    with open(output_config_file_path, 'w') as file:
        file.write(json.dumps(config, indent=4, ensure_ascii=False))


def parse_output_files(output_files_paths: list[str]) -> np.ndarray:
    results = []

    for output_file_path in output_files_paths:
        with open(output_file_path, 'r') as file:
            results.append( np.array(list(map(int, file.read().strip().split()))) )
    
    return results


def run(
    dir_path: str,
    cities: np.ndarray,
    heat_map_train_timeout: float = 360.0,
    heat_map_inference_timeout: float = 60.0,
    tsp_compilation_timeout: float = 10.0,
    tsp_run_timeout: float = 60.0,
    verbose: bool = False,
    output_saver: dict | None = None
) -> dict:
    create_dir(f"{dir_path}/input_files")
    create_dir(f"{dir_path}/output_files")
    create_dir(f"{dir_path}/config_files")

    input_paths = [f"{dir_path}/input_files/instance_{i:05d}.txt" for i in range(cities.shape[0])]
    output_paths = [f"{dir_path}/output_files/instance_{i:05d}.txt" for i in range(cities.shape[0])]
    config_paths = [f"{dir_path}/config_files/instance_{i:05d}.json" for i in range(cities.shape[0])]

    # train
    heat_map_train_data = run_python_heat_map_train(f"{dir_path}/heat_map_train.py", timeout=heat_map_train_timeout)
    if verbose: print("Heat map train complete.")
    if output_saver is not None: output_saver["heat_map_train_data"] = heat_map_train_data

    # inference
    heat_map_inference_results = run_heat_maps_parallel(f"{dir_path}/heat_map_inference.py", cities, f"{dir_path}/heat_map_results", heat_map_inference_timeout, max_workers=None)  # in format [(index, npy_path, time_elapsed), ...]
    if verbose: print("Heat map inference complete.")
    if output_saver is not None: output_saver["heat_map_inference_data"] = heat_map_inference_results

    # compilation of TSP.cpp
    compile_tsp_executable(dir_path, timeout=tsp_compilation_timeout)
    if verbose: print("Compilation complete.")

    # building input files
    for i in range(cities.shape[0]):
        format_input_file(input_paths[i], cities[i], np.load(heat_map_inference_results["heat_map_paths"][i]))
    if verbose: print("Building input files complete.")

    # config changing
    for i in range(cities.shape[0]):
        change_config_file(f"{dir_path}/config.json", config_paths[i], cities.shape[1], input_paths[i], output_paths[i])
    if verbose: print("Building config files complete.")

    # running compiled binary
    tsp_run_data = run_runner_parallel(f"{dir_path}/bin/runner", config_paths, timeout=tsp_run_timeout)
    if verbose: print("TSP run complete.")
    if output_saver is not None: output_saver["tsp_run_data"] = tsp_run_data

    # parsing output
    solutions = parse_output_files(output_paths)
    if verbose: print("Parsing output files complete.")
    if output_saver is not None: output_saver["solutions"] = solutions

    return {
        "heat_map_train_data": heat_map_train_data,
        "heat_map_inference_data": heat_map_inference_results,
        "tsp_run_data": tsp_run_data,
        "solutions": solutions,
    }
