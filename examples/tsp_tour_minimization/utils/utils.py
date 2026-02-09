import os
import math
import shutil
import pathlib
import typing as tp
import concurrent.futures

# numpy & related imports
import numpy as np


BHH_CONSTANT_2D = 0.7120


def get_cpu_cores_number() -> int:
    return os.cpu_count()


def default_for_none(value: tp.Any, default: tp.Any) -> tp.Any:
    if value is None: return default
    return value


def create_dir(dir_path: str) -> None:
    os.makedirs(dir_path, exist_ok=True)


def remove_dir(dir_path: str) -> None:
    p = pathlib.Path(dir_path)

    if not p.exists():
        return

    if not p.is_dir():
        raise NotADirectoryError(f"{dir_path} is not a directory")

    shutil.rmtree(p)


def touch_file(file_path: str) -> None:
    file_path = pathlib.Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch(exist_ok=True)


def read_file(file_path: str) -> str:
    with open(file_path, 'r', encoding="utf-8") as file:
        return file.read().strip()


def is_file_exist(file_path: str) -> bool:
    return pathlib.Path(file_path).exists()


def expected_random_cycle_length(n: int, h: float = 1.0, w: float = 1.0) -> float:
    if n < 3:
        raise ValueError("n must be >= 3 for a Hamiltonian cycle")
    if h <= 0 or w <= 0:
        raise ValueError("h and w must be positive")

    w2, h2 = w * w, h * h
    d = math.hypot(w, h)  # sqrt(w^2 + h^2)

    term1 = (w ** 3) / h2 + (h ** 3) / w2
    term2 = d * (3.0 - w2 / h2 - h2 / w2)
    term3 = 2.5 * ((h2 / w) * math.log((w + d) / h) + (w2 / h) * math.log((h + d) / w))

    mean_edge = (term1 + term2 + term3) / 15.0
    return n * mean_edge


def approximation_using_BHH_constant(n: int, h: float = 1.0, w: float = 1.0) -> float:
    return BHH_CONSTANT_2D * ((n * h * w) ** 0.5)


def run_with_timeout(func: tp.Callable, args=(), kwargs={}, timeout: float = 5.0):
    """
    Run a function with a timeout using concurrent.futures

    Args:
        func: Function to run
        args: Arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        timeout_seconds: Timeout in seconds

    Returns:
        Result of the function or raises TimeoutError
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout)
            return result
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function timed out after {timeout} seconds.")


def check_on_hamiltonian_cycle(cities: np.ndarray, solution: np.ndarray) -> bool:
    """
    Return True if:
      - cities is ndarray of shape (n, 2), n >= 3, all finite
      - solution is 1D ndarray of length n
      - solution contains each index 0..(n-1) exactly once (permutation)
    """
    if not isinstance(cities, np.ndarray) or not isinstance(solution, np.ndarray):
        return False
    if cities.ndim != 2 or cities.shape[1] != 2:
        return False

    n = cities.shape[0]
    if n < 3:
        return False
    if solution.ndim != 1 or solution.size != n:
        return False

    if not np.isfinite(cities).all():
        return False

    if not np.issubdtype(solution.dtype, np.integer):
        if not np.isfinite(solution).all():
            return False
        if not np.all(solution == np.floor(solution)):
            return False
        solution = solution.astype(np.int64, copy=False)
    else:
        solution = solution.astype(np.int64, copy=False)

    if solution.min() < 0 or solution.max() >= n:
        return False

    counts = np.bincount(solution, minlength=n)
    if counts.size != n or not np.all(counts == 1):
        return False

    return True


def calc_total_cycle_distance(cities: np.ndarray, solutions: list[np.ndarray] | np.ndarray) -> np.ndarray:
    """
    Compute total cycle distances for a batch of instances

    Args:
        cities: np.ndarray of shape (B, n, 2). All finite. n >= 3
        solutions: either
            - list of length B, each element a 1D np.ndarray of shape (n,), or
            - np.ndarray of shape (B, n) with integer-like indices

    Returns:
        np.ndarray of shape (B,) with total tour length (float64) per instance

    Raises:
        TypeError, ValueError with explicit messages when inputs are invalid
    """
    if not isinstance(cities, np.ndarray):
        raise TypeError(f"`cities` must be a numpy.ndarray, got {type(cities).__name__}")
    if cities.ndim != 3 or cities.shape[-1] != 2:
        raise ValueError(f"`cities` must have shape (batch_size, n, 2), got {cities.shape}")
    if not np.isfinite(cities).all():
        raise ValueError("`cities` contains non-finite values (NaN or inf)")

    B, n, _ = cities.shape
    if B < 1:
        raise ValueError("`cities` must have batch_size >= 1")
    if n < 3:
        raise ValueError("`cities` must have n >= 3")

    if isinstance(solutions, list):
        if len(solutions) != B:
            raise ValueError(f"`solutions` list length {len(solutions)} != batch size, which is {B}")
        
        sols = np.empty((B, n), dtype=np.int64)

        for b in range(B):
            sol_b = np.asarray(solutions[b])
            ok = check_on_hamiltonian_cycle(cities[b], sol_b)

            if not ok:
                raise ValueError(
                    f"`solutions[{b}]` is not a valid Hamiltonian permutation for its cities (expected 1D permutation of 0..{n-1} and cities shape (n, 2))"
                )
            
            sols[b] = sol_b.astype(np.int64, copy=False)
    elif isinstance(solutions, np.ndarray):
        if solutions.ndim != 2 or solutions.shape != (B, n):
            raise ValueError(f"`solutions` must have shape (B, n) when given as ndarray, got {solutions.shape}")

        sols = np.empty_like(solutions, dtype=np.int64)

        for b in range(B):
            sol_b = solutions[b]
            ok = check_on_hamiltonian_cycle(cities[b], sol_b)

            if not ok:
                raise ValueError(f"`solutions[b]` at b={b} is not a valid Hamiltonian permutation for its cities")
            
            sols[b] = sol_b.astype(np.int64, copy=False)
    else:
        raise TypeError(f"`solutions` must be list[np.ndarray] or np.ndarray, got {type(solutions).__name__}")

    ordered = cities[np.arange(B)[:, None], sols]  # (B, n, 2)
    next_ordered = np.roll(ordered, shift=-1, axis=1)  # (B, n, 2)
    deltas = next_ordered - ordered  # (B, n, 2)
    segment_lengths = np.linalg.norm(deltas, axis=2)  # (B, n)
    totals = segment_lengths.sum(axis=1).astype(np.float64)  # (B,)

    return totals
