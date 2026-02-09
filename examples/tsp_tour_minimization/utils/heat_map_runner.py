import os
import io
import sys
import time
import signal
import pathlib
import asyncio
import subprocess
import contextlib
import typing as tp

BASE_DIR = pathlib.Path(__file__).parents[1]
sys.path.append(str(BASE_DIR))

# numpy & related imports
import numpy as np

# local imports
from utils.utils import *


# forcing single-threaded BLAS for every spawned child
THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def run_python_heat_map_train(
    program_path: str,
    timeout: float,
    python_executable: str = sys.executable,
) -> dict:
    """
    Runs python program as a subprocess, returns stdout/stderr + elapsed time
    Kills the whole process group on timeout (POSIX)
    """

    cmd = [python_executable, program_path]

    popen_kwargs: dict = dict(
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env={**os.environ, **THREAD_ENV},  # stopping OpenBLAS from trying to spawn many threads
    )

    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    else:
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    time_start = time.monotonic()
    process: subprocess.Popen[str] = subprocess.Popen(cmd, **popen_kwargs)

    try:
        stdout, stderr = process.communicate("", timeout=timeout)
    except subprocess.TimeoutExpired:
        # kill (group on POSIX, process on Windows)
        if os.name == "posix":
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError, OSError, PermissionError):
                    os.killpg(process.pid, signal.SIGKILL)
        else:
            if process.poll() is None:
                with contextlib.suppress(OSError, PermissionError):
                    process.kill()

        raise TimeoutError(f"Time exceeded {timeout} seconds.")

    time_elapsed = time.monotonic() - time_start

    if process.returncode != 0:
        raise RuntimeError(
            f"Program failed with exit code {process.returncode}.\nSTDERR:\n{stderr}"
        )

    return {
        "time_elapsed": time_elapsed,
        "stdout": stdout,
        "stderr": stderr,
    }


def format_heat_map_stdin(cities: np.ndarray) -> str:
    return f"{cities.shape[0]}\n" + "\n".join(f"{x:.17g} {y:.17g}" for x, y in cities)


def parse_heat_map_stdout(stdout: str, n: int) -> np.ndarray:
    array = np.atleast_2d(np.loadtxt(io.StringIO(stdout)))

    if array.shape != (n, n):
        raise ValueError(f"Expected heat map of shape ({n}, {n}), but got {array.shape}")

    return array


def _default_workers(n: tp.Optional[int]) -> int:
    if n and n > 0:
        return n

    cores_count = get_cpu_cores_number() or 1
    return max(1, min(cores_count, 4))


async def _run_one_inference(
    index: int,
    program_path: str,
    python_executable: str,
    cities_i: np.ndarray,
    out_path: str,
    timeout: float,
) -> tuple[int, str, float, bytes, bytes]:
    """
    Launch one child, send coordinates via stdin, child writes .npy to out_path
    Returns (idx, out_path_str, elapsed_sec, raw_stdout, raw_stderr)
    Raises on error/timeout
    Kills the whole process group on cancel/timeout
    """

    stdin_bytes = format_heat_map_stdin(cities_i).encode("utf-8")

    creationflags = 0
    start_new_session = False
    if os.name == "posix":
        start_new_session = True
    else:
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    # inherit env but force BLAS to one thread to avoid
    # "pthread_create failed ... RLIMIT_NPROC" error
    child_env = os.environ.copy()
    child_env.update(THREAD_ENV)

    process = await asyncio.create_subprocess_exec(
        python_executable,
        program_path,
        "--out",
        out_path,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=start_new_session,
        **({"creationflags": creationflags} if not os.name == "posix" else {}),
        env=child_env,
    )

    start_time = time.monotonic()

    try:
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(stdin_bytes), timeout=timeout
            )
        except asyncio.TimeoutError:
            # hard kill on timeout
            if os.name == "posix":
                if process.returncode is None:
                    with contextlib.suppress(ProcessLookupError, OSError, PermissionError):
                        os.killpg(process.pid, signal.SIGKILL)
            else:
                if process.returncode is None:
                    with contextlib.suppress(OSError, PermissionError):
                        process.send_signal(signal.CTRL_BREAK_EVENT)

                    await asyncio.sleep(0.2)
                    if process.returncode is None:
                        with contextlib.suppress(OSError, PermissionError):
                            process.kill()

            raise TimeoutError(f"[{index}] timed out after {timeout}s.")

        return_code = process.returncode
        time_elapsed = time.monotonic() - start_time

        if return_code != 0:
            error = (stderr.decode("utf-8", "ignore") if stderr else "").strip()
            raise RuntimeError(f"[{index}] exited with {return_code} return code. stderr:\n{error[:4000]}")

        # shape check
        array = np.load(out_path, mmap_mode="r")
        n = cities_i.shape[0]
        if array.shape != (n, n):
            raise ValueError(f"[{index}] bad shape {array.shape} in {out_path}, expected ({n}, {n})")

        return index, out_path, time_elapsed, stdout, stderr

    except asyncio.CancelledError:
        # ensuring that child is dead
        if os.name == "posix":
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError, OSError, PermissionError):
                    os.killpg(process.pid, signal.SIGKILL)
        else:
            if process.returncode is None:
                with contextlib.suppress(OSError, PermissionError):
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                await asyncio.sleep(0.2)
                if process.returncode is None:
                    with contextlib.suppress(OSError, PermissionError):
                        process.kill()

        raise


def run_heat_maps_parallel(
    program_path: str,
    cities: np.ndarray,  # (B, n, 2)
    out_dir: str,
    timeout: float,  # in seconds
    max_workers: tp.Optional[int] = None,
    capture_index: int = 0,
    python_executable: str = sys.executable,
) -> dict:
    """
    Runs batch_size instances in parallel (max_workers concurrently)
    On any failure: cancels the rest, kills children, re-raises the first error
    On success: returns dict with paths, times, captured stdout/stderr
    """

    create_dir(out_dir)

    B = cities.shape[0]
    assert cities.shape[2] == 2, "`cities` must be (B, n, 2)"

    out_paths = [f"{out_dir}/heat_map_{i:05d}.npy" for i in range(B)]

    async def _runner() -> tuple[list[float | None], tp.Optional[bytes], tp.Optional[bytes]]:
        sem = asyncio.Semaphore(_default_workers(max_workers))
        ordered_times: list[float | None] = [None] * B
        instance_stdout: tp.Optional[bytes] = None
        instance_stderr: tp.Optional[bytes] = None

        async def _guarded(i: int) -> None:
            nonlocal ordered_times, instance_stdout, instance_stderr

            async with sem:
                index, path, time_elapsed, stdout_raw, stderr_raw = await _run_one_inference(
                    i,
                    program_path,
                    python_executable,
                    cities[i],
                    out_paths[i],
                    timeout=timeout,
                )
                ordered_times[index] = time_elapsed

                if index == capture_index:
                    instance_stdout, instance_stderr = stdout_raw, stderr_raw

        try:
            tg = getattr(asyncio, "TaskGroup", None)

            if tg is None:
                # Python <3.11: manual sibling cancelling
                tasks = [asyncio.create_task(_guarded(i)) for i in range(B)]
                try:
                    await asyncio.gather(*tasks)
                except Exception:
                    for t in tasks:
                        t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise
            else:
                async with asyncio.TaskGroup() as group:
                    for i in range(B):
                        group.create_task(_guarded(i))
        except Exception:
            for out_path in out_paths:
                tmp = pathlib.Path(out_path + ".tmp")

                if tmp.exists():
                    with contextlib.suppress(OSError):
                        tmp.unlink()
            raise

        return ordered_times, instance_stdout, instance_stderr

    ordered_times, instance_stdout, instance_stderr = asyncio.run(_runner())

    # decoding only if present
    if isinstance(instance_stdout, (bytes, bytearray)):
        instance_stdout = instance_stdout.decode("utf-8", "ignore")
    if isinstance(instance_stderr, (bytes, bytearray)):
        instance_stderr = instance_stderr.decode("utf-8", "ignore")

    return {
        "heat_map_paths": out_paths,
        "time_elapsed": ordered_times,
        "instance_stdout": instance_stdout,
        "instance_stderr": instance_stderr,
    }
