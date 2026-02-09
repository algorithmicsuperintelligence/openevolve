import os
import sys
import time
import json
import signal
import psutil
import pathlib
import asyncio
import platform
import subprocess
import contextlib
import typing as tp

BASE_DIR = pathlib.Path(__file__).parents[1]
sys.path.append(str(BASE_DIR))

# local imports
from utils.utils import *


class RunnerError(Exception):
    """Base for runner errors"""


class RunnerTimeout(RunnerError):
    """Subprocess exceeded time limit"""


class RunnerExitCodeError(RunnerError):
    """Subprocess exited with non-zero code"""


class RunnerInternalError(RunnerError):
    """Unexpected internal failure while running subprocess"""


def compile_tsp_executable(
    dir_path: str,
    timeout: float = 120.0,
    cxx: str = "g++",
    extra_flags: tp.Optional[tp.Sequence[str]] = None,
) -> pathlib.Path:
    """
    Compile TSP.cpp in `dir_path`, creating an executable `dir_path/bin/runner`
    Raises:
      FileNotFoundError, TimeoutError, RuntimeError
    """
    source_path = f"{dir_path}/TSP.cpp"
    include_path = f"{dir_path}/include"
    output_dir_path = f"{dir_path}/bin"
    output_bin_path = f"{output_dir_path}/runner"

    if not is_file_exist(source_path):
        raise FileNotFoundError(f'Missing source file: "{source_path}"')
    if not is_file_exist(include_path):
        raise FileNotFoundError(f'Missing include directory: "{include_path}"')

    create_dir(output_dir_path)

    cmd = [
        cxx,
        "-std=gnu++17",
        "-O3",
        "-DNDEBUG",
        "-march=native",
        "-funroll-loops",
        "-ffast-math",
        "-I", "include",
        str(pathlib.Path(source_path).name),
        "-o", output_bin_path,
    ]

    sysname = platform.system().lower()
    if sysname == "linux":
        cmd += ["-lpthread", "-lm", "-ldl"]
    elif sysname == "darwin":
        cmd += ["-lpthread", "-lm"]

    if extra_flags:
        cmd += list(extra_flags)

    popen_kwargs: dict[str, tp.Any] = dict(
        cwd=str(dir_path),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    else:
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    start_time = time.monotonic()
    try:
        process = subprocess.Popen(cmd, **popen_kwargs)
    except FileNotFoundError:
        # g++ missing
        raise

    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        finally:
            with contextlib.suppress(Exception):
                process.wait()
        raise TimeoutError(f"Compilation timed out after {timeout:.2f}s")

    elapsed = time.monotonic() - start_time

    if process.returncode != 0:
        raise RuntimeError(
            f"Compilation failed with exit code {process.returncode} in {elapsed:.3f}s.\n{stderr}"
        )

    if not is_file_exist(output_bin_path):
        raise RuntimeError(f'Compilation reported success, but binary not found: "{output_bin_path}"')

    return pathlib.Path(output_bin_path)


def _available_ram_gb() -> float:
    return psutil.virtual_memory().available / 1e9


def _estimate_mem_gb_for_config(
    config_file_path: str,
    default_dtype_bytes: int = 4,
    alpha: float = 2.0,
    beta_gb: float = 0.5,
) -> float:
    config = json.loads(read_file(config_file_path))
    n = config["cities_number"]
    bytes_required = n * n * default_dtype_bytes
    return alpha * (bytes_required / 1e9) + beta_gb


def _decide_workers(
    config_paths: list[str],
    max_workers: int | None = None,
    mem_per_proc_gb: float | None = None,
    alpha: float = 2.0,
    beta_gb: float = 0.5,
    safety_margin_gb: float = 2.0,
) -> tuple[int, float]:
    cpu_bound = max(1, get_cpu_cores_number() or 1)

    if mem_per_proc_gb is None or mem_per_proc_gb <= 0:
        ests: list[float] = []

        for config_path in config_paths:
            ests.append(_estimate_mem_gb_for_config(config_path, alpha=alpha, beta_gb=beta_gb))

        mem_per_proc_gb = max([e for e in ests if e > 0] or [1.0])

    avail = max(0.0, _available_ram_gb() - safety_margin_gb)
    mem_bound = max(1, int(avail // max(mem_per_proc_gb, 0.5)))

    if max_workers and max_workers > 0:
        return (max(1, min(cpu_bound, mem_bound, max_workers))), mem_per_proc_gb

    return (max(1, min(cpu_bound, mem_bound))), mem_per_proc_gb


async def _terminate_process(proc: asyncio.subprocess.Process) -> None:
    """
    Kill process (and its group on POSIX) and await .wait()
    """
    if proc.returncode is not None:
        return

    try:
        if os.name == "posix":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
        else:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
                await asyncio.sleep(0.2)
                proc.kill()
    finally:
        with contextlib.suppress(Exception):
            await proc.wait()


async def _run_one(
    index: int,
    runner_path: str,
    config_path: str,
    timeout: float | None,
    capture_stdout: bool,
    env_overrides: dict[str, str] | None,
) -> tuple[int, int, float, str | None, str]:
    """
    On success: (idx, returncode, elapsed, stdout_or_None, stderr)
    On timeout: raise RunnerTimeout
    On non-zero: raise RunnerExitCodeError
    """
    base_env = os.environ.copy()
    base_env.setdefault("OMP_NUM_THREADS", "1")
    base_env.setdefault("MKL_NUM_THREADS", "1")
    base_env.setdefault("OPENBLAS_NUM_THREADS", "1")
    base_env.setdefault("NUMEXPR_NUM_THREADS", "1")
    base_env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    if env_overrides:
        base_env.update(env_overrides)

    start_new_session = False
    creationflags = 0
    if os.name == "posix":
        start_new_session = True
    else:
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    stdout_target = asyncio.subprocess.PIPE if capture_stdout else asyncio.subprocess.DEVNULL
    stderr_target = asyncio.subprocess.PIPE

    proc = await asyncio.create_subprocess_exec(
        runner_path,
        config_path,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=stdout_target,
        stderr=stderr_target,
        start_new_session=start_new_session,
        creationflags=creationflags,
        env=base_env,
    )

    start_time = time.monotonic()

    try:
        if not timeout or timeout <= 0:
            stdout, stderr = await proc.communicate()
        else:
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                await _terminate_process(proc)
                raise RunnerTimeout(
                    f"[{index}] timed out after {timeout}s on {config_path}"
                )

        rc = proc.returncode
        elapsed = time.monotonic() - start_time

        out_s = stdout.decode("utf-8", "replace") if (stdout is not None and capture_stdout) else None
        err_s = stderr.decode("utf-8", "replace") if stderr is not None else ""

        if rc != 0:
            raise RunnerExitCodeError(
                f"[{index}] non-zero exit code {rc} for '{config_path}'\n"
                f"stderr (first 4000):\n{err_s[:4000]}"
            )

        return index, rc, elapsed, out_s, err_s

    except asyncio.CancelledError:
        # sibling failed: killing this process too
        await _terminate_process(proc)
        raise

    except RunnerError:
        # already cleaned in timeout branch
        raise

    except Exception as exc:
        await _terminate_process(proc)
        raise RunnerInternalError(
            f"[{index}] internal error on '{config_path}': {exc!r}"
        ) from exc


def run_runner_parallel(
    runner_path: str,
    config_paths: tp.Sequence[str],
    timeout: float | None = None,
    max_workers: int | None = 12,
    mem_per_proc_gb: float | None = None,
    alpha: float = 2.0,
    beta_gb: float = 0.5,
    safety_margin_gb: float = 2.0,
    capture_index: int = 0,
    env_overrides: dict[str, str] | None = None,
) -> dict:
    """
    Run all configs in parallel
    On first failure raises (timeout, exit or internal)
    Evaluator's try/except will handle it
    """

    used_workers, chosen_mem = _decide_workers(
        list(config_paths),
        max_workers=max_workers,
        mem_per_proc_gb=mem_per_proc_gb,
        alpha=alpha,
        beta_gb=beta_gb,
        safety_margin_gb=safety_margin_gb,
    )

    async def _driver() -> dict:
        sem = asyncio.Semaphore(used_workers)
        results: list[tuple[int, int, float, str | None, str] | None] = [None] * len(config_paths)
        first_out = ""
        first_err = ""

        async def _one(i: int):
            async with sem:
                capture = (i == capture_index)
                res = await _run_one(
                    i,
                    runner_path,
                    config_paths[i],
                    timeout,
                    capture,
                    env_overrides,
                )
                results[i] = res

        # TaskGroup will cancel siblings on first exception and _run_one will kill its proc
        async with asyncio.TaskGroup() as tg:
            for i in range(len(config_paths)):
                tg.create_task(_one(i))

        exit_codes: list[int] = []
        elapsed: list[float] = []

        for rec in results:
            assert rec is not None, "internal error: missing task result"

            i, rc, t, out_s, err_s = rec
            exit_codes.append(rc)
            elapsed.append(t)
            if i == capture_index:
                first_out = out_s or ""
                first_err = err_s or ""

        return {
            "exit_codes": exit_codes,
            "time_elapsed": elapsed,
            "instance_stdout": first_out,
            "instance_stderr": first_err,
            "used_workers": used_workers,
            "mem_per_proc_gb": chosen_mem,
        }

    # asyncio.run will close the loop after _driver finishes,
    # and by that time we have awaited all process.wait() in error paths
    return asyncio.run(_driver())
