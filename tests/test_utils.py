"""
Test utilities for OpenEvolve tests
Provides common functions and constants for consistent testing
"""

import os
import sys
import time
import subprocess
import requests
import socket
from typing import Optional, Tuple
from openai import OpenAI
from openevolve.config import Config, LLMModelConfig

# Standard test model for integration tests - small and fast.
# codelion/dhara-250m is a 250M tri-mode (AR + block-diffusion) model served via
# optillm's local inference. Note: on Apple Silicon its depthwise-conv layers hit
# a oneDNN multithreading pathology in torch.conv1d that makes generation ~40x
# slower; if running these integration tests locally on macOS, start the optillm
# server with OMP_NUM_THREADS=1 (Linux x86 CI is unaffected).
TEST_MODEL = "codelion/dhara-250m"
DEFAULT_PORT = 8000
DEFAULT_BASE_URL = f"http://localhost:{DEFAULT_PORT}/v1"


def find_free_port(start_port: int = 8000, max_tries: int = 100) -> int:
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + max_tries):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("localhost", port))
            sock.close()
            return port
        except OSError:
            continue
        finally:
            sock.close()
    raise RuntimeError(f"Could not find free port in range {start_port}-{start_port + max_tries}")


def setup_test_env():
    """Set up test environment with local inference"""
    os.environ["OPTILLM_API_KEY"] = "optillm"
    return TEST_MODEL


def get_test_client(base_url: str = DEFAULT_BASE_URL) -> OpenAI:
    """Get OpenAI client configured for local optillm"""
    return OpenAI(api_key="optillm", base_url=base_url)


def start_test_server(
    model: str = TEST_MODEL, port: Optional[int] = None
) -> Tuple[subprocess.Popen, int]:
    """
    Start optillm server for testing
    Returns tuple of (process_handle, actual_port_used)
    """
    if port is None:
        port = find_free_port()

    # Set environment for local inference
    env = os.environ.copy()
    env["OPTILLM_API_KEY"] = "optillm"

    # Pass HF_TOKEN if available (needed for model downloads in CI)
    if "HF_TOKEN" in os.environ:
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]

    print(f"Starting optillm server on port {port}...")

    # Start server (don't capture output to avoid pipe buffer deadlock)
    proc = subprocess.Popen(["optillm", "--model", model, "--port", str(port)], env=env)

    # Wait for server to start
    for i in range(30):
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                print(f"✅ optillm server started successfully on port {port}")
                return proc, port
        except Exception as e:
            if i < 5:  # Only print for first few attempts to avoid spam
                print(f"Attempt {i+1}: Waiting for server... ({e})")
            pass
        time.sleep(1)

    # Server didn't start in time - clean up
    error_msg = f"optillm server failed to start on port {port}"
    print(f"❌ {error_msg} - check that optillm is installed and model is available")

    # Clean up
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    raise RuntimeError(error_msg)


def stop_test_server(proc: subprocess.Popen):
    """Stop the test server"""
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def is_server_running(port: int = DEFAULT_PORT) -> bool:
    """Check if optillm server is running on the given port"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_integration_config(port: int = DEFAULT_PORT) -> Config:
    """Get config for integration tests with optillm"""
    config = Config()
    config.max_iterations = 5  # Very small for CI speed
    config.checkpoint_interval = 2
    config.database.in_memory = True
    config.evaluator.parallel_evaluations = 2
    config.evaluator.timeout = 10  # Short timeout for CI

    # Disable cascade evaluation to avoid warnings in simple test evaluators
    config.evaluator.cascade_evaluation = False

    # Set long timeout with no retries for integration tests
    config.llm.retries = 0  # No retries to fail fast
    config.llm.timeout = 120  # Long timeout to allow model to respond

    # Bound generation length. dhara-250m served via optillm does not stop at the
    # ChatML <|im_end|> turn marker (optillm uses the tokenizer's eos), so without a
    # cap it rambles to max_tokens on every call. The 4096 default meant single
    # generations of 10-20 minutes; 256 keeps each call fast while still leaving
    # room for a small code diff.
    config.llm.max_tokens = 256

    # Configure to use optillm server
    base_url = f"http://localhost:{port}/v1"
    config.llm.api_base = base_url
    config.llm.models = [
        LLMModelConfig(
            name=TEST_MODEL,
            api_key="optillm",
            api_base=base_url,
            weight=1.0,
            max_tokens=256,  # Bound generation length (see note above)
            timeout=120,  # Long timeout
            retries=0,  # No retries
        )
    ]

    return config


def get_simple_test_messages():
    """Get simple test messages for basic validation"""
    return [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a simple Python function that returns 'hello'."},
    ]


def get_evolution_test_program():
    """Get a simple program for evolution testing"""
    return """# EVOLVE-BLOCK-START
def solve(x):
    return x * 2
# EVOLVE-BLOCK-END
"""


def get_evolution_test_evaluator():
    """Get a simple evaluator for evolution testing"""
    return """def evaluate(program_path):
    return {"score": 0.5, "complexity": 10, "combined_score": 0.5}
"""
