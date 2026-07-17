"""
measure.py — Docker-based Memory and Latency Measurement (Host Side)

What this file does:
    Spins up a Docker container with capped RAM and CPU,
    runs the benchmark inside it, reads the results, and returns them.

The division of labour:
    measure.py (this file)      — runs on YOUR machine, controls Docker
    container_measure.py        — runs INSIDE the container, measures the model

Why Docker?
    We cannot cap the RAM of just one Python function running in your
    terminal session — that would affect your whole shell. Docker lets us
    create a fully isolated process with a hard memory ceiling.

    When the container hits its memory limit:
    - The Linux kernel's OOM (Out Of Memory) killer terminates the process
    - Docker catches this and reports the exit code
    - We detect it and report "OOM" in the results table

cgroups v2:
    Docker's memory limits (--memory flag) use cgroups under the hood.
    cgroups v2 is the modern Linux kernel feature that enforces these limits
    at the hardware level — it's not just a suggestion.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from lodevem.profiles import DeviceProfile

logger = logging.getLogger(__name__)

# Name for our Docker image — built once, reused across all benchmarks
DOCKER_IMAGE_NAME = "lodevem-benchmark:latest"

# Path to the project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent


def _get_docker_client():
    """
    Get a Docker client. Raises a clear error if Docker isn't running.

    We import docker here (not at module level) so 'lodevem list'
    works even without Docker.
    """
    try:
        import docker
        client = docker.from_env()
        client.ping()  # Test connection — raises if Docker daemon isn't running
        return client
    except ImportError:
        raise ImportError("Run: pip install docker")
    except Exception:
        raise RuntimeError(
            "Cannot connect to Docker. Make sure Docker is running:\n"
            "  sudo systemctl start docker\n"
            "or open Docker Desktop if you're using it."
        )


def build_image(force_rebuild: bool = False) -> None:
    """
    Build the Docker image from our Dockerfile.

    This only needs to happen once — Docker caches the image.
    Subsequent runs are instant. We check if the image exists
    before building to avoid unnecessary waits.

    Args:
        force_rebuild: If True, rebuild even if the image already exists.
    """
    client = _get_docker_client()

    # Check if the image already exists
    existing = client.images.list(name=DOCKER_IMAGE_NAME)
    if existing and not force_rebuild:
        logger.info(f"Docker image '{DOCKER_IMAGE_NAME}' already exists. Skipping build.")
        return

    logger.info(f"Building Docker image '{DOCKER_IMAGE_NAME}'...")
    logger.info("(First build takes ~2 minutes — PyTorch is large. Subsequent runs are instant.)")

    client.images.build(
        path=str(PROJECT_ROOT),
        tag=DOCKER_IMAGE_NAME,
        rm=True,          # Remove intermediate containers after build
        forcerm=True,     # Remove intermediate containers even on failure
    )
    logger.info("Docker image built successfully.")


def measure_memory(
    model_path: str | Path,
    profile: DeviceProfile,
    warmup_runs: int = 5,
    timed_runs: int = 50,
) -> dict:
    """
    Run the model inside a RAM-capped Docker container and return measurements.

    Args:
        model_path:   Path to your .pt model file on the host machine.
        profile:      The device profile — determines the RAM cap and CPU count.
        warmup_runs:  Number of inference passes before timing starts.
        timed_runs:   Number of timed inference passes.

    Returns a dict:
        {
            "status":             "ok" | "oom" | "error",
            "peak_ram_mb":        float,    # Peak RAM used inside the container
            "median_latency_ms":  float,    # Median of timed inference runs
            "p95_latency_ms":     float,    # 95th percentile (for variance)
            "fits_in_ram":        bool,     # True if model loaded without OOM
            "ram_limit_mb":       int,      # What the limit was
            "error":              str,      # Only present if status != "ok"
        }
    """
    client = _get_docker_client()
    model_path = Path(model_path).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # The model file lives on your machine. We mount it into the container
    # at /models/<filename> so the container script can find it.
    container_model_path = f"/models/{model_path.name}"

    logger.info(
        f"  Running in container: {profile.name} "
        f"[RAM cap: {profile.ram_mb}MB, CPUs: {profile.cores}]"
    )

    try:
        # Start the container with constraints matching the device profile.
        # --memory:     Hard RAM limit. The kernel's OOM killer enforces this.
        # --memory-swap: Set equal to --memory to disable swap.
        #                (Real phones don't have swap space.)
        # --cpus:       Limit CPU time to simulate fewer/slower cores.
        #               Note: this limits CPU *quota*, not clock speed.
        result = client.containers.run(
            image=DOCKER_IMAGE_NAME,
            command=[
                container_model_path,
                str(warmup_runs),
                str(timed_runs),
            ],
            volumes={
                str(model_path.parent): {
                    "bind": "/models",
                    "mode": "ro",  # Read-only — container can't modify your model
                }
            },
            mem_limit=f"{profile.ram_mb}m",         # e.g. "1024m" = 1GB
            memswap_limit=f"{profile.ram_mb}m",      # Disables swap
            nano_cpus=int(profile.cores * 1e9),      # Docker uses nanocpus
            remove=True,             # Auto-remove container after it exits
            stdout=True,
            stderr=False,
        )

        # result is the raw stdout bytes from the container
        raw_output = result.decode("utf-8").strip()

        # Parse the JSON the container script printed
        data = json.loads(raw_output)

        # Add context about the RAM limit
        data["fits_in_ram"] = data.get("status") == "ok"
        data["ram_limit_mb"] = profile.ram_mb

        return data

    except Exception as e:
        error_str = str(e)

        # Docker raises an error when the container is OOM-killed.
        # The error message contains "137" (Linux OOM kill signal).
        if "137" in error_str or "OOMKilled" in error_str:
            return {
                "status": "oom",
                "fits_in_ram": False,
                "ram_limit_mb": profile.ram_mb,
                "peak_ram_mb": None,
                "median_latency_ms": None,
                "error": f"OOM: model was killed by the kernel (RAM limit: {profile.ram_mb}MB)",
            }

        return {
            "status": "error",
            "fits_in_ram": False,
            "ram_limit_mb": profile.ram_mb,
            "peak_ram_mb": None,
            "median_latency_ms": None,
            "error": error_str,
        }
