"""
measure.py — Memory Measurement (Two Modes)

lodevem supports two measurement modes:

  FULL MODE  (Docker available — e.g. your Linux machine)
      Spins up a RAM-capped Docker container. The kernel's OOM killer
      enforces the memory limit hard. Can detect true OOM failures.

  LITE MODE  (No Docker — e.g. Kaggle, Colab, Windows, macOS)
      Uses psutil to track the process's peak RAM during inference.
      Cannot enforce a memory cap, but measures real usage accurately.
      Reports whether the measured RAM fits within the device profile's limit.

Mode is selected automatically — if Docker is reachable, full mode is used.
If not, lite mode kicks in with a one-time notice.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from lodevem.profiles import DeviceProfile

logger = logging.getLogger(__name__)

DOCKER_IMAGE_NAME = "lodevem-benchmark:latest"
# Build context = the lodevem package directory (contains Dockerfile + container_measure.py)
# This works both when cloned from git AND when installed via pip
DOCKER_BUILD_CONTEXT = Path(__file__).parent


def _docker_available() -> bool:
    """Check if Docker is installed and the daemon is running."""
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


def measure_memory_lite(
    model_path: str | Path,
    profile: DeviceProfile,
    warmup_runs: int = 5,
    timed_runs: int = 50,
) -> dict:
    """
    Measure memory using psutil — no Docker required.

    Works on Kaggle, Colab, or any environment.

    What it does:
        Loads your model, runs inference N times, and tracks
        peak RAM usage of the current process using psutil.

    Limitation vs full mode:
        It cannot actually cap memory — so it won't kill your process
        if you exceed a device's RAM limit. Instead, it compares your
        measured peak RAM against the profile's limit and flags it.
    """
    try:
        import psutil
        import torch
    except ImportError as e:
        return {"status": "error", "error": str(e)}

    model_path = Path(model_path)
    process = psutil.Process(os.getpid())

    try:
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        model.eval()
    except Exception as e:
        return {"status": "error", "error": f"Failed to load model: {e}"}

    dummy_input = torch.zeros(1, 3, 224, 224)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    # Timed + memory measurement
    latencies_ms = []
    peak_ram_mb = 0.0

    with torch.no_grad():
        for _ in range(timed_runs):
            t_start = time.perf_counter()
            _ = model(dummy_input)
            t_end = time.perf_counter()

            ram_mb = process.memory_info().rss / (1024 * 1024)
            peak_ram_mb = max(peak_ram_mb, ram_mb)
            latencies_ms.append((t_end - t_start) * 1000)

    latencies_ms.sort()
    n = len(latencies_ms)

    fits = peak_ram_mb <= profile.ram_mb

    return {
        "status": "ok",
        "mode": "lite",
        "peak_ram_mb": round(peak_ram_mb, 2),
        "median_latency_ms": round(latencies_ms[n // 2], 2),
        "p95_latency_ms": round(latencies_ms[int(n * 0.95)], 2),
        "fits_in_ram": fits,
        "ram_limit_mb": profile.ram_mb,
    }


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
        path=str(DOCKER_BUILD_CONTEXT),
        tag=DOCKER_IMAGE_NAME,
        rm=True,
        forcerm=True,
    )
    logger.info("Docker image built successfully.")


_lite_mode_noticed = False   # print the notice only once per run


def measure_memory(
    model_path: str | Path,
    profile: DeviceProfile,
    warmup_runs: int = 5,
    timed_runs: int = 50,
) -> dict:
    """
    Measure peak RAM and latency — automatically chooses the right mode.

    If Docker is available: uses a RAM-capped container (full mode).
    If Docker is not available: uses psutil (lite mode).

    Both modes return the same dict shape so the rest of the tool
    doesn't need to know which mode was used.
    """
    global _lite_mode_noticed

    if not _docker_available():
        if not _lite_mode_noticed:
            logger.warning(
                "\n[lodevem] Docker not detected — running in LITE MODE.\n"
                "  Memory is measured via psutil (no hard RAM cap enforced).\n"
                "  'fits_in_ram' is based on measured vs profile limit.\n"
                "  For full OOM detection, run on a Linux machine with Docker.\n"
            )
            _lite_mode_noticed = True
        return measure_memory_lite(model_path, profile, warmup_runs, timed_runs)

    # --- Full Docker mode ---
    client = _get_docker_client()
    model_path = Path(model_path).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    container_model_path = f"/models/{model_path.name}"

    logger.info(
        f"  Running in container: {profile.name} "
        f"[RAM cap: {profile.ram_mb}MB, CPUs: {profile.cores}]"
    )

    try:
        result = client.containers.run(
            image=DOCKER_IMAGE_NAME,
            command=[container_model_path, str(warmup_runs), str(timed_runs)],
            volumes={
                str(model_path.parent): {"bind": "/models", "mode": "ro"}
            },
            mem_limit=f"{profile.ram_mb}m",
            memswap_limit=f"{profile.ram_mb}m",
            nano_cpus=int(profile.cores * 1e9),
            remove=True,
            stdout=True,
            stderr=False,
        )

        raw_output = result.decode("utf-8").strip()
        data = json.loads(raw_output)
        data["fits_in_ram"] = data.get("status") == "ok"
        data["ram_limit_mb"] = profile.ram_mb
        data["mode"] = "docker"
        return data

    except Exception as e:
        error_str = str(e)
        if "137" in error_str or "OOMKilled" in error_str:
            return {
                "status": "oom",
                "mode": "docker",
                "fits_in_ram": False,
                "ram_limit_mb": profile.ram_mb,
                "peak_ram_mb": None,
                "median_latency_ms": None,
                "error": f"OOM: model killed by kernel (RAM limit: {profile.ram_mb}MB)",
            }
        return {
            "status": "error",
            "mode": "docker",
            "fits_in_ram": False,
            "ram_limit_mb": profile.ram_mb,
            "peak_ram_mb": None,
            "median_latency_ms": None,
            "error": error_str,
        }
