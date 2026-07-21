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
import subprocess
import sys
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


def _read_rss_kb() -> int:
    """Read Resident Set Size from /proc/self/status if available."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except FileNotFoundError:
        pass
    return 0


def _run_lite_subprocess(
    model_path: str | Path,
    profile: DeviceProfile,
    warmup_runs: int,
    timed_runs: int,
    simulate_throttling: bool = False,
) -> dict:
    """Run the lite measurement in an isolated subprocess."""
    cmd = [
        sys.executable,
        "-m",
        "lodevem.measure",
        "--lite-worker",
        str(model_path),
        str(profile.ram_mb),
        str(profile.cores),
        str(warmup_runs),
        str(timed_runs),
    ]
    if simulate_throttling:
        cmd.append("--simulate-throttling")

    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Lite worker failed: "
            f"returncode={completed.returncode}, stderr={completed.stderr.strip()}"
        )

    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Lite worker returned invalid JSON: {e}\nstdout={completed.stdout}\nstderr={completed.stderr}"
        )

    data["mode"] = "lite"
    data["ram_limit_mb"] = profile.ram_mb
    data["fits_in_ram"] = data.get("status") == "ok"
    return data


def measure_memory_lite(
    model_path: str | Path,
    profile: DeviceProfile,
    warmup_runs: int = 5,
    timed_runs: int = 50,
    simulate_throttling: bool = False,
) -> dict:
    """
    Measure memory using an isolated worker process — no Docker required.

    This is closer to Docker mode because:
      - the benchmark runs in a separate process,
      - it can enforce a memory cap via OS resource limits when supported,
      - it can limit PyTorch thread count to the profile CPU count.

    If the subprocess worker cannot be used, we fall back to in-process
    measurement.
    """
    try:
        return _run_lite_subprocess(model_path, profile, warmup_runs, timed_runs, simulate_throttling)
    except Exception as e:
        logger.warning(
            "Lite subprocess worker unavailable, falling back to in-process lite mode: %s",
            e,
        )

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
        throttle_multiplier = 1.0
        current_threads = profile.cores
        for i in range(1, timed_runs + 1):
            t_start = time.perf_counter()
            _ = model(dummy_input)
            t_end = time.perf_counter()

            # Apply simulated thermal throttling if requested
            elapsed_ms = (t_end - t_start) * 1000
            if simulate_throttling:
                # After every 10 passes, reduce threads by 1 and increase latency multiplier
                if i % 10 == 0:
                    current_threads = max(1, current_threads - 1)
                    try:
                        if hasattr(torch, "set_num_threads"):
                            torch.set_num_threads(current_threads)
                        if hasattr(torch, "set_num_interop_threads"):
                            torch.set_num_interop_threads(current_threads)
                    except Exception:
                        pass
                    throttle_multiplier *= 1.15
                elapsed_ms *= throttle_multiplier

            ram_mb = process.memory_info().rss / (1024 * 1024)
            peak_ram_mb = max(peak_ram_mb, ram_mb)
            latencies_ms.append(elapsed_ms)

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
    simulate_throttling: bool = False,
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
                "  Benchmark runs in an isolated subprocess when possible.\n"
                "  Local memory/thread limits are applied to approximate Docker behavior.\n"
                "  For exact container enforcement, run on a Linux machine with Docker.\n"
            )
            _lite_mode_noticed = True
        return measure_memory_lite(model_path, profile, warmup_runs, timed_runs, simulate_throttling)

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


def _set_memory_limit(ram_limit_mb: int) -> None:
    """Set a hard address-space limit if the platform supports it."""
    try:
        import resource
    except ImportError:
        return

    if hasattr(resource, "RLIMIT_AS"):
        limit = ram_limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def _lite_worker(
    model_path: str,
    ram_limit_mb: int,
    num_threads: int,
    warmup_runs: int,
    timed_runs: int,
    simulate_throttling: bool = False,
) -> None:
    import json
    import time
    import traceback

    try:
        import torch
    except ImportError as e:
        print(json.dumps({"status": "error", "error": str(e)}))
        sys.exit(1)

    _set_memory_limit(ram_limit_mb)

    if hasattr(torch, "set_num_threads"):
        torch.set_num_threads(num_threads)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(num_threads)

    try:
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        model.eval()
    except Exception as e:
        print(json.dumps({"status": "error", "error": f"Failed to load model: {e}"}))
        sys.exit(1)

    dummy_input = torch.zeros(1, 3, 224, 224)
    latencies_ms = []
    peak_rss_kb = 0

    try:
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy_input)

            # Thermal throttling simulation variables
            throttle_multiplier = 1.0
            current_threads = num_threads

            for i in range(1, timed_runs + 1):
                t_start = time.perf_counter()
                _ = model(dummy_input)
                t_end = time.perf_counter()

                # Apply throttling if requested via environment flag
                # Note: subprocess invocation will include the flag to trigger throttling
                lat_ms = (t_end - t_start) * 1000 * throttle_multiplier
                latencies_ms.append(lat_ms)

                rss = _read_rss_kb()
                peak_rss_kb = max(peak_rss_kb, rss)

                # Apply simulated thermal throttling when requested
                if simulate_throttling and i % 10 == 0:
                    current_threads = max(1, current_threads - 1)
                    try:
                        if hasattr(torch, "set_num_threads"):
                            torch.set_num_threads(current_threads)
                        if hasattr(torch, "set_num_interop_threads"):
                            torch.set_num_interop_threads(current_threads)
                    except Exception:
                        pass
                    throttle_multiplier *= 1.15
    except MemoryError:
        print(json.dumps({
            "status": "oom",
            "error": "Out of memory — model could not be loaded within the RAM limit.",
        }))
        sys.exit(0)

    latencies_ms.sort()
    n = len(latencies_ms)

    result = {
        "status": "ok",
        "peak_ram_mb": round(peak_rss_kb / 1024, 2),
        "median_latency_ms": round(latencies_ms[n // 2], 2),
        "p95_latency_ms": round(latencies_ms[int(n * 0.95)], 2),
        "min_latency_ms": round(latencies_ms[0], 2),
        "max_latency_ms": round(latencies_ms[-1], 2),
        "timed_runs": timed_runs,
        "warmup_runs": warmup_runs,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lite worker for lodevem measurement")
    parser.add_argument("--lite-worker", action="store_true")
    parser.add_argument("--simulate-throttling", action="store_true")
    parser.add_argument("model_path", nargs="?", help="Path to the model file")
    parser.add_argument("ram_limit_mb", nargs="?", type=int, help="RAM limit in MB")
    parser.add_argument("num_threads", nargs="?", type=int, help="Number of CPU threads")
    parser.add_argument("warmup_runs", nargs="?", type=int, help="Warmup run count")
    parser.add_argument("timed_runs", nargs="?", type=int, help="Timed run count")
    args = parser.parse_args()

    if args.lite_worker:
        if not args.model_path or args.ram_limit_mb is None or args.num_threads is None or args.warmup_runs is None or args.timed_runs is None:
            print(json.dumps({"status": "error", "error": "Missing lite worker arguments"}))
            sys.exit(1)
        _lite_worker(
            args.model_path,
            args.ram_limit_mb,
            args.num_threads,
            args.warmup_runs,
            args.timed_runs,
            simulate_throttling=args.simulate_throttling,
        )
    else:
        parser.print_help()
