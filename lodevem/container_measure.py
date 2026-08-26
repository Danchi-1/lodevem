"""
container_measure.py — Runs INSIDE the Docker container

What this script does:
    1. Loads the model
    2. Runs inference N times (warmup + timed runs)
    3. Measures peak RAM usage from /proc/self/status
    4. Prints results as JSON to stdout

Why JSON to stdout?
    The container's stdout is the only communication channel back to
    the host (measure.py). JSON is easy to parse and unambiguous.

Why /proc/self/status?
    This is a Linux kernel file that reports memory usage for the current
    process. VmRSS (Virtual Memory Resident Set Size) is the amount of
    RAM actually in physical memory — the most honest measure of what the
    device needs to hold in RAM.

This file has NO imports from the lodevem package —
it runs standalone inside the container where lodevem isn't installed.
"""

import json
import sys
import time
import traceback


def read_rss_kb() -> int:
    """
    Read current RSS (Resident Set Size) from /proc/self/status.

    RSS = the actual RAM this process is using right now.
    We read it from the kernel directly — no external library needed.
    Returns kilobytes.
    """
    with open("/proc/self/status", "r") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                # Line format: "VmRSS:    42316 kB"
                return int(line.split()[1])
    return 0


def run_benchmark(model_path: str, warmup_runs: int, timed_runs: int, input_shape: tuple[int, ...] = (1, 3, 224, 224)) -> dict:
    """
    Load the model and measure inference latency + peak memory.

    We track peak RSS across all inference calls, not just one,
    because PyTorch may allocate additional memory during inference
    (e.g. activation buffers, gradient buffers).
    """
    try:
        from lodevem.backends import get_backend
    except ImportError as e:
        return {"status": "error", "error": f"Missing dependency: {e}"}

    # --- Load model ---
    try:
        backend = get_backend(model_path)
    except Exception as e:
        return {"status": "error", "error": f"Failed to load backend: {e}"}

    # --- Prepare input ---
    try:
        dummy_input = backend.generate_inputs(input_shape)
    except Exception as e:
        return {"status": "error", "error": f"Failed to generate inputs: {e}"}

    # --- Warmup passes ---
    # The first few inference calls are slower because the backend may be
    # setting up internal buffers. Warmup runs let us measure steady state.
    try:
        for _ in range(warmup_runs):
            _ = backend.execute(dummy_input)
    except Exception as e:
        return {"status": "error", "error": f"Inference failed during warmup: {e}"}

    # --- Timed runs ---
    latencies_ms = []
    peak_rss_kb = 0

    try:
        for _ in range(timed_runs):
            rss_before = read_rss_kb()

            t_start = time.perf_counter()
            _ = backend.execute(dummy_input)
            t_end = time.perf_counter()

            rss_after = read_rss_kb()

            latencies_ms.append((t_end - t_start) * 1000)
            peak_rss_kb = max(peak_rss_kb, rss_after)
    except Exception as e:
        return {"status": "error", "error": f"Inference failed during timed runs: {e}"}

    # Sort latencies to compute percentiles
    latencies_ms.sort()
    n = len(latencies_ms)

    return {
        "status": "ok",
        "peak_ram_mb": round(peak_rss_kb / 1024, 2),
        "median_latency_ms": round(latencies_ms[n // 2], 2),
        "p95_latency_ms": round(latencies_ms[int(n * 0.95)], 2),
        "min_latency_ms": round(latencies_ms[0], 2),
        "max_latency_ms": round(latencies_ms[-1], 2),
        "timed_runs": timed_runs,
        "warmup_runs": warmup_runs,
    }


if __name__ == "__main__":
    # Arguments passed in by measure.py when it starts the container:
    # sys.argv[1] = path to the model file (inside the container)
    # sys.argv[2] = number of warmup runs
    # sys.argv[3] = number of timed runs

    if len(sys.argv) < 4:
        print(json.dumps({
            "status": "error",
            "error": "Usage: container_measure.py <model_path> <warmup_runs> <timed_runs>"
        }))
        sys.exit(1)

    model_path = sys.argv[1]
    warmup_runs = int(sys.argv[2])
    timed_runs = int(sys.argv[3])
    input_shape_str = sys.argv[4] if len(sys.argv) > 4 else "1,3,224,224"
    input_shape = tuple(map(int, input_shape_str.split(",")))

    try:
        result = run_benchmark(model_path, warmup_runs, timed_runs, input_shape)
    except MemoryError:
        # If the container hits the RAM cap, Python raises MemoryError.
        # We catch it here and report OOM cleanly rather than crashing.
        result = {
            "status": "oom",
            "error": "Out of memory — model could not be loaded within the RAM limit.",
        }
    except Exception as e:
        result = {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    # Print result as JSON — measure.py on the host reads this from stdout
    print(json.dumps(result))
