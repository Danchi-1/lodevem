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


def run_benchmark(model_path: str, warmup_runs: int, timed_runs: int, input_shape: tuple[int, ...], prompt: str | None = None, max_new_tokens: int = 20) -> dict:
    try:
        from lodevem.backends import get_backend
    except ImportError as e:
        return {"status": "error", "error": f"Missing dependency: {e}"}

    try:
        backend = get_backend(model_path)
    except Exception as e:
        return {"status": "error", "error": f"Failed to load backend: {e}"}

    try:
        dummy_input = backend.generate_inputs(input_shape, prompt=prompt)
    except Exception as e:
        return {"status": "error", "error": f"Failed to generate inputs: {e}"}

    try:
        for _ in range(warmup_runs):
            if backend.is_llm():
                _ = backend.execute_llm(dummy_input, max_new_tokens)
            else:
                _ = backend.execute(dummy_input)
    except Exception as e:
        return {"status": "error", "error": f"Inference failed during warmup: {e}"}

    if backend.is_llm():
        results_list = []
        try:
            for _ in range(timed_runs):
                res = backend.execute_llm(dummy_input, max_new_tokens)
                res["peak_ram_mb"] = max(res["peak_ram_mb"], read_rss_kb() / 1024.0)
                results_list.append(res)
        except Exception as e:
            return {"status": "error", "error": f"LLM inference failed: {e}"}
            
        results_list.sort(key=lambda x: x["ttft_ms"])
        n = len(results_list)
        mid = results_list[n // 2]
        peak_ram_mb = max(r["peak_ram_mb"] for r in results_list)
        
        return {
            "status": "ok",
            "peak_ram_mb": round(peak_ram_mb, 2),
            "median_latency_ms": None,
            "median_ttft_ms": round(mid["ttft_ms"], 2),
            "median_tps": round(mid["tps"], 2) if mid["tps"] else None,
            "generated_tokens": mid["generated_tokens"],
            "prompt_length": mid["prompt_length"],
            "timed_runs": timed_runs,
            "warmup_runs": warmup_runs,
        }

    # Standard Timed runs
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("warmup_runs", type=int)
    parser.add_argument("timed_runs", type=int)
    parser.add_argument("input_shape_str", nargs="?", default="1,3,224,224")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    args = parser.parse_args()

    input_shape = tuple(map(int, args.input_shape_str.split(",")))

    try:
        result = run_benchmark(args.model_path, args.warmup_runs, args.timed_runs, input_shape, args.prompt, args.max_new_tokens)
    except MemoryError:
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

    print(json.dumps(result))
