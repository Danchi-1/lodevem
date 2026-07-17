"""
runner.py — Benchmark Orchestrator

What this file does:
    Loops over every combination of (model file × device profile),
    calls predict.py and measure.py for each combination,
    and collects all the results into one list.

The nested loop structure:
    for each model file the user provided:
        for each device profile:
            1. Predict latency with nn-Meter
            2. Measure memory with Docker
            3. Store the combined result

With 3 model files × 16 profiles = 48 benchmark runs.
Each Docker run takes ~15-30 seconds, so the full suite takes 10-20 minutes.
We show a progress bar so you know it's working.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from lodevem import profiles as profile_loader
from lodevem.measure import build_image, measure_memory
from lodevem.predict import load_model, predict_latency, get_model_size_mb
from lodevem.profiles import DeviceProfile

logger = logging.getLogger(__name__)


def run_benchmark(
    model_paths: list[str | Path],
    profile_ids: list[str] | None = None,
    tier: int | None = None,
    warmup_runs: int = 5,
    timed_runs: int = 50,
) -> list[dict]:
    """
    Run the full benchmark: all models × selected device profiles.

    Args:
        model_paths:  List of .pt model file paths. Each file is one row group.
        profile_ids:  If provided, only benchmark against these specific profile IDs.
        tier:         If provided, only benchmark against profiles in this tier (1/2/3).
                      If neither profile_ids nor tier is given, all 16 profiles are used.
        warmup_runs:  Warmup inference passes inside the container.
        timed_runs:   Timed inference passes inside the container.

    Returns:
        A list of result dicts — one per (model × profile) combination.
        Each dict has all the fields needed for the results table.
    """

    # --- Determine which profiles to benchmark against ---
    if profile_ids:
        device_profiles = [profile_loader.load_by_id(pid) for pid in profile_ids]
    elif tier is not None:
        device_profiles = profile_loader.load_tier(tier)
    else:
        device_profiles = profile_loader.load_all()

    logger.info(f"Benchmarking {len(model_paths)} model(s) against {len(device_profiles)} profile(s)")
    logger.info(f"Total runs: {len(model_paths) * len(device_profiles)}")

    # --- Build the Docker image once before starting ---
    # This is cached after the first run so it's fast on subsequent calls.
    logger.info("Ensuring Docker image is ready...")
    build_image()

    results = []
    total_runs = len(model_paths) * len(device_profiles)

    # --- Progress bar ---
    # Rich's Progress gives us a nice animated progress bar in the terminal.
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Benchmarking...", total=total_runs)

        for model_path in model_paths:
            model_path = Path(model_path)
            model_label = model_path.name   # e.g. "cocoa_int8.pt"
            model_size_mb = get_model_size_mb(model_path)

            # Load the model for nn-Meter (latency prediction only)
            # This happens on the host — not inside Docker.
            logger.info(f"Loading model for prediction: {model_label}")
            model = load_model(model_path)

            for profile in device_profiles:
                progress.update(
                    task,
                    description=f"{model_label} → {profile.name}"
                )

                # --- Step 1: Predict latency (nn-Meter, runs on host) ---
                try:
                    latency_result = predict_latency(model, profile)
                    predicted_latency_ms = latency_result["scaled_latency_ms"]
                    prediction_status = "ok"
                except Exception as e:
                    logger.warning(f"Latency prediction failed for {profile.id}: {e}")
                    predicted_latency_ms = None
                    prediction_status = f"error: {e}"

                # --- Step 2: Measure memory (Docker, runs in container) ---
                try:
                    mem_result = measure_memory(
                        model_path=model_path,
                        profile=profile,
                        warmup_runs=warmup_runs,
                        timed_runs=timed_runs,
                    )
                except Exception as e:
                    logger.error(f"Memory measurement failed for {profile.id}: {e}")
                    mem_result = {
                        "status": "error",
                        "fits_in_ram": False,
                        "peak_ram_mb": None,
                        "median_latency_ms": None,
                        "error": str(e),
                    }

                # --- Combine into one result record ---
                result = {
                    # Model info
                    "model_file":          model_label,
                    "model_size_mb":       model_size_mb,

                    # Device info
                    "device_id":           profile.id,
                    "device_name":         profile.name,
                    "tier":                profile.tier,
                    "tier_label":          profile.tier_label,
                    "core_type":           profile.core_type,
                    "ram_limit_mb":        profile.ram_mb,

                    # Latency (from nn-Meter)
                    "predicted_latency_ms": predicted_latency_ms,
                    "prediction_status":    prediction_status,

                    # Memory (from Docker)
                    "peak_ram_mb":          mem_result.get("peak_ram_mb"),
                    "fits_in_ram":          mem_result.get("fits_in_ram", False),
                    "measure_status":       mem_result.get("status"),
                    "error":                mem_result.get("error"),
                }

                results.append(result)
                progress.advance(task)

    return results
