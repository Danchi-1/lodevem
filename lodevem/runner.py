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
from lodevem.measure import build_image, measure_memory, _docker_available
from lodevem.predict import predict_latency
from lodevem.preflight import PreFlightAnalyzer
from lodevem.profiles import DeviceProfile
from lodevem.backends import get_backend

logger = logging.getLogger(__name__)


def _create_result_record(
    model_label: str,
    model_size_mb: float,
    profile: DeviceProfile,
    preflight_status: str,
    preflight_reason: str,
    estimated_memory_mb: float,
    prediction_status: str | None = None,
    predicted_latency_ms: float | None = None,
    measure_status: str | None = None,
    fits_in_ram: bool = False,
    peak_ram_mb: float | None = None,
    measured_latency_ms: float | None = None,
    measured_p95_ms: float | None = None,
    median_ttft_ms: float | None = None,
    median_tps: float | None = None,
    decode_time_ms: float | None = None,
    generated_tokens: int | None = None,
    prompt_length: int | None = None,
    error: str | None = None,
) -> dict:
    return {
        "model_file": model_label,
        "model_size_mb": model_size_mb,
        "device_id": profile.id,
        "device_name": profile.name,
        "tier": profile.tier,
        "tier_label": profile.tier_label,
        "core_type": profile.core_type,
        "ram_limit_mb": profile.ram_mb,
        "preflight_status": preflight_status,
        "preflight_reason": preflight_reason,
        "estimated_memory_mb": estimated_memory_mb,
        "predicted_latency_ms": predicted_latency_ms,
        "prediction_status": prediction_status,
        "peak_ram_mb": peak_ram_mb,
        "measured_latency_ms": measured_latency_ms,
        "measured_p95_ms": measured_p95_ms,
        "median_ttft_ms": median_ttft_ms,
        "median_tps": median_tps,
        "decode_time_ms": decode_time_ms,
        "generated_tokens": generated_tokens,
        "prompt_length": prompt_length,
        "fits_in_ram": fits_in_ram,
        "measure_status": measure_status,
        "error": error,
    }


def run_benchmark(
    model_paths: list[str | Path],
    profile_ids: list[str] | None = None,
    tier: int | None = None,
    warmup_runs: int = 5,
    timed_runs: int = 50,
    simulate_throttling: bool = False,
    no_predict: bool = False,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    prompt: str | None = None,
    max_new_tokens: int = 20,
    allow_untrusted: bool = False,
) -> list[dict]:
    """
    Run the full benchmark: all models × selected device profiles.

    Args:
        model_paths:  List of model file paths. Each file is one row group.
        profile_ids:  If provided, only benchmark against these specific profile IDs.
        tier:         If provided, only benchmark against profiles in this tier (1/2/3).
                      If neither profile_ids nor tier is given, all 16 profiles are used.
        warmup_runs:  Warmup inference passes inside the container.
        timed_runs:   Timed inference passes inside the container.
        allow_untrusted: If True, explicitly allow loading untrusted models on the host.

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

    # --- Build Docker image if Docker is available ---
    # Only runs in full mode. In lite mode (Kaggle, no Docker), this is skipped.
    if _docker_available():
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

            # --- Step 0: Pre-Flight Analysis (all profiles, before backend loading) ---
            # Run preflight against every profile first. Only load the backend
            # if at least one profile survives.
            preflight_results = {}
            any_runnable = False
            for profile in device_profiles:
                pf = PreFlightAnalyzer.analyze(model_path, profile.ram_mb)
                preflight_results[profile.id] = pf
                if pf["preflight_status"] != "preflight_oom":
                    any_runnable = True

            if not any_runnable:
                # Every profile failed preflight — emit OOM records, skip backend loading
                for profile in device_profiles:
                    pf = preflight_results[profile.id]
                    results.append(_create_result_record(
                        model_label=model_label,
                        model_size_mb=pf["model_size_mb"],
                        profile=profile,
                        preflight_status=pf["preflight_status"],
                        preflight_reason=pf["preflight_reason"],
                        estimated_memory_mb=pf["estimated_memory_mb"],
                        measure_status="skipped (pre-flight oom)",
                        fits_in_ram=False,
                        error="Skipped due to pre-flight OOM estimate.",
                    ))
                    progress.advance(task)
                continue

            # --- Load the model backend adapter (for latency prediction on host) ---
            # Unsafe model formats (pickle-based .pt, .joblib, TorchScript) MUST NEVER be loaded
            # on the host without explicit acknowledgment. If Docker is available, skip host prediction;
            # if running in Lite mode without acknowledgment, raise SecurityPolicyError.
            backend = None
            from lodevem.security import is_host_safe_format, SecurityPolicyError
            is_safe = is_host_safe_format(model_path)

            if not no_predict:
                if not is_safe and not allow_untrusted:
                    if _docker_available():
                        logger.info(
                            f"Skipping host latency prediction for untrusted format '{model_label}'. "
                            "Benchmark will execute safely within isolated Docker container."
                        )
                    else:
                        raise SecurityPolicyError(
                            f"Security Policy Violation: Model '{model_label}' uses an unsafe or unverified format.\n"
                            "Loading this model in Lite Mode executes arbitrary code on your host machine without container sandboxing.\n\n"
                            "Remediation Options:\n"
                            "  1. Run with Docker for automated sandbox confinement (recommended).\n"
                            "  2. Convert model to ONNX or Safetensors (safe, pickle-free formats).\n"
                            "  3. If you completely trust this model file, re-run with: --dangerously-allow-untrusted-model"
                        )
                else:
                    logger.info(f"Loading backend adapter for prediction: {model_label}")
                    try:
                        backend = get_backend(model_path, allow_untrusted=allow_untrusted)
                    except Exception as e:
                        logger.error(f"Failed to load backend for {model_label}: {e}")
                        pass

            for profile in device_profiles:
                progress.update(
                    task,
                    description=f"{model_label} → {profile.name}"
                )

                pf = preflight_results[profile.id]
                model_size_mb = pf["model_size_mb"]
                estimated_memory_mb = pf["estimated_memory_mb"]
                preflight_status = pf["preflight_status"]
                preflight_reason = pf["preflight_reason"]

                if preflight_status == "preflight_oom":
                    # Skip execution and return an immediate OOM record
                    results.append(_create_result_record(
                        model_label=model_label,
                        model_size_mb=model_size_mb,
                        profile=profile,
                        preflight_status=preflight_status,
                        preflight_reason=preflight_reason,
                        estimated_memory_mb=estimated_memory_mb,
                        measure_status="skipped (pre-flight oom)",
                        fits_in_ram=False,
                        error="Skipped due to pre-flight OOM estimate."
                    ))
                    progress.advance(task)
                    continue

                # --- Step 1: Predict latency (nn-Meter, runs on host) ---
                if no_predict or backend is None:
                    predicted_latency_ms = None
                    if not is_safe and not allow_untrusted:
                        prediction_status = "skipped (untrusted format on host)"
                    elif backend is None and not no_predict:
                        prediction_status = "skipped (no backend)"
                    else:
                        prediction_status = "skipped"
                elif backend.is_llm():
                    predicted_latency_ms = None
                    prediction_status = "unsupported (llm)"
                else:
                    try:
                        latency_result = predict_latency(backend, profile, input_shape=input_shape)
                        predicted_latency_ms = latency_result["scaled_latency_ms"]
                        prediction_status = "ok"
                    except Exception as e:
                        logger.warning(f"Latency prediction failed for {profile.id}: {e}")
                        predicted_latency_ms = None
                        prediction_status = f"error: {e}"

                # --- Step 2: Measure memory (Docker or Lite) ---
                try:
                    mem_result = measure_memory(
                        model_path=model_path,
                        profile=profile,
                        warmup_runs=warmup_runs,
                        timed_runs=timed_runs,
                        simulate_throttling=simulate_throttling,
                        input_shape=input_shape,
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        allow_untrusted=allow_untrusted,
                    )
                except Exception as e:
                    logger.error(f"Memory measurement failed for {profile.id}: {e}")
                    mem_result = {
                        "status": "error",
                        "fits_in_ram": False,
                        "peak_ram_mb": None,
                        "median_latency_ms": None,
                        "p95_latency_ms": None,
                        "error": str(e),
                    }

                # --- Combine into one result record ---
                result = _create_result_record(
                    model_label=model_label,
                    model_size_mb=model_size_mb,
                    profile=profile,
                    preflight_status=preflight_status,
                    preflight_reason=preflight_reason,
                    estimated_memory_mb=estimated_memory_mb,
                    prediction_status=prediction_status,
                    predicted_latency_ms=predicted_latency_ms,
                    measure_status=mem_result.get("status"),
                    fits_in_ram=mem_result.get("fits_in_ram", False),
                    peak_ram_mb=mem_result.get("peak_ram_mb"),
                    measured_latency_ms=mem_result.get("median_latency_ms"),
                    measured_p95_ms=mem_result.get("p95_latency_ms"),
                    median_ttft_ms=mem_result.get("median_ttft_ms"),
                    median_tps=mem_result.get("median_tps"),
                    decode_time_ms=mem_result.get("decode_time_ms"),
                    generated_tokens=mem_result.get("generated_tokens"),
                    prompt_length=mem_result.get("prompt_length"),
                    error=mem_result.get("error"),
                )

                results.append(result)
                progress.advance(task)

    return results

