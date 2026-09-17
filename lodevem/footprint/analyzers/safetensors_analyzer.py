"""
safetensors_analyzer.py — Safetensors Model Footprint Analyzer

Statically analyzes Safetensors (.safetensors) models:
- Validates file integrity and offsets using validate_safetensors_security()
- Computes exact total parameters, precision breakdown, and weights memory
- Purely static parsing without code execution or full tensor loading
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional

from lodevem.footprint.schema import ModelFootprint
from lodevem.guards import detect_host_gpus
from lodevem.security import validate_safetensors_security

logger = logging.getLogger(__name__)

DTYPE_MAP: Dict[str, tuple[str, int]] = {
    "F32": ("FP32", 4),
    "F16": ("FP16", 2),
    "BF16": ("BF16", 2),
    "F64": ("FP64", 8),
    "I64": ("INT64", 8),
    "I32": ("INT32", 4),
    "I16": ("INT16", 2),
    "I8": ("INT8", 1),
    "U8": ("UINT8", 1),
    "BOOL": ("BOOL", 1),
}


def analyze_safetensors_footprint(
    model_path: str | Path,
    input_shape: tuple[int, ...] | None = None,
    use_gpu: bool = False,
    allow_untrusted: bool = False,
) -> ModelFootprint:
    """
    Analyze a Safetensors model file and return its ModelFootprint.
    """
    path = Path(model_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Safetensors file not found: {path}")

    file_size_bytes = path.stat().st_size
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

    # Perform strict static security and format validation on header
    header = validate_safetensors_security(path)

    total_params = 0
    param_bytes = 0
    precision_breakdown: Dict[str, int] = {}
    warnings: list[str] = []
    metrics_status: Dict[str, str] = {}

    for key, info in header.items():
        if key == "__metadata__":
            continue
        shape = info.get("shape", [])
        numel = math.prod(shape) if shape else 1
        total_params += numel

        raw_dtype = info.get("dtype", "UNKNOWN")
        friendly_dtype, byte_size = DTYPE_MAP.get(raw_dtype, (raw_dtype, 2))

        offsets = info.get("data_offsets", [0, 0])
        tensor_bytes = offsets[1] - offsets[0] if len(offsets) == 2 else numel * byte_size
        param_bytes += tensor_bytes

        precision_breakdown[friendly_dtype] = precision_breakdown.get(friendly_dtype, 0) + numel

    metrics_status["parameters"] = "exact"
    metrics_status["weights_memory"] = "exact"
    metrics_status["precision"] = "exact"
    metrics_status["flops"] = "not_applicable"

    param_memory_mb = round(param_bytes / (1024 * 1024), 2) if param_bytes else None
    estimated_minimum_ram_mb = (
        round((param_memory_mb * 1.2) + 20.0, 1) if param_memory_mb else file_size_mb + 20.0
    )

    host_gpus = detect_host_gpus()

    return ModelFootprint(
        model_path=str(path),
        model_name=path.name,
        backend="safetensors",
        file_size_bytes=file_size_bytes,
        file_size_mb=file_size_mb,
        input_shape=input_shape,
        total_parameters=total_params if total_params > 0 else None,
        trainable_parameters=0,
        parameter_memory_bytes=param_bytes if param_bytes > 0 else None,
        parameter_memory_mb=param_memory_mb,
        precision_breakdown=precision_breakdown,
        flops=None,
        macs=None,
        flops_status="not_applicable",
        estimated_minimum_ram_mb=estimated_minimum_ram_mb,
        gpu_diagnostic_mode=use_gpu,
        host_gpus_detected=host_gpus,
        metrics_status=metrics_status,
        unavailable_metrics=[],
        warnings=warnings,
    )
