"""
onnx.py — ONNX Model Footprint Analyzer

Analyzes ONNX (.onnx) models:
- Model initializers (weights/biases) parameter counts and precision
- Weight memory footprint
- Input/output graph signatures
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional

from lodevem.footprint.schema import ModelFootprint
from lodevem.guards import detect_host_gpus

logger = logging.getLogger(__name__)

# Map ONNX TensorProto data_type enum to friendly names and byte sizes
ONNX_TYPE_MAP: Dict[int, tuple[str, int]] = {
    1: ("FP32", 4),
    2: ("UINT8", 1),
    3: ("INT8", 1),
    4: ("UINT16", 2),
    5: ("INT16", 2),
    6: ("INT32", 4),
    7: ("INT64", 8),
    8: ("STRING", 0),
    9: ("BOOL", 1),
    10: ("FP16", 2),
    11: ("FP64", 8),
    12: ("UINT32", 4),
    13: ("UINT64", 8),
    14: ("COMPLEX64", 8),
    15: ("COMPLEX128", 16),
    16: ("BF16", 2),
}


def analyze_onnx_footprint(
    model_path: str | Path,
    input_shape: tuple[int, ...] | None = None,
    use_gpu: bool = False,
    allow_untrusted: bool = False,
) -> ModelFootprint:
    """
    Analyze an ONNX model file and return its ModelFootprint.
    """
    path = Path(model_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"ONNX model file not found: {path}")

    from lodevem.security import validate_onnx_security
    validate_onnx_security(path)

    file_size_bytes = path.stat().st_size
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

    total_params = 0
    param_bytes = 0
    precision_breakdown: Dict[str, int] = {}
    warnings: list[str] = []
    unavailable_metrics: list[str] = []
    metrics_status: Dict[str, str] = {}

    try:
        import onnx

        model_proto = onnx.load(str(path), load_external_data=False)
        graph = model_proto.graph

        for init in graph.initializer:
            dims = list(init.dims)
            numel = math.prod(dims) if dims else 1
            total_params += numel

            type_info = ONNX_TYPE_MAP.get(init.data_type, ("UNKNOWN", 4))
            dtype_name, byte_size = type_info

            param_bytes += numel * byte_size
            precision_breakdown[dtype_name] = precision_breakdown.get(dtype_name, 0) + numel

        metrics_status["parameters"] = "exact"
        metrics_status["weights_memory"] = "exact"
        metrics_status["precision"] = "exact"
    except ImportError:
        # Fallback to onnxruntime if onnx package is not installed
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            warnings.append(
                "'onnx' package not installed; loaded via onnxruntime without initializer tensor inspection."
            )
            unavailable_metrics.extend(["parameters", "precision_breakdown"])
            metrics_status["parameters"] = "unsupported"
            metrics_status["weights_memory"] = "unsupported"
            metrics_status["precision"] = "unsupported"
        except Exception as ort_err:
            raise RuntimeError(f"Failed to inspect ONNX model: {ort_err}") from ort_err
    except Exception as e:
        raise RuntimeError(f"Failed to read ONNX proto from '{path}': {e}") from e

    param_memory_mb = round(param_bytes / (1024 * 1024), 2) if param_bytes else None

    # FLOPs estimation for ONNX
    flops_status = "unsupported_operators"
    metrics_status["flops"] = "unsupported"
    warnings.append("Static FLOPs counting across arbitrary ONNX graphs is unsupported.")

    estimated_minimum_ram_mb = (
        round((param_memory_mb * 1.3) + 30.0, 1) if param_memory_mb else file_size_mb + 30.0
    )

    host_gpus = detect_host_gpus()

    return ModelFootprint(
        model_path=str(path),
        model_name=path.name,
        backend="onnx",
        file_size_bytes=file_size_bytes,
        file_size_mb=file_size_mb,
        input_shape=input_shape,
        total_parameters=total_params if total_params > 0 else None,
        trainable_parameters=0,  # ONNX models are deployed for inference
        parameter_memory_bytes=param_bytes if param_bytes > 0 else None,
        parameter_memory_mb=param_memory_mb,
        precision_breakdown=precision_breakdown,
        flops=None,
        macs=None,
        flops_status=flops_status,
        estimated_minimum_ram_mb=estimated_minimum_ram_mb,
        gpu_diagnostic_mode=use_gpu,
        host_gpus_detected=host_gpus,
        metrics_status=metrics_status,
        unavailable_metrics=unavailable_metrics,
        warnings=warnings,
    )
