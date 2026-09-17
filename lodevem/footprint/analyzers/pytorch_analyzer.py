"""
pytorch.py — PyTorch Model Footprint Analyzer

Analyzes PyTorch (.pt, .pth) models:
- Total, trainable, and non-trainable parameters
- Exact precision breakdown by dtype (FP32, FP16, BF16, INT8, etc.)
- Model weights memory footprint (excluding framework runtime overhead)
- Operator-level FLOPs / MACs via torch.utils.flop_counter.FlopCounterMode
- Strict CPU verification
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from lodevem.footprint.schema import ModelFootprint
from lodevem.guards import assert_model_on_cpu, detect_host_gpus

logger = logging.getLogger(__name__)


def _format_dtype(dtype: Any) -> str:
    """Normalize torch dtype to a friendly display string."""
    s = str(dtype).replace("torch.", "")
    mapping = {
        "float32": "FP32",
        "float": "FP32",
        "float16": "FP16",
        "half": "FP16",
        "bfloat16": "BF16",
        "float64": "FP64",
        "double": "FP64",
        "int8": "INT8",
        "qint8": "INT8 (quant)",
        "quint8": "UINT8 (quant)",
        "int16": "INT16",
        "int32": "INT32",
        "int64": "INT64",
        "long": "INT64",
        "bool": "BOOL",
    }
    return mapping.get(s, s.upper())


def analyze_pytorch_footprint(
    model_path: str | Path,
    input_shape: tuple[int, ...] | None = None,
    use_gpu: bool = False,
    allow_untrusted: bool = False,
) -> ModelFootprint:
    """
    Analyze a PyTorch model file (.pt, .pth) and return its ModelFootprint.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError as e:
        raise ImportError(f"Missing PyTorch dependency: {e}") from e

    path = Path(model_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    file_size_bytes = path.stat().st_size
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

    # Scan pickle opcodes defensively before deserialization
    from lodevem.security import scan_pickle_security, validate_input_shape, SecurityPolicyError, is_in_docker
    scan_pickle_security(path)

    # 1. Check for TorchScript (Native C++ IR execution)
    is_torchscript = False
    model = None
    try:
        model = torch.jit.load(str(path), map_location="cpu")
        is_torchscript = True
    except Exception:
        pass

    if is_torchscript:
        if not allow_untrusted and not is_in_docker():
            raise SecurityPolicyError(
                f"Security Policy: TorchScript model '{path.name}' executes native PyTorch C++ bytecode.\n"
                "To analyze on host in Lite Mode, pass --dangerously-allow-untrusted-model or run inside Docker."
            )

    # 2. Non-TorchScript Checkpoint: Attempt weights_only=True first
    if not is_torchscript:
        try:
            model = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            if not allow_untrusted and not is_in_docker():
                raise SecurityPolicyError(
                    f"Security Policy Violation: Model '{path.name}' contains legacy pickled Python objects.\n"
                    "Analyzing this model in Lite Mode executes arbitrary Python bytecode on your host machine without container sandboxing.\n\n"
                    "Remediation Options:\n"
                    "  1. Run with Docker for automated sandbox confinement (recommended).\n"
                    "  2. Convert model to ONNX or Safetensors (safe, pickle-free formats).\n"
                    "  3. If you completely trust this model file, re-run with: --dangerously-allow-untrusted-model"
                )
            try:
                model = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as e:
                raise RuntimeError(f"Failed to load PyTorch model from '{path}': {e}") from e

    # Handle State Dict (safe weights-only dictionary)
    if isinstance(model, dict):
        total_params = 0
        param_bytes = 0
        precision_breakdown: Dict[str, int] = {}
        for k, v in model.items():
            if isinstance(v, torch.Tensor):
                numel = v.numel()
                total_params += numel
                element_size = v.element_size()
                param_bytes += numel * element_size
                dtype_str = _format_dtype(v.dtype)
                precision_breakdown[dtype_str] = precision_breakdown.get(dtype_str, 0) + numel

        param_memory_mb = round(param_bytes / (1024 * 1024), 2) if param_bytes else None
        estimated_minimum_ram_mb = (
            round((param_memory_mb * 1.3) + 30.0, 1) if param_memory_mb else file_size_mb + 30.0
        )
        host_gpus = detect_host_gpus()
        return ModelFootprint(
            model_path=str(path),
            model_name=path.name,
            backend="pytorch",
            file_size_bytes=file_size_bytes,
            file_size_mb=file_size_mb,
            input_shape=input_shape,
            total_parameters=total_params,
            trainable_parameters=0,
            parameter_memory_bytes=param_bytes,
            parameter_memory_mb=param_memory_mb,
            precision_breakdown=precision_breakdown,
            flops=None,
            macs=None,
            flops_status="state_dict_only",
            estimated_minimum_ram_mb=estimated_minimum_ram_mb,
            gpu_diagnostic_mode=use_gpu,
            host_gpus_detected=host_gpus,
            metrics_status={"parameters": "exact", "weights_memory": "exact", "precision": "exact", "flops": "not_applicable"},
            unavailable_metrics=[],
            warnings=["Model contains a state_dict rather than an executable nn.Module. FLOPs counting skipped."],
        )

    if not isinstance(model, nn.Module):
        raise TypeError(
            f"'{path.name}' contains {type(model).__name__}, not an nn.Module or state_dict.\n"
            "Only serialized nn.Module, state_dict, or TorchScript models can be analyzed."
        )

    # Assert model parameters strictly on CPU
    assert_model_on_cpu(model)
    model.eval()

    # Parameter & Precision Breakdown
    total_params = 0
    trainable_params = 0
    param_bytes = 0
    precision_breakdown: Dict[str, int] = {}

    for param in model.parameters():
        numel = param.numel()
        total_params += numel
        if param.requires_grad:
            trainable_params += numel

        element_size = param.element_size()
        param_bytes += numel * element_size

        dtype_str = _format_dtype(param.dtype)
        precision_breakdown[dtype_str] = precision_breakdown.get(dtype_str, 0) + numel

    # Also account for persistent buffers (e.g. BatchNorm running stats)
    for buf in model.buffers():
        numel = buf.numel()
        element_size = buf.element_size()
        param_bytes += numel * element_size
        dtype_str = _format_dtype(buf.dtype)
        precision_breakdown[dtype_str] = precision_breakdown.get(dtype_str, 0) + numel

    param_memory_mb = (
        round(param_bytes / (1024 * 1024), 4)
        if param_bytes < (1024 * 1024)
        else round(param_bytes / (1024 * 1024), 2)
    )

    # Input shape resolution
    target_shape = input_shape
    if target_shape is None:
        target_shape = getattr(model, "expected_input_shape", (1, 3, 224, 224))
    target_shape = validate_input_shape(target_shape)

    # FLOPs counting
    flops: Optional[int] = None
    macs: Optional[int] = None
    flops_status = "not_applicable"
    warnings: list[str] = []
    unavailable_metrics: list[str] = []
    metrics_status: Dict[str, str] = {
        "parameters": "exact",
        "weights_memory": "exact",
        "precision": "exact",
    }

    try:
        dummy_input = torch.randn(*target_shape)
        # Attempt FlopCounterMode (PyTorch 2.x feature)
        try:
            from torch.utils.flop_counter import FlopCounterMode

            with FlopCounterMode(display=False) as flop_counter:
                with torch.no_grad():
                    _ = model(dummy_input)

            total_flops = flop_counter.get_total_flops()
            if total_flops > 0:
                flops = int(total_flops)
                macs = flops // 2
                flops_status = "exact"
                metrics_status["flops"] = "exact"
            else:
                flops_status = "unsupported_operators"
                metrics_status["flops"] = "unsupported"
                warnings.append("FlopCounterMode returned 0 FLOPs for this architecture.")
        except ImportError:
            flops_status = "unsupported_operators"
            metrics_status["flops"] = "unsupported"
            warnings.append("torch.utils.flop_counter is not available in this PyTorch version.")
        except Exception as flop_err:
            flops_status = "unsupported_operators"
            metrics_status["flops"] = "unsupported"
            warnings.append(f"FLOPs counting failed on model forward pass: {flop_err}")
    except Exception as input_err:
        flops_status = "unsupported_operators"
        metrics_status["flops"] = "unsupported"
        warnings.append(f"Could not forward model with shape {target_shape}: {input_err}")

    # Estimated minimum RAM: parameter bytes + ~30% activation/workspace buffer + framework baseline
    estimated_minimum_ram_mb = round((param_memory_mb * 1.3) + 50.0, 1)

    # GPU diagnostic handling
    host_gpus = detect_host_gpus()
    gpu_peak_mb = None
    if use_gpu:
        if not host_gpus:
            warnings.append("GPU diagnostic mode requested, but no host GPU detected.")

    return ModelFootprint(
        model_path=str(path),
        model_name=path.name,
        backend="pytorch",
        file_size_bytes=file_size_bytes,
        file_size_mb=file_size_mb,
        input_shape=target_shape,
        total_parameters=total_params,
        trainable_parameters=trainable_params,
        parameter_memory_bytes=param_bytes,
        parameter_memory_mb=param_memory_mb,
        precision_breakdown=precision_breakdown,
        flops=flops,
        macs=macs,
        flops_status=flops_status,
        estimated_minimum_ram_mb=estimated_minimum_ram_mb,
        gpu_diagnostic_mode=use_gpu,
        host_gpus_detected=host_gpus,
        gpu_peak_memory_mb=gpu_peak_mb,
        metrics_status=metrics_status,
        unavailable_metrics=unavailable_metrics,
        warnings=warnings,
    )
