"""
lodevem.footprint — Model Footprint Profiler

Dispatches model footprint analysis across PyTorch, Hugging Face, ONNX, and Scikit-Learn backends,
with SHA-256 metadata caching and strict CPU guardrails.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from lodevem.footprint.cache import get_cached_footprint, save_cached_footprint
from lodevem.footprint.schema import ModelFootprint

logger = logging.getLogger(__name__)


def detect_backend_name(model_path: str | Path) -> str:
    """
    Determine the backend handler name from the model path.
    """
    path = Path(model_path)
    if path.is_dir() and (path / "config.json").exists():
        return "huggingface"
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        return "safetensors"
    if suffix in {".pt", ".pth"}:
        return "pytorch"
    if suffix == ".onnx":
        return "onnx"
    if suffix in {".pkl", ".joblib"}:
        return "sklearn"
    raise ValueError(f"Unsupported model format or extension: {path}")


def analyze_footprint(
    model_path: str | Path,
    input_shape: tuple[int, ...] | None = None,
    use_gpu: bool = False,
    use_cache: bool = True,
    allow_untrusted: bool = False,
) -> ModelFootprint:
    """
    Analyze model footprint and return a structured ModelFootprint.
    Uses disk caching by default based on file modification timestamp and size.
    """
    path = Path(model_path).resolve()
    backend_name = detect_backend_name(path)

    # 1. Check Cache
    if use_cache:
        cached = get_cached_footprint(path, backend_name, input_shape, use_gpu)
        if cached is not None:
            logger.debug(f"Footprint cache hit for {path}")
            return cached

    # 2. Run Analyzer
    if backend_name == "safetensors":
        from lodevem.footprint.analyzers.safetensors_analyzer import analyze_safetensors_footprint

        footprint = analyze_safetensors_footprint(path, input_shape, use_gpu, allow_untrusted=allow_untrusted)
    elif backend_name == "pytorch":
        from lodevem.footprint.analyzers.pytorch_analyzer import analyze_pytorch_footprint

        footprint = analyze_pytorch_footprint(path, input_shape, use_gpu, allow_untrusted=allow_untrusted)
    elif backend_name == "huggingface":
        from lodevem.footprint.analyzers.huggingface_analyzer import analyze_huggingface_footprint

        footprint = analyze_huggingface_footprint(path, input_shape, use_gpu, allow_untrusted=allow_untrusted)
    elif backend_name == "onnx":
        from lodevem.footprint.analyzers.onnx_analyzer import analyze_onnx_footprint

        footprint = analyze_onnx_footprint(path, input_shape, use_gpu, allow_untrusted=allow_untrusted)
    elif backend_name == "sklearn":
        from lodevem.footprint.analyzers.sklearn_analyzer import analyze_sklearn_footprint

        footprint = analyze_sklearn_footprint(path, input_shape, use_gpu, allow_untrusted=allow_untrusted)
    else:
        raise ValueError(f"Unknown backend '{backend_name}' for model {path}")

    # 3. Save Cache
    if use_cache:
        try:
            save_cached_footprint(footprint)
        except Exception as e:
            logger.debug(f"Failed to cache footprint: {e}")

    return footprint


__all__ = [
    "ModelFootprint",
    "analyze_footprint",
    "detect_backend_name",
]
