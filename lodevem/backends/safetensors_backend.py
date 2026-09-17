"""
safetensors_backend.py — Backend adapter for Safetensors models (.safetensors).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from lodevem.backends.base import BenchmarkBackend

logger = logging.getLogger(__name__)


class SafetensorsBackend(BenchmarkBackend):
    """Backend adapter for Safetensors models."""

    def __init__(self, tensors: dict[str, Any], metadata: dict[str, Any]):
        self.tensors = tensors
        self.metadata = metadata

    @classmethod
    def supports(cls, path: str | Path) -> bool:
        return Path(path).suffix == ".safetensors"

    @classmethod
    def load(cls, path: str | Path, allow_untrusted: bool = False) -> BenchmarkBackend:
        from lodevem.security import validate_safetensors_security
        validate_safetensors_security(path)

        path = Path(path)
        raise TypeError(
            f"'{path.name}' is a raw Safetensors weight archive, not an executable model.\n"
            "Safetensors stores raw tensor weights without computational graph or forward execution logic.\n"
            f"  - To inspect parameters, memory, and precision: lodevem footprint {path.name}\n"
            "  - To benchmark inference: provide an ONNX model, TorchScript model, or a Hugging Face model directory."
        )

    def generate_inputs(self, shape: tuple[int, ...] | None, **kwargs) -> Any:
        import torch
        from lodevem.security import validate_input_shape
        target_shape = validate_input_shape(shape)
        return torch.randn(*target_shape)

    def execute(self, inputs: Any) -> Any:
        return inputs
