from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from lodevem.backends.base import BenchmarkBackend
from lodevem.security import is_in_docker

logger = logging.getLogger(__name__)


class PyTorchBackend(BenchmarkBackend):
    """Backend adapter for PyTorch models (.pt, .pth)."""

    def __init__(self, model: Any):
        self.model = model

    @classmethod
    def supports(cls, path: str | Path) -> bool:
        path = Path(path)
        return path.suffix in {".pt", ".pth"}

    @classmethod
    def load(cls, path: str | Path, allow_untrusted: bool = False) -> BenchmarkBackend:
        try:
            import torch
            import torch.nn as nn
        except ImportError as e:
            raise ImportError(f"Missing dependency. Run: pip install 'lodevem[pytorch]'. Details: {e}") from e

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: '{path}'")

        from lodevem.security import scan_pickle_security, SecurityPolicyError
        scan_pickle_security(path)

        # 1. Check for TorchScript (Native C++ IR execution)
        is_torchscript = False
        try:
            model = torch.jit.load(str(path), map_location="cpu")
            is_torchscript = True
        except Exception:
            pass

        if is_torchscript:
            if not allow_untrusted and not is_in_docker():
                raise SecurityPolicyError(
                    f"Security Policy: TorchScript model '{path.name}' executes native PyTorch C++ bytecode.\n"
                    "To execute on host in Lite Mode, pass --dangerously-allow-untrusted-model or run inside Docker."
                )
            from lodevem.guards import assert_model_on_cpu
            assert_model_on_cpu(model)
            return cls(model)

        # 2. Non-TorchScript Checkpoint: Attempt weights_only=True first
        model = None
        try:
            model = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            # Checkpoint contains full Python objects requiring unsafe deserialization
            if not allow_untrusted and not is_in_docker():
                raise SecurityPolicyError(
                    f"Security Policy Violation: Model '{path.name}' contains legacy pickled Python objects.\n"
                    "Loading this model in Lite Mode executes arbitrary Python bytecode on your host machine without container sandboxing.\n\n"
                    "Remediation Options:\n"
                    "  1. Run with Docker for automated sandbox confinement (recommended).\n"
                    "  2. Convert model to ONNX or Safetensors (safe, pickle-free formats).\n"
                    "  3. If you completely trust this model file, re-run with: --dangerously-allow-untrusted-model"
                )
            logger.warning(
                f"[SECURITY WARNING] Loading untrusted pickled model '{path.name}' with weights_only=False."
            )
            model = torch.load(path, map_location="cpu", weights_only=False)

        if isinstance(model, nn.Module):
            from lodevem.guards import assert_model_on_cpu
            assert_model_on_cpu(model)
            model.eval()
            return cls(model)

        raise TypeError(
            f"'{path.name}' contains a state dict, not a full model.\n"
            "To benchmark execution, provide a full model or use ONNX/TorchScript."
        )

    def generate_inputs(self, shape: tuple[int, ...] | None, **kwargs) -> Any:
        import torch
        from lodevem.guards import assert_tensor_on_cpu
        from lodevem.security import validate_input_shape

        # Use model-embedded shape if available, otherwise fallback to provided shape
        shape_to_use = getattr(self.model, "expected_input_shape", shape)
        shape_to_use = validate_input_shape(shape_to_use)
            
        tensor = torch.randn(*shape_to_use)
        assert_tensor_on_cpu(tensor, "PyTorch input tensor")
        return tensor

    def execute(self, inputs: Any) -> Any:
        import torch
        with torch.no_grad():
            return self.model(inputs)

    def set_threads(self, num_threads: int) -> None:
        import torch
        if hasattr(torch, "set_num_threads"):
            torch.set_num_threads(num_threads)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(num_threads)
            except Exception:
                pass
