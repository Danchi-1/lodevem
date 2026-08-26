from __future__ import annotations

from pathlib import Path
from typing import Any

from lodevem.backends.base import BenchmarkBackend


class PyTorchBackend(BenchmarkBackend):
    """Backend adapter for PyTorch models (.pt, .pth)."""

    def __init__(self, model: Any):
        self.model = model

    @classmethod
    def supports(cls, path: str | Path) -> bool:
        path = Path(path)
        return path.suffix in {".pt", ".pth"}

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkBackend:
        try:
            import torch
            import torch.nn as nn
        except ImportError as e:
            raise ImportError(f"Missing dependency. Run: pip install 'lodevem[pytorch]'. Details: {e}") from e

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: '{path}'")

        try:
            model = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(model, nn.Module):
                model.eval()
                return cls(model)

            raise TypeError(
                f"'{path.name}' contains a state dict, not a full model.\n"
                "To save a full model: torch.save(model, 'path.pt')"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model from '{path}': {e}") from e

    def generate_inputs(self, shape: tuple[int, ...] | None) -> Any:
        import torch
        # Use model-embedded shape if available, otherwise fallback to provided shape
        shape_to_use = getattr(self.model, "expected_input_shape", shape)
        if not shape_to_use:
            shape_to_use = (1, 3, 224, 224)
            
        return torch.randn(*shape_to_use)

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
