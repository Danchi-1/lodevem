from __future__ import annotations

from pathlib import Path
from typing import Any

from lodevem.backends.base import BenchmarkBackend
from lodevem.security import is_in_docker


class SklearnBackend(BenchmarkBackend):
    """Backend adapter for Scikit-Learn models via Hummingbird."""

    def __init__(self, hb_model: Any, n_features: int):
        self.hb_model = hb_model
        self.n_features = n_features

    @classmethod
    def supports(cls, path: str | Path) -> bool:
        path = Path(path)
        return path.suffix in {".pkl", ".joblib"}

    @classmethod
    def load(cls, path: str | Path, allow_untrusted: bool = False) -> BenchmarkBackend:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: '{path}'")

        from lodevem.security import scan_pickle_security, SecurityPolicyError
        scan_pickle_security(path)

        if not allow_untrusted and not is_in_docker():
            raise SecurityPolicyError(
                f"Security Policy Violation: Scikit-Learn model '{path.name}' uses Python pickle serialization.\n"
                "Loading this model in Lite Mode executes arbitrary Python bytecode on your host machine without container sandboxing.\n\n"
                "Remediation Options:\n"
                "  1. Run with Docker for automated sandbox confinement (recommended).\n"
                "  2. Convert model to ONNX or Safetensors (safe, pickle-free formats).\n"
                "  3. If you completely trust this model file, re-run with: --dangerously-allow-untrusted-model"
            )

        try:
            import joblib
            from hummingbird.ml import convert
            import torch
        except ImportError as e:
            raise ImportError(f"Missing dependency. Run: pip install 'lodevem[sklearn]'. Details: {e}") from e

        try:
            sklearn_model = joblib.load(path)
            
            # Determine input features
            n_features = getattr(sklearn_model, "n_features_in_", None)
            if n_features is None:
                # Fallback if the estimator doesn't expose it
                n_features = 10
                
            # Convert to PyTorch backend via Hummingbird
            # We convert to PyTorch because lodevem relies heavily on CPU thread simulation 
            # and memory profiling which works seamlessly with PyTorch backends.
            hb_model = convert(sklearn_model, 'pytorch')
            hb_model.model.eval()
            
            return cls(hb_model, n_features)
        except Exception as e:
            raise RuntimeError(f"Failed to load Sklearn model from '{path}': {e}") from e

    def generate_inputs(self, shape: tuple[int, ...] | None, **kwargs) -> Any:
        import torch
        
        batch_size = shape[0] if shape else 1
        # Ignore user provided feature count and use what the model specifically requires
        return torch.randn(batch_size, self.n_features)

    def execute(self, inputs: Any) -> Any:
        import torch
        with torch.no_grad():
            return self.hb_model.predict(inputs)

    def set_threads(self, num_threads: int) -> None:
        import torch
        if hasattr(torch, "set_num_threads"):
            torch.set_num_threads(num_threads)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(num_threads)
            except Exception:
                pass
