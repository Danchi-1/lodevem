from pathlib import Path
from typing import Type

from lodevem.backends.base import BenchmarkBackend


def get_backend(path: str | Path, allow_untrusted: bool = False) -> BenchmarkBackend:
    """
    Given a model file or directory, return an initialized backend adapter.
    """
    path = Path(path)
    
    # We import adapters locally to avoid circular imports and ensure
    # that any unexpected top-level imports in an adapter don't crash the CLI.
    from lodevem.backends.safetensors_backend import SafetensorsBackend
    from lodevem.backends.pytorch_backend import PyTorchBackend
    from lodevem.backends.onnx_backend import ONNXBackend
    from lodevem.backends.huggingface_backend import HuggingFaceBackend
    from lodevem.backends.sklearn_backend import SklearnBackend
    
    # List of available backend adapters
    backends: list[Type[BenchmarkBackend]] = [
        SafetensorsBackend,
        HuggingFaceBackend, # Check HF before PyTorch to avoid evaluating HF dirs as simple files
        PyTorchBackend,
        ONNXBackend,
        SklearnBackend,
    ]
    
    for backend_class in backends:
        if backend_class.supports(path):
            return backend_class.load(path, allow_untrusted=allow_untrusted)
            
    raise ValueError(f"No supported backend found for model: {path}")
