from pathlib import Path
from typing import Type

from lodevem.backends.base import BenchmarkBackend


def get_backend(path: str | Path) -> BenchmarkBackend:
    """
    Given a model file or directory, return an initialized backend adapter.
    """
    path = Path(path)
    
    # We import adapters locally to avoid circular imports and ensure
    # that any unexpected top-level imports in an adapter don't crash the CLI.
    from lodevem.backends.pytorch import PyTorchBackend
    from lodevem.backends.onnx import ONNXBackend
    from lodevem.backends.huggingface import HuggingFaceBackend
    from lodevem.backends.sklearn import SklearnBackend
    
    # List of available backend adapters
    backends: list[Type[BenchmarkBackend]] = [
        HuggingFaceBackend, # Check HF before PyTorch to avoid evaluating HF dirs as simple files
        PyTorchBackend,
        ONNXBackend,
        SklearnBackend,
    ]
    
    for backend_class in backends:
        if backend_class.supports(path):
            return backend_class.load(path)
            
    raise ValueError(f"No supported backend found for model: {path}")
