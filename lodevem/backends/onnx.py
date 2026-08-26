from __future__ import annotations

from pathlib import Path
from typing import Any

from lodevem.backends.base import BenchmarkBackend


class ONNXBackend(BenchmarkBackend):
    """Backend adapter for ONNX models (.onnx)."""

    def __init__(self, session: Any, input_names: list[str], input_types: dict[str, Any], input_shapes: dict[str, tuple]):
        self.session = session
        self.input_names = input_names
        self.input_types = input_types
        self.input_shapes = input_shapes

    @classmethod
    def supports(cls, path: str | Path) -> bool:
        return Path(path).suffix == ".onnx"

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkBackend:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(f"Missing dependency. Run: pip install 'lodevem[onnx]'. Details: {e}") from e

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: '{path}'")

        try:
            session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
            
            input_names = []
            input_types = {}
            input_shapes = {}
            
            for node in session.get_inputs():
                input_names.append(node.name)
                input_types[node.name] = node.type
                
                # Handle symbolic dimensions (e.g. 'batch_size' -> 1)
                shape = []
                for dim in node.shape:
                    if isinstance(dim, str) or dim is None:
                        shape.append(1) # Default symbolic dim to 1
                    else:
                        shape.append(dim)
                input_shapes[node.name] = tuple(shape)
                
            return cls(session, input_names, input_types, input_shapes)
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model from '{path}': {e}") from e

    def generate_inputs(self, shape: tuple[int, ...] | None, **kwargs) -> Any:
        import numpy as np
        
        inputs = {}
        for name in self.input_names:
            node_shape = self.input_shapes[name]
            node_type = self.input_types[name]
            
            # Map ONNX types to numpy types
            dtype = np.float32
            if "int64" in node_type:
                dtype = np.int64
            elif "int32" in node_type:
                dtype = np.int32
            elif "float16" in node_type:
                dtype = np.float16
                
            # If user provided a shape via CLI, we can override the first input's shape
            if shape and name == self.input_names[0]:
                use_shape = shape
            else:
                use_shape = node_shape
                
            inputs[name] = np.random.randn(*use_shape).astype(dtype)
            
        return inputs

    def execute(self, inputs: Any) -> Any:
        return self.session.run(None, inputs)

    def set_threads(self, num_threads: int) -> None:
        # We can't change threads on an active InferenceSession in ONNX Runtime.
        # We would need to recreate the session with new SessionOptions.
        # For now, we skip dynamic thread simulation for ONNX.
        pass
