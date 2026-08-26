from __future__ import annotations

import abc
from pathlib import Path
from typing import Any


class BenchmarkBackend(abc.ABC):
    """
    Abstract Base Class for universal model benchmarking adapters.
    Separates model loading, input generation, and execution logic
    from measurement orchestration.
    """

    @classmethod
    @abc.abstractmethod
    def supports(cls, path: str | Path) -> bool:
        """
        Return True if this backend can load and execute the given model path.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, path: str | Path) -> BenchmarkBackend:
        """
        Load the model and return an initialized backend adapter instance.
        """
        pass

    @abc.abstractmethod
    def generate_inputs(self, shape: tuple[int, ...] | None, **kwargs) -> Any:
        """
        Generate dummy inputs suitable for this backend.
        The shape tuple is provided by the user CLI, but backends may override
        this if the model embeds expected shapes (e.g. ONNX metadata).
        """
        pass

    @abc.abstractmethod
    def execute(self, inputs: Any) -> Any:
        """
        Perform a single forward pass / inference step with the given inputs.
        """
        pass

    def set_threads(self, num_threads: int) -> None:
        """
        Set the number of threads used for execution.
        Called to simulate CPU throttling on constrained devices.
        Defaults to a no-op if the backend doesn't support thread control.
        """
        pass

    def is_llm(self) -> bool:
        """
        Return True if this backend is currently loaded with an LLM and supports LLM metrics.
        """
        return False
        
    def execute_llm(self, inputs: Any, max_new_tokens: int) -> dict:
        """
        Perform an autoregressive generation loop.
        
        Returns:
            dict with strict schema:
            {
                "ttft_ms": float,
                "decode_time_ms": float,
                "tps": float | None,
                "generated_tokens": int,
                "prompt_length": int,
                "peak_ram_mb": float
            }
        """
        raise NotImplementedError("LLM generation is not supported by this backend.")
