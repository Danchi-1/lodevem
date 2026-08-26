from __future__ import annotations

from pathlib import Path
from typing import Any

from lodevem.backends.base import BenchmarkBackend


class HuggingFaceBackend(BenchmarkBackend):
    """Backend adapter for Hugging Face Transformers models."""

    def __init__(self, model: Any, tokenizer: Any = None):
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def supports(cls, path: str | Path) -> bool:
        path = Path(path)
        return path.is_dir() and (path / "config.json").exists()

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkBackend:
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
        except ImportError as e:
            raise ImportError(f"Missing dependency. Run: pip install 'lodevem[hf]'. Details: {e}") from e

        path = Path(path)
        
        try:
            model = AutoModel.from_pretrained(str(path))
            model.eval()
            
            # Try to load tokenizer, but don't fail if it doesn't exist
            tokenizer = None
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(path))
            except Exception:
                pass
                
            return cls(model, tokenizer)
        except Exception as e:
            raise RuntimeError(f"Failed to load Hugging Face model from '{path}': {e}") from e

    def generate_inputs(self, shape: tuple[int, ...] | None) -> Any:
        import torch
        
        # If user provides a shape, use it, otherwise default to a batch of 1 with sequence length 128
        batch_size = shape[0] if shape else 1
        seq_len = shape[1] if shape and len(shape) > 1 else 128
        
        # Default fallback inputs if no tokenizer exists
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        
        if self.tokenizer:
            # If we have a tokenizer, use a dummy string to get perfectly valid input shapes/types
            dummy_text = ["Hello world, this is a benchmark test."] * batch_size
            inputs = self.tokenizer(dummy_text, return_tensors="pt", padding="max_length", max_length=seq_len, truncation=True)
            return inputs
            
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def execute(self, inputs: Any) -> Any:
        import torch
        with torch.no_grad():
            return self.model(**inputs)

    def set_threads(self, num_threads: int) -> None:
        import torch
        if hasattr(torch, "set_num_threads"):
            torch.set_num_threads(num_threads)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(num_threads)
            except Exception:
                pass
