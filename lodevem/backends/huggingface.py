from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict

from lodevem.backends.base import BenchmarkBackend


def _read_rss_kb() -> int:
    """Read current RSS (Resident Set Size) from /proc/self/status in KB."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0


class HuggingFaceBackend(BenchmarkBackend):
    """Backend adapter for Hugging Face Transformers models."""

    def __init__(self, model: Any, tokenizer: Any = None, is_causal_lm: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self._is_causal_lm = is_causal_lm

    @classmethod
    def supports(cls, path: str | Path) -> bool:
        path = Path(path)
        return path.is_dir() and (path / "config.json").exists()

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkBackend:
        try:
            from transformers import AutoModel, AutoModelForCausalLM, AutoConfig, AutoTokenizer
            import torch
        except ImportError as e:
            raise ImportError(f"Missing dependency. Run: pip install 'lodevem[hf]'. Details: {e}") from e

        path = Path(path)
        
        try:
            config = AutoConfig.from_pretrained(str(path))
            
            # Detect model capabilities
            # For this phase, we explicitly support Causal LMs (decoder-only)
            # or fallback to general models for simple forward passes.
            is_causal_lm = getattr(config, "is_decoder", False)
            # Some models don't set is_decoder but have CausalLM architectures
            if not is_causal_lm and hasattr(config, "architectures") and config.architectures:
                if any("CausalLM" in arch for arch in config.architectures):
                    is_causal_lm = True
                    
            if is_causal_lm:
                model = AutoModelForCausalLM.from_pretrained(str(path))
            else:
                # Reject encoder-only if they are expected to do generation, but we support 
                # them for standard non-LLM `execute` benchmarks.
                model = AutoModel.from_pretrained(str(path))
                
            model.eval()
            
            # Load tokenizer (Required for LLMs, optional for normal models)
            tokenizer = None
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(path))
            except Exception:
                if is_causal_lm:
                    raise RuntimeError(f"Tokenizer not found in '{path}'. LLM execution requires a valid tokenizer.")
                
            return cls(model, tokenizer, is_causal_lm=is_causal_lm)
        except Exception as e:
            raise RuntimeError(f"Failed to load Hugging Face model from '{path}': {e}") from e

    def is_llm(self) -> bool:
        return self._is_causal_lm

    def generate_inputs(self, shape: tuple[int, ...] | None = None, **kwargs) -> Any:
        import torch
        
        prompt = kwargs.get("prompt", None)

        if self._is_causal_lm:
            if not prompt:
                prompt = "The quick brown fox"
            if not self.tokenizer:
                raise RuntimeError("Cannot generate LLM inputs without a tokenizer.")
            
            # Ensure pad token exists
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 0
                
            inputs = self.tokenizer(prompt, return_tensors="pt")
            return inputs
            
        else:
            # Fallback for standard encoder models
            batch_size = shape[0] if shape else 1
            seq_len = shape[1] if shape and len(shape) > 1 else 128
            
            if self.tokenizer:
                dummy_text = ["Hello world, this is a benchmark test."] * batch_size
                inputs = self.tokenizer(dummy_text, return_tensors="pt", padding="max_length", max_length=seq_len, truncation=True)
                return inputs
                
            input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    def execute(self, inputs: Any) -> Any:
        import torch
        with torch.no_grad():
            return self.model(**inputs)

    def execute_llm(self, inputs: Any, max_new_tokens: int) -> dict:
        import torch
        if not self._is_causal_lm:
            raise NotImplementedError("Model is not a Causal LM.")
            
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
            
        input_ids = inputs["input_ids"]
        prompt_length = input_ids.shape[-1]
        
        generated_tokens = 0
        peak_rss_kb = _read_rss_kb()
        
        # Prepare for generation loop
        # We manually drive the forward pass to measure exact TTFT and TPS
        past_key_values = None
        current_input_ids = input_ids
        
        eos_token_id = self.model.config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        
        ttft_ms = 0.0
        decode_time_ms = 0.0
        
        t_start = time.perf_counter()
        t_first_token = 0.0
        
        with torch.no_grad():
            for i in range(max_new_tokens):
                # Sample memory during loop
                rss = _read_rss_kb()
                if rss > peak_rss_kb:
                    peak_rss_kb = rss
                
                outputs = self.model(
                    input_ids=current_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]
                
                # Deterministic greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                
                if i == 0:
                    t_first_token = time.perf_counter()
                    ttft_ms = (t_first_token - t_start) * 1000
                
                generated_tokens += 1
                
                # Check EOS
                if eos_token_id is not None and next_token.item() in eos_token_id:
                    break
                    
                # Setup next iteration
                current_input_ids = next_token

        t_end = time.perf_counter()
        decode_time_ms = (t_end - t_first_token) * 1000
        
        tps = None
        if generated_tokens > 1:
            tps = (generated_tokens - 1) / (decode_time_ms / 1000.0)
            
        return {
            "ttft_ms": ttft_ms,
            "decode_time_ms": decode_time_ms,
            "tps": tps,
            "generated_tokens": generated_tokens,
            "prompt_length": prompt_length,
            "peak_ram_mb": peak_rss_kb / 1024.0
        }

    def set_threads(self, num_threads: int) -> None:
        import torch
        if hasattr(torch, "set_num_threads"):
            torch.set_num_threads(num_threads)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(num_threads)
            except Exception:
                pass
