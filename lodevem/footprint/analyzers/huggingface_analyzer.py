"""
huggingface.py — Hugging Face Transformers Footprint Analyzer

Analyzes Hugging Face model directories:
- Architecture parameters: vocab size, hidden dim, layer count, attention heads, KV heads
- Context window limit
- KV cache memory footprint per token and projections at key context lengths (512, 1024, 2048, 4096)
- Model parameter weights and precision breakdown
- Strict CPU verification
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from lodevem.footprint.schema import ModelFootprint
from lodevem.guards import assert_model_on_cpu, detect_host_gpus

logger = logging.getLogger(__name__)


def analyze_huggingface_footprint(
    model_path: str | Path,
    input_shape: tuple[int, ...] | None = None,
    use_gpu: bool = False,
    allow_untrusted: bool = False,
) -> ModelFootprint:
    """
    Analyze a Hugging Face model repository directory and return its ModelFootprint.
    """
    try:
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
        import torch
    except ImportError as e:
        raise ImportError(f"Missing Hugging Face dependency: {e}") from e

    path = Path(model_path).resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"Hugging Face model path must be a directory: {path}")

    from lodevem.security import SecurityPolicyError, is_in_docker
    has_safetensors = (path / "model.safetensors").exists() or any(path.glob("*.safetensors"))
    has_pickle = (path / "pytorch_model.bin").exists() or any(path.glob("*.bin"))

    if not has_safetensors and has_pickle and not allow_untrusted and not is_in_docker():
        raise SecurityPolicyError(
            f"Security Policy Violation: Hugging Face model in '{path.name}' contains pickled weights (.bin).\n"
            "Analyzing this model in Lite Mode executes arbitrary Python bytecode on your host machine without container sandboxing.\n\n"
            "Remediation Options:\n"
            "  1. Run with Docker for automated sandbox confinement (recommended).\n"
            "  2. Convert model weights to Safetensors format.\n"
            "  3. If you completely trust this model, re-run with: --dangerously-allow-untrusted-model"
        )

    config_file = path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Missing 'config.json' in '{path}'")

    # Directory total file size
    weights_files = [p for p in path.glob("**/*") if p.is_file()]
    file_size_bytes = sum(p.stat().st_size for p in weights_files)
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

    config = AutoConfig.from_pretrained(str(path), trust_remote_code=False)

    # Capabilities & LLM detection
    is_causal_lm = getattr(config, "is_decoder", False)
    if not is_causal_lm and hasattr(config, "architectures") and config.architectures:
        if any("CausalLM" in arch for arch in config.architectures):
            is_causal_lm = True

    # Architectural parameters
    vocab_size = getattr(config, "vocab_size", None)
    hidden_size = getattr(config, "hidden_size", getattr(config, "d_model", None))
    num_layers = getattr(
        config,
        "num_hidden_layers",
        getattr(config, "n_layer", getattr(config, "num_layers", None)),
    )
    num_attention_heads = getattr(config, "num_attention_heads", getattr(config, "n_head", None))
    num_key_value_heads = getattr(
        config,
        "num_key_value_heads",
        getattr(config, "num_kv_heads", num_attention_heads),
    )
    context_window = getattr(
        config,
        "max_position_embeddings",
        getattr(config, "n_positions", getattr(config, "seq_length", None)),
    )

    # KV Cache bytes per token calculation:
    # 2 (K & V) * num_layers * num_key_value_heads * head_dim * bytes_per_element
    kv_cache_bytes_per_token: Optional[float] = None
    kv_projections: Dict[int, float] = {}

    # Determine bytes per element from config torch_dtype if available
    torch_dtype_str = str(getattr(config, "torch_dtype", "float16"))
    dtype_bytes = 2.0  # default FP16 / BF16
    if "float32" in torch_dtype_str:
        dtype_bytes = 4.0
    elif "int8" in torch_dtype_str:
        dtype_bytes = 1.0

    if is_causal_lm and num_layers and hidden_size and num_attention_heads:
        head_dim = hidden_size // num_attention_heads
        kv_heads = num_key_value_heads or num_attention_heads
        # 2 * num_layers * kv_heads * head_dim * dtype_bytes
        kv_cache_bytes_per_token = 2 * num_layers * kv_heads * head_dim * dtype_bytes

        # Project memory at standard context lengths
        for ctx in [512, 1024, 2048, 4096, 8192]:
            if context_window is None or ctx <= context_window:
                kv_projections[ctx] = round((kv_cache_bytes_per_token * ctx) / (1024 * 1024), 2)

    # Load model on CPU to obtain exact parameter counts and precision
    try:
        if is_causal_lm:
            try:
                model = AutoModelForCausalLM.from_pretrained(str(path), device_map="cpu")
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(str(path)).to("cpu")
        else:
            try:
                model = AutoModel.from_pretrained(str(path), device_map="cpu")
            except Exception:
                model = AutoModel.from_pretrained(str(path)).to("cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to load Hugging Face model from '{path}': {e}") from e

    assert_model_on_cpu(model)
    model.eval()

    total_params = 0
    trainable_params = 0
    param_bytes = 0
    precision_breakdown: Dict[str, int] = {}

    for param in model.parameters():
        numel = param.numel()
        total_params += numel
        if param.requires_grad:
            trainable_params += numel

        element_size = param.element_size()
        param_bytes += numel * element_size

        dtype_name = str(param.dtype).replace("torch.", "").upper()
        precision_breakdown[dtype_name] = precision_breakdown.get(dtype_name, 0) + numel

    param_memory_mb = round(param_bytes / (1024 * 1024), 2)

    # Status tracking
    warnings: list[str] = []
    unavailable_metrics: list[str] = []
    metrics_status: Dict[str, str] = {
        "parameters": "exact",
        "weights_memory": "exact",
        "precision": "exact",
        "architecture_config": "exact",
    }

    if is_causal_lm:
        flops_status = "not_applicable"
        metrics_status["flops"] = "not_applicable"
        warnings.append(
            "FLOPs metric is not applicable to autoregressive decoder LLMs due to dynamic context scaling."
        )
    else:
        flops_status = "unsupported_operators"
        metrics_status["flops"] = "unsupported"
        warnings.append("FLOPs counting for generic encoder models is not supported statically.")

    # Estimated minimum RAM: parameters + minimum baseline
    estimated_minimum_ram_mb = round((param_memory_mb * 1.25) + 100.0, 1)

    host_gpus = detect_host_gpus()

    return ModelFootprint(
        model_path=str(path),
        model_name=path.name,
        backend="huggingface",
        file_size_bytes=file_size_bytes,
        file_size_mb=file_size_mb,
        input_shape=input_shape,
        total_parameters=total_params,
        trainable_parameters=trainable_params,
        parameter_memory_bytes=param_bytes,
        parameter_memory_mb=param_memory_mb,
        precision_breakdown=precision_breakdown,
        flops=None,
        macs=None,
        flops_status=flops_status,
        is_llm=is_causal_lm,
        context_window=context_window,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        kv_cache_bytes_per_token=kv_cache_bytes_per_token,
        kv_cache_projections_mb=kv_projections,
        estimated_minimum_ram_mb=estimated_minimum_ram_mb,
        gpu_diagnostic_mode=use_gpu,
        host_gpus_detected=host_gpus,
        metrics_status=metrics_status,
        unavailable_metrics=unavailable_metrics,
        warnings=warnings,
    )
