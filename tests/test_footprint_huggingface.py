"""
Unit tests for Hugging Face model footprint analyzer.
"""

import json
from pathlib import Path
import pytest

try:
    import transformers
    import torch
    HAS_HF = True
except ImportError:
    HAS_HF = False


def test_huggingface_kv_cache_formula_logic():
    num_layers = 4
    hidden_size = 256
    num_attention_heads = 8
    num_key_value_heads = 4
    dtype_bytes = 2.0
    head_dim = hidden_size // num_attention_heads
    kv_heads = num_key_value_heads
    kv_cache_bytes_per_token = 2 * num_layers * kv_heads * head_dim * dtype_bytes
    assert kv_cache_bytes_per_token == 2048.0

    # 512 and 2048 context projection in MB
    mb_512 = round((kv_cache_bytes_per_token * 512) / (1024 * 1024), 2)
    mb_2048 = round((kv_cache_bytes_per_token * 2048) / (1024 * 1024), 2)
    assert mb_512 == 1.0
    assert mb_2048 == 4.0


@pytest.mark.skipif(not HAS_HF, reason="transformers is required for this test")
def test_huggingface_kv_cache_and_footprint(tmp_path):
    model_dir = tmp_path / "dummy_llm"
    model_dir.mkdir()

    # Create dummy config for a CausalLM decoder model
    config = {
        "architectures": ["LlamaForCausalLM"],
        "is_decoder": True,
        "vocab_size": 32000,
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,  # Grouped-Query Attention (GQA)
        "max_position_embeddings": 2048,
        "torch_dtype": "float16",
        "model_type": "llama",
    }
    (model_dir / "config.json").write_text(json.dumps(config))

    # Also save a tiny state dict so from_pretrained succeeds or analyze handles it
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_config(cfg)
    model.save_pretrained(str(model_dir))

    from lodevem.footprint.analyzers.huggingface import analyze_huggingface_footprint

    fp = analyze_huggingface_footprint(model_dir)

    assert fp.backend == "huggingface"
    assert fp.is_llm is True
    assert fp.context_window == 2048
    assert fp.vocab_size == 32000
    assert fp.hidden_size == 256
    assert fp.num_layers == 4
    assert fp.num_attention_heads == 8
    assert fp.num_key_value_heads == 4

    # KV Cache verification:
    # head_dim = 256 // 8 = 32
    # kv_heads = 4
    # dtype_bytes = 2.0 (float16)
    # per_token = 2 * 4 * 4 * 32 * 2.0 = 2048 bytes / token
    assert fp.kv_cache_bytes_per_token == 2048.0
    assert 512 in fp.kv_cache_projections_mb
    # 2048 bytes * 512 tokens = 1048576 bytes = 1.0 MB
    assert fp.kv_cache_projections_mb[512] == 1.0
    assert fp.kv_cache_projections_mb[2048] == 4.0

    assert fp.flops_status == "not_applicable"
    assert fp.total_parameters > 0
    assert fp.parameter_memory_mb > 0
