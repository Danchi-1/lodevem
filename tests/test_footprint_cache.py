"""
Unit tests for ModelFootprint schema and caching system.
"""

import time
from pathlib import Path
from lodevem.footprint.schema import ModelFootprint
from lodevem.footprint.cache import (
    compute_cache_key,
    get_cached_footprint,
    save_cached_footprint,
)


def test_model_footprint_serialization():
    fp = ModelFootprint(
        model_path="/path/to/model.pt",
        model_name="model.pt",
        backend="pytorch",
        file_size_bytes=1048576,
        file_size_mb=1.0,
        input_shape=(1, 3, 224, 224),
        total_parameters=1000,
        parameter_memory_mb=0.01,
        precision_breakdown={"FP32": 1000},
        flops=2000,
        macs=1000,
        flops_status="exact",
    )

    data = fp.to_dict()
    assert data["model_name"] == "model.pt"
    assert data["backend"] == "pytorch"
    assert data["total_parameters"] == 1000
    assert data["input_shape"] == [1, 3, 224, 224]

    restored = ModelFootprint.from_dict(data)
    assert restored.model_name == fp.model_name
    assert restored.total_parameters == 1000
    assert restored.input_shape == (1, 3, 224, 224)


def test_cache_key_invalidation_on_change(tmp_path):
    dummy_model = tmp_path / "model.pt"
    dummy_model.write_bytes(b"initial_content")

    key1 = compute_cache_key(dummy_model, "pytorch", input_shape=(1, 3, 224, 224))
    key2 = compute_cache_key(dummy_model, "pytorch", input_shape=(1, 3, 224, 224))
    assert key1 == key2

    # Different input shape must change key
    key_diff_shape = compute_cache_key(dummy_model, "pytorch", input_shape=(1, 1, 28, 28))
    assert key1 != key_diff_shape

    # Different GPU mode must change key
    key_gpu = compute_cache_key(dummy_model, "pytorch", input_shape=(1, 3, 224, 224), gpu_mode=True)
    assert key1 != key_gpu

    # Modifying file content / size must change key
    dummy_model.write_bytes(b"updated_content_with_different_size")
    key3 = compute_cache_key(dummy_model, "pytorch", input_shape=(1, 3, 224, 224))
    assert key1 != key3


def test_save_and_retrieve_cache(tmp_path):
    cache_dir = tmp_path / ".cache"
    dummy_model = tmp_path / "model.pt"
    dummy_model.write_bytes(b"model_weights")

    fp = ModelFootprint(
        model_path=str(dummy_model),
        model_name=dummy_model.name,
        backend="pytorch",
        file_size_bytes=dummy_model.stat().st_size,
        file_size_mb=round(dummy_model.stat().st_size / (1024 * 1024), 2),
        input_shape=(1, 3, 224, 224),
        total_parameters=50000,
        parameter_memory_mb=0.2,
        precision_breakdown={"FP32": 50000},
    )

    # Initially cache should miss
    assert get_cached_footprint(dummy_model, "pytorch", (1, 3, 224, 224), cache_dir=cache_dir) is None

    # Save to cache
    save_cached_footprint(fp, cache_dir=cache_dir)

    # Retrieve from cache
    cached = get_cached_footprint(dummy_model, "pytorch", (1, 3, 224, 224), cache_dir=cache_dir)
    assert cached is not None
    assert cached.total_parameters == 50000
    assert cached.model_name == "model.pt"
