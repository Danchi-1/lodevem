"""
Unit tests for Safetensors security validation, format parsing, and backend execution.
"""

import json
import struct
from pathlib import Path
import pytest
import torch

from lodevem.security import (
    SecurityError,
    validate_safetensors_security,
    MAX_SAFETENSORS_HEADER_BYTES,
)
from lodevem.backends.safetensors import SafetensorsBackend
from lodevem.footprint.analyzers.safetensors import analyze_safetensors_footprint


def _create_minimal_safetensors(path: Path, tensors_meta: dict, tensor_bytes: bytes) -> None:
    header_json = json.dumps(tensors_meta).encode("utf-8")
    header_len = len(header_json)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_json)
        f.write(tensor_bytes)


def test_valid_safetensors_validation(tmp_path: Path):
    model_file = tmp_path / "model.safetensors"
    meta = {
        "weight1": {
            "dtype": "F32",
            "shape": [2, 3],
            "data_offsets": [0, 24],
        },
        "__metadata__": {"format": "pt"},
    }
    raw_bytes = b"\x00" * 24
    _create_minimal_safetensors(model_file, meta, raw_bytes)

    header = validate_safetensors_security(model_file)
    assert "weight1" in header
    assert header["weight1"]["shape"] == [2, 3]


def test_safetensors_truncated_file(tmp_path: Path):
    model_file = tmp_path / "truncated.safetensors"
    model_file.write_bytes(b"\x04\x00")  # Only 2 bytes

    with pytest.raises(SecurityError, match="Truncated Safetensors file"):
        validate_safetensors_security(model_file)


def test_safetensors_invalid_header_len(tmp_path: Path):
    model_file = tmp_path / "zero_header.safetensors"
    # Header length 0
    with open(model_file, "wb") as f:
        f.write(struct.pack("<Q", 0))

    with pytest.raises(SecurityError, match="Invalid Safetensors header length"):
        validate_safetensors_security(model_file)


def test_safetensors_header_too_large(tmp_path: Path):
    model_file = tmp_path / "huge_header.safetensors"
    huge_len = MAX_SAFETENSORS_HEADER_BYTES + 1024
    with open(model_file, "wb") as f:
        f.write(struct.pack("<Q", huge_len))

    with pytest.raises(SecurityError, match="exceeds limit"):
        validate_safetensors_security(model_file)


def test_safetensors_malformed_json(tmp_path: Path):
    model_file = tmp_path / "bad_json.safetensors"
    bad_bytes = b"not json at all"
    with open(model_file, "wb") as f:
        f.write(struct.pack("<Q", len(bad_bytes)))
        f.write(bad_bytes)

    with pytest.raises(SecurityError, match="Malformed JSON"):
        validate_safetensors_security(model_file)


def test_safetensors_inverted_offsets(tmp_path: Path):
    model_file = tmp_path / "inverted.safetensors"
    meta = {
        "weight1": {
            "dtype": "F32",
            "shape": [2, 2],
            "data_offsets": [20, 10],  # begin > end
        }
    }
    _create_minimal_safetensors(model_file, meta, b"\x00" * 30)

    with pytest.raises(SecurityError, match="Negative or inverted data_offsets"):
        validate_safetensors_security(model_file)


def test_safetensors_offset_out_of_bounds(tmp_path: Path):
    model_file = tmp_path / "oob.safetensors"
    meta = {
        "weight1": {
            "dtype": "F32",
            "shape": [2, 2],
            "data_offsets": [0, 100],  # buffer only has 20 bytes
        }
    }
    _create_minimal_safetensors(model_file, meta, b"\x00" * 20)

    with pytest.raises(SecurityError, match="offset out of bounds"):
        validate_safetensors_security(model_file)


def test_safetensors_footprint_analysis(tmp_path: Path):
    model_file = tmp_path / "model.safetensors"
    meta = {
        "layer.weight": {
            "dtype": "F32",
            "shape": [10, 20],
            "data_offsets": [0, 800],
        },
        "layer.bias": {
            "dtype": "F32",
            "shape": [20],
            "data_offsets": [800, 880],
        },
    }
    _create_minimal_safetensors(model_file, meta, b"\x00" * 880)

    fp = analyze_safetensors_footprint(model_file)
    assert fp.backend == "safetensors"
    assert fp.total_parameters == 200 + 20
    assert fp.precision_breakdown == {"FP32": 220}
    assert fp.parameter_memory_bytes == 880
    assert fp.metrics_status["parameters"] == "exact"


def test_safetensors_backend_load_rejects_raw_weights(tmp_path: Path):
    model_file = tmp_path / "linear.safetensors"
    meta = {
        "weight": {
            "dtype": "F32",
            "shape": [10],
            "data_offsets": [0, 40],
        }
    }
    _create_minimal_safetensors(model_file, meta, b"\x00" * 40)

    assert SafetensorsBackend.supports(model_file)
    with pytest.raises(TypeError, match="raw Safetensors weight archive, not an executable model"):
        SafetensorsBackend.load(model_file)
