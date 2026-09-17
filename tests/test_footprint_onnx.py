"""
Unit tests for ONNX model footprint analyzer.
"""

from pathlib import Path
import pytest
from lodevem.footprint.analyzers.onnx import analyze_onnx_footprint, ONNX_TYPE_MAP


def test_onnx_type_map():
    assert ONNX_TYPE_MAP[1] == ("FP32", 4)
    assert ONNX_TYPE_MAP[3] == ("INT8", 1)
    assert ONNX_TYPE_MAP[10] == ("FP16", 2)


def test_onnx_file_not_found(tmp_path):
    missing_file = tmp_path / "non_existent.onnx"
    with pytest.raises(FileNotFoundError):
        analyze_onnx_footprint(missing_file)
