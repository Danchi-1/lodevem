"""
Unit tests for ONNX security validation:
- Resource ceilings (file size, graph node count, tensor dimensions, parameter counts)
- External data path traversal defense (absolute paths, directory traversal, symlink escapes)
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from lodevem.security import (
    SecurityError,
    validate_onnx_security,
    MAX_FILE_SIZE_BYTES,
)


def test_onnx_file_not_found(tmp_path: Path):
    missing = tmp_path / "missing.onnx"
    with pytest.raises(FileNotFoundError):
        validate_onnx_security(missing)


def test_onnx_file_size_exceeded(tmp_path: Path):
    fake_onnx = tmp_path / "huge.onnx"
    fake_onnx.touch()
    with patch.object(Path, "stat") as mock_stat:
        mock_stat_res = MagicMock()
        mock_stat_res.st_size = MAX_FILE_SIZE_BYTES + 1024
        mock_stat.return_value = mock_stat_res
        with pytest.raises(SecurityError, match="exceeds security ceiling"):
            validate_onnx_security(fake_onnx)


def test_onnx_node_count_limit(tmp_path: Path):
    model_file = tmp_path / "deep.onnx"
    model_file.write_bytes(b"dummy onnx content")

    mock_onnx = MagicMock()
    mock_model = MagicMock()
    mock_model.graph.node = [MagicMock()] * 500_001
    mock_onnx.load.return_value = mock_model

    with patch.dict(sys.modules, {"onnx": mock_onnx}):
        with pytest.raises(SecurityError, match="exceeding safety limit of 500,000"):
            validate_onnx_security(model_file)


def test_onnx_external_data_traversal_blocked(tmp_path: Path):
    model_dir = tmp_path / "model_folder"
    model_dir.mkdir()
    model_file = model_dir / "model.onnx"
    model_file.write_bytes(b"dummy onnx content")

    mock_onnx = MagicMock()
    mock_onnx.TensorProto.EXTERNAL = 1

    tensor = MagicMock()
    tensor.dims = [10, 10]
    tensor.data_location = 1
    
    entry = MagicMock()
    entry.key = "location"
    entry.value = "../../etc/shadow"
    tensor.external_data = [entry]

    mock_model = MagicMock()
    mock_model.graph.node = []
    mock_model.graph.initializer = [tensor]
    mock_onnx.load.return_value = mock_model

    with patch.dict(sys.modules, {"onnx": mock_onnx}):
        with pytest.raises(SecurityError, match="ONNX External Data Path Traversal"):
            validate_onnx_security(model_file)


def test_onnx_external_data_absolute_path_blocked(tmp_path: Path):
    model_dir = tmp_path / "model_folder"
    model_dir.mkdir()
    model_file = model_dir / "model.onnx"
    model_file.write_bytes(b"dummy onnx content")

    mock_onnx = MagicMock()
    mock_onnx.TensorProto.EXTERNAL = 1

    tensor = MagicMock()
    tensor.dims = [10, 10]
    tensor.data_location = 1

    entry = MagicMock()
    entry.key = "location"
    entry.value = "/var/data/weights.bin"
    tensor.external_data = [entry]

    mock_model = MagicMock()
    mock_model.graph.node = []
    mock_model.graph.initializer = [tensor]
    mock_onnx.load.return_value = mock_model

    with patch.dict(sys.modules, {"onnx": mock_onnx}):
        with pytest.raises(SecurityError, match="ONNX External Data Path Traversal"):
            validate_onnx_security(model_file)


def test_onnx_external_data_symlink_escape_blocked(tmp_path: Path):
    model_dir = tmp_path / "model_folder"
    model_dir.mkdir()
    model_file = model_dir / "model.onnx"
    model_file.write_bytes(b"dummy onnx content")

    secret_file = tmp_path / "secret.bin"
    secret_file.write_bytes(b"confidential data")

    symlink_file = model_dir / "weights_symlink.bin"
    symlink_file.symlink_to(secret_file)

    mock_onnx = MagicMock()
    mock_onnx.TensorProto.EXTERNAL = 1

    tensor = MagicMock()
    tensor.dims = [10, 10]
    tensor.data_location = 1

    entry = MagicMock()
    entry.key = "location"
    entry.value = "weights_symlink.bin"
    tensor.external_data = [entry]

    mock_model = MagicMock()
    mock_model.graph.node = []
    mock_model.graph.initializer = [tensor]
    mock_onnx.load.return_value = mock_model

    with patch.dict(sys.modules, {"onnx": mock_onnx}):
        with pytest.raises(SecurityError, match="escapes model directory"):
            validate_onnx_security(model_file)


def test_onnx_valid_graph_passes(tmp_path: Path):
    model_dir = tmp_path / "model_folder"
    model_dir.mkdir()
    model_file = model_dir / "model.onnx"
    model_file.write_bytes(b"dummy onnx content")

    data_file = model_dir / "weights.bin"
    data_file.write_bytes(b"\x00" * 400)

    mock_onnx = MagicMock()
    mock_onnx.TensorProto.EXTERNAL = 1

    tensor = MagicMock()
    tensor.dims = [10, 10]
    tensor.data_location = 1

    entry = MagicMock()
    entry.key = "location"
    entry.value = "weights.bin"
    mock_node = MagicMock()
    mock_node.domain = ""
    mock_node.op_type = "Relu"
    mock_model = MagicMock()
    mock_model.graph.node = [mock_node] * 10
    mock_model.graph.initializer = [tensor]
    mock_onnx.load.return_value = mock_model

    with patch.dict(sys.modules, {"onnx": mock_onnx}):
        # Should not raise
        validate_onnx_security(model_file)
