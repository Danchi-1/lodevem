"""
Integration tests for CLI footprint command and GPU guardrail enforcement.
"""

import json
import pytest
from unittest.mock import patch
import joblib
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from lodevem.cli import build_parser, main


def test_cli_footprint_command(tmp_path, capsys):
    model = DecisionTreeRegressor(max_depth=2)
    X = np.random.randn(10, 3)
    y = np.random.randn(10)
    model.fit(X, y)

    model_file = tmp_path / "tree.joblib"
    joblib.dump(model, model_file)
    json_out = tmp_path / "footprint.json"

    parser = build_parser()

    # 1. Blocked without --dangerously-allow-untrusted-model
    args_blocked = parser.parse_args(["footprint", str(model_file), "--json", str(json_out)])
    with pytest.raises(SystemExit) as exc_info:
        args_blocked.func(args_blocked)
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "[SECURITY POLICY BLOCKED]" in captured.out
    assert "--dangerously-allow-untrusted-model" in captured.out

    # 2. Allowed with --dangerously-allow-untrusted-model
    args_allowed = parser.parse_args([
        "footprint",
        str(model_file),
        "--json",
        str(json_out),
        "--dangerously-allow-untrusted-model",
    ])
    args_allowed.func(args_allowed)

    captured_allowed = capsys.readouterr()
    assert "lodevem Model Footprint Scorecard" in captured_allowed.out
    assert "DecisionTreeRegressor" in captured_allowed.out
    assert json_out.exists()

    data = json.loads(json_out.read_text())
    assert len(data) == 1
    assert data[0]["estimator_type"] == "DecisionTreeRegressor"
    assert data[0]["backend"] == "sklearn"


def test_cli_start_rejects_use_gpu_without_override(tmp_path, capsys):
    model = DecisionTreeRegressor(max_depth=2)
    X = np.random.randn(10, 3)
    y = np.random.randn(10)
    model.fit(X, y)
    model_file = tmp_path / "tree.joblib"
    joblib.dump(model, model_file)

    parser = build_parser()
    args = parser.parse_args(["start", str(model_file), "--use-gpu"])

    with pytest.raises(SystemExit) as exc_info:
        args.func(args)

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "'--use-gpu' cannot be used with 'lodevem start'" in captured.out
    assert "--allow-invalid-mobile-gpu" in captured.out
