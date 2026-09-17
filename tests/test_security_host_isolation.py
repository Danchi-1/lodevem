"""
Unit and integration tests for Host Prediction Security Isolation,
TorchScript classification, and sandboxing policy boundaries.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import torch
import torch.nn as nn
import joblib

from lodevem.security import (
    SecurityPolicyError,
    is_host_safe_format,
    scan_pickle_security,
)
from lodevem.backends.pytorch import PyTorchBackend
from lodevem.backends.sklearn import SklearnBackend
from lodevem.profiles import DeviceProfile
from lodevem.runner import run_benchmark


class DummyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 5)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def dummy_profile():
    from lodevem.profiles import load_all
    return load_all()[0]


def test_is_host_safe_format(tmp_path: Path):
    safe_st = tmp_path / "model.safetensors"
    safe_st.touch()
    assert is_host_safe_format(safe_st) is True

    safe_onnx = tmp_path / "model.onnx"
    safe_onnx.touch()
    assert is_host_safe_format(safe_onnx) is True

    unsafe_pt = tmp_path / "model.pt"
    unsafe_pt.touch()
    assert is_host_safe_format(unsafe_pt) is False

    unsafe_joblib = tmp_path / "model.joblib"
    unsafe_joblib.touch()
    assert is_host_safe_format(unsafe_joblib) is False


def test_host_prediction_skipped_when_docker_available(tmp_path: Path, dummy_profile):
    """
    Ensure unsafe models are NEVER loaded on the host for latency prediction
    when Docker is available. Host prediction must be skipped.
    """
    model_file = tmp_path / "model.pt"
    torch.save(DummyModule(), model_file)

    with patch("lodevem.runner._docker_available", return_value=True), \
         patch("lodevem.runner.build_image"), \
         patch("lodevem.runner.profile_loader.load_by_id", return_value=dummy_profile), \
         patch("lodevem.runner.measure_memory") as mock_measure, \
         patch("lodevem.runner.get_backend") as mock_get_backend:

        mock_measure.return_value = {
            "status": "ok",
            "fits_in_ram": True,
            "peak_ram_mb": 120.0,
            "median_latency_ms": 15.0,
            "p95_latency_ms": 18.0,
        }

        results = run_benchmark(
            model_paths=[model_file],
            profile_ids=["test_device"],
            allow_untrusted=False,
        )

        assert len(results) == 1
        record = results[0]
        # Host backend adapter was NOT loaded
        mock_get_backend.assert_not_called()
        # Prediction was skipped for host security
        assert record["prediction_status"] == "skipped (untrusted format on host)"
        assert record["predicted_latency_ms"] is None
        # Docker measurement ran
        mock_measure.assert_called_once()
        assert record["measure_status"] == "ok"


def test_lite_mode_untrusted_model_raises_security_policy_error(tmp_path: Path, dummy_profile):
    """
    In Lite mode (no Docker), loading an untrusted model without acknowledgment
    must fail early with SecurityPolicyError to prevent code execution on host.
    """
    model_file = tmp_path / "model.pt"
    torch.save(DummyModule(), model_file)

    with patch("lodevem.runner._docker_available", return_value=False), \
         patch("lodevem.runner.profile_loader.load_by_id", return_value=dummy_profile):

        with pytest.raises(SecurityPolicyError) as exc_info:
            run_benchmark(
                model_paths=[model_file],
                profile_ids=["test_device"],
                allow_untrusted=False,
            )

        assert "Security Policy Violation" in str(exc_info.value)
        assert "--dangerously-allow-untrusted-model" in str(exc_info.value)


def test_lite_mode_with_allow_untrusted_succeeds(tmp_path: Path, dummy_profile):
    """
    In Lite mode, passing allow_untrusted=True explicitly allows execution.
    """
    model_file = tmp_path / "model.pt"
    torch.save(DummyModule(), model_file)

    with patch("lodevem.runner._docker_available", return_value=False), \
         patch("lodevem.runner.profile_loader.load_by_id", return_value=dummy_profile), \
         patch("lodevem.runner.measure_memory") as mock_measure:

        mock_measure.return_value = {
            "status": "ok",
            "fits_in_ram": True,
            "peak_ram_mb": 110.0,
            "median_latency_ms": 10.0,
            "p95_latency_ms": 12.0,
        }

        results = run_benchmark(
            model_paths=[model_file],
            profile_ids=["test_device"],
            allow_untrusted=True,
            no_predict=True,
        )

        assert len(results) == 1
        assert results[0]["measure_status"] == "ok"


def test_torchscript_requires_allow_untrusted_in_lite_mode(tmp_path: Path):
    """
    TorchScript models execute native C++ IR. They require allow_untrusted=True
    outside Docker.
    """
    mod = DummyModule()
    ts_model = torch.jit.script(mod)
    ts_file = tmp_path / "scripted.pt"
    ts_model.save(str(ts_file))

    # Without allow_untrusted on host -> raises SecurityPolicyError
    with patch("lodevem.backends.pytorch_backend.is_in_docker", return_value=False):
        with pytest.raises(SecurityPolicyError, match="TorchScript model"):
            PyTorchBackend.load(ts_file, allow_untrusted=False)

    # With allow_untrusted=True -> succeeds
    with patch("lodevem.backends.pytorch_backend.is_in_docker", return_value=False):
        backend = PyTorchBackend.load(ts_file, allow_untrusted=True)
        assert backend is not None

    # Inside Docker -> succeeds without allow_untrusted
    with patch("lodevem.backends.pytorch_backend.is_in_docker", return_value=True):
        backend = PyTorchBackend.load(ts_file, allow_untrusted=False)
        assert backend is not None


def test_sklearn_requires_allow_untrusted_in_lite_mode(tmp_path: Path):
    """
    Scikit-Learn pickle/joblib requires allow_untrusted=True outside Docker.
    """
    from sklearn.linear_model import LinearRegression
    import numpy as np

    reg = LinearRegression()
    reg.fit(np.array([[1], [2]]), np.array([1, 2]))
    model_file = tmp_path / "linear.joblib"
    joblib.dump(reg, model_file)

    with patch("lodevem.backends.sklearn_backend.is_in_docker", return_value=False):
        with pytest.raises(SecurityPolicyError, match="uses Python pickle serialization"):
            SklearnBackend.load(model_file, allow_untrusted=False)

    mock_hb = MagicMock()
    mock_hb_model = MagicMock()
    mock_hb.ml.convert.return_value = mock_hb_model

    with patch.dict("sys.modules", {"hummingbird": mock_hb, "hummingbird.ml": mock_hb.ml}):
        with patch("lodevem.backends.sklearn_backend.is_in_docker", return_value=True):
            backend = SklearnBackend.load(model_file, allow_untrusted=False)
            assert backend is not None
