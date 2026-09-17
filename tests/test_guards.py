"""
Unit tests for CPU isolation guards and hardware guardrails.
"""

import os
import subprocess
import sys
import pytest

from lodevem.guards import (
    CPU_MASK_ENV,
    get_cpu_isolated_env,
    isolate_cpu_pre_import,
    detect_host_gpus,
    assert_tensor_on_cpu,
    assert_inputs_on_cpu,
    assert_model_on_cpu,
)


def test_get_cpu_isolated_env():
    env = get_cpu_isolated_env({"EXISTING_VAR": "123", "CUDA_VISIBLE_DEVICES": "0,1"})
    assert env["EXISTING_VAR"] == "123"
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["ROCR_VISIBLE_DEVICES"] == ""
    assert env["HIP_VISIBLE_DEVICES"] == ""


def test_isolate_cpu_pre_import(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")

    isolate_cpu_pre_import()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert os.environ["ROCR_VISIBLE_DEVICES"] == ""
    assert os.environ["HIP_VISIBLE_DEVICES"] == ""


def test_subprocess_isolation_execution():
    """Verify fresh subprocess spawned with get_cpu_isolated_env() has masked GPU environment."""
    code = (
        "import os\n"
        "assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''\n"
        "assert os.environ.get('ROCR_VISIBLE_DEVICES') == ''\n"
        "assert os.environ.get('HIP_VISIBLE_DEVICES') == ''\n"
        "print('ISOLATED_OK')\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        env=get_cpu_isolated_env(),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "ISOLATED_OK" in res.stdout


def test_assert_tensor_on_cpu_mock():
    class MockCPUTensor:
        class Device:
            type = "cpu"
        device = Device()

    class MockGPUTensor:
        class Device:
            type = "cuda"
        device = Device()

    # CPU tensor should not raise
    assert_tensor_on_cpu(MockCPUTensor())

    # GPU tensor must raise
    with pytest.raises(RuntimeError, match="CPU Guardrail Violation"):
        assert_tensor_on_cpu(MockGPUTensor())


def test_assert_inputs_on_cpu_nested():
    class MockCPUTensor:
        class Device:
            type = "cpu"
        device = Device()

    class MockGPUTensor:
        class Device:
            type = "cuda"
        device = Device()

    # Valid dict of CPU tensors
    assert_inputs_on_cpu({"input_ids": MockCPUTensor(), "attention_mask": MockCPUTensor()})

    # Dict containing a GPU tensor must raise
    with pytest.raises(RuntimeError, match="CPU Guardrail Violation"):
        assert_inputs_on_cpu({"input_ids": MockCPUTensor(), "attention_mask": MockGPUTensor()})
