"""
Unit tests for PyTorch model footprint analyzer.
"""

import pytest

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))
else:
    SimpleModel = None


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for this test")
def test_pytorch_footprint_analysis(tmp_path):
    model = SimpleModel()
    model_path = tmp_path / "simple.pt"
    torch.save(model, model_path)

    from lodevem.footprint.analyzers.pytorch import analyze_pytorch_footprint
    from lodevem.security import SecurityPolicyError

    # Without allow_untrusted, full nn.Module pickle must raise SecurityPolicyError
    with pytest.raises(SecurityPolicyError) as exc_info:
        analyze_pytorch_footprint(model_path, input_shape=(1, 10))
    assert "Security Policy Violation" in str(exc_info.value)
    assert "--dangerously-allow-untrusted-model" in str(exc_info.value)

    # With allow_untrusted=True, analysis succeeds
    fp = analyze_pytorch_footprint(model_path, input_shape=(1, 10), allow_untrusted=True)

    assert fp.backend == "pytorch"
    # fc1: 10*20 + 20 = 220; fc2: 20*5 + 5 = 105; total = 325
    assert fp.total_parameters == 325
    assert fp.trainable_parameters == 325
    assert fp.precision_breakdown == {"FP32": 325}
    assert fp.parameter_memory_bytes == 325 * 4
    assert fp.parameter_memory_mb is not None
    assert fp.parameter_memory_mb > 0
    assert fp.estimated_minimum_ram_mb is not None
    assert fp.estimated_minimum_ram_mb > 0

    # FLOPs verification
    assert fp.flops is not None
    assert fp.flops > 0
    assert fp.flops_status == "exact"
    assert fp.macs == fp.flops // 2


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for this test")
def test_pytorch_footprint_state_dict_safe(tmp_path):
    model = SimpleModel()
    state_path = tmp_path / "model_weights.pt"
    torch.save(model.state_dict(), state_path)

    from lodevem.footprint.analyzers.pytorch import analyze_pytorch_footprint

    # Safe state_dict can be loaded via weights_only=True without allow_untrusted
    fp = analyze_pytorch_footprint(state_path, input_shape=(1, 10))
    assert fp.backend == "pytorch"
    assert fp.total_parameters == 325
    assert fp.precision_breakdown == {"FP32": 325}
    assert fp.flops_status == "state_dict_only"
