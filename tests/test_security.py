"""
Unit tests for security auditing, pickle RCE prevention, zip-slip defense,
and input tensor shape validation.
"""

import io
import pickle
import zipfile
import pytest
from pathlib import Path

from lodevem.security import (
    SecurityError,
    scan_pickle_security,
    validate_input_shape,
    sanitize_display_text,
)


class MaliciousPayload:
    def __reduce__(self):
        import os
        return (os.system, ("echo compromised",))


class SubprocessPayload:
    def __reduce__(self):
        import subprocess
        return (subprocess.Popen, (["echo", "pwned"],))


class EvalPayload:
    def __reduce__(self):
        return (eval, ("1 + 1",))


def test_malicious_pickle_os_system_blocked(tmp_path: Path):
    bad_file = tmp_path / "exploit.pkl"
    with open(bad_file, "wb") as f:
        pickle.dump(MaliciousPayload(), f)

    with pytest.raises(SecurityError, match="Prohibited module '(os|posix)'"):
        scan_pickle_security(bad_file)


def test_malicious_pickle_subprocess_blocked(tmp_path: Path):
    bad_file = tmp_path / "subprocess.joblib"
    with open(bad_file, "wb") as f:
        pickle.dump(SubprocessPayload(), f)

    with pytest.raises(SecurityError, match="Prohibited module 'subprocess'"):
        scan_pickle_security(bad_file)


def test_malicious_pickle_eval_blocked(tmp_path: Path):
    bad_file = tmp_path / "eval.pkl"
    with open(bad_file, "wb") as f:
        pickle.dump(EvalPayload(), f)

    with pytest.raises(SecurityError, match="Prohibited callable 'eval'"):
        scan_pickle_security(bad_file)


def test_pytorch_zip_archive_malicious_pkl_blocked(tmp_path: Path):
    pt_archive = tmp_path / "malicious_model.pt"
    
    # Pack malicious pickle inside a zip archive mimicking PyTorch container
    with zipfile.ZipFile(pt_archive, "w") as zf:
        zf.writestr("archive/data.pkl", pickle.dumps(MaliciousPayload()))
        zf.writestr("archive/version", "3")

    with pytest.raises(SecurityError, match="Prohibited module '(os|posix)'"):
        scan_pickle_security(pt_archive)


def test_zip_slip_path_traversal_blocked(tmp_path: Path):
    bad_zip = tmp_path / "zipslip.pt"
    
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../../etc/passwd", "malicious_content")

    with pytest.raises(SecurityError, match="Zip Slip detected"):
        scan_pickle_security(bad_zip)


def test_safe_pickle_allowed(tmp_path: Path):
    safe_file = tmp_path / "safe.pkl"
    safe_data = {"weights": [1.0, 2.0, 3.0], "bias": 0.5, "name": "linear"}
    with open(safe_file, "wb") as f:
        pickle.dump(safe_data, f)

    # Should not raise any error
    scan_pickle_security(safe_file)


def test_validate_input_shape_valid():
    assert validate_input_shape((1, 3, 224, 224)) == (1, 3, 224, 224)
    assert validate_input_shape([1, 10]) == (1, 10)
    assert validate_input_shape(None) == (1, 3, 224, 224)


def test_validate_input_shape_negative_dim():
    with pytest.raises(ValueError, match="dimensions must be positive integers"):
        validate_input_shape((1, -3, 224))


def test_validate_input_shape_dimension_ceiling():
    with pytest.raises(ValueError, match="exceeds safety ceiling"):
        validate_input_shape((1, 200_000))


def test_validate_input_shape_allocation_bomb():
    # 50,000 * 50,000 = 2,500,000,000 elements (> 500,000,000 limit)
    with pytest.raises(ValueError, match="allocation bomb protection"):
        validate_input_shape((1, 50_000, 50_000))


def test_sanitize_display_text():
    raw = "\x1b[31mRed Text\x1b[0m and \x1b[1mBold\x1b[0m"
    sanitized = sanitize_display_text(raw)
    assert sanitized == "Red Text and Bold"
    assert "\x1b" not in sanitized
